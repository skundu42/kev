"""Benchmark a local export's CUDA inference; retain errors and never refit calibration."""

import argparse
import gc
import math
import statistics
import time
from importlib.metadata import version
from itertools import islice, product
from pathlib import Path

from .core import load_config, parse_questions, read_rows, require_cuda, softmax, validate_row, write_json
from .evaluate import Metrics, file_sha256
from .inference import COMPILE_MODES, DEFAULT_MAX_BATCH_TOKENS, DEFAULT_PAD_MULTIPLE, DecisionModel


def configurations(batch_sizes=(8, 16, 32), request_batch_sizes=(1, 4), *,
                   max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS, pad_to_multiple_of=DEFAULT_PAD_MULTIPLE, compile_model=False,
                   compile_mode="default", flash_attention=False):
    """Keep a fixed reference and bound expensive backend experiments to the largest batch."""
    if not batch_sizes or not request_batch_sizes or min(*batch_sizes, *request_batch_sizes) < 1:
        raise ValueError("Batch sizes must be positive")
    if max_batch_tokens < 1 or pad_to_multiple_of < 1:
        raise ValueError("Token budget and padding multiple must be positive")
    baseline = {"name": "baseline", "pair_batch_size": 8, "max_batch_tokens": None,
                    "group_by_length": False, "weight_dtype": "float32", "attn_implementation": "sdpa",
                    "compile_model": False, "compile_mode": compile_mode, "pad_to_multiple_of": 1,
                    "requests_per_batch": 1}
    result = [baseline]
    for size, dtype, combined in product(dict.fromkeys(batch_sizes), ("float32", "bfloat16"),
                                         dict.fromkeys(request_batch_sizes)):
        for backend, compiled in [("sdpa", False), ("sdpa", True),
                                  ("flash_attention_2", False), ("flash_attention_2", True)]:
            if compiled and not compile_model:
                continue
            if backend == "flash_attention_2" and (not flash_attention or dtype != "bfloat16"):
                continue
            if (compiled or backend != "sdpa") and size != max(batch_sizes):
                continue
            name = f"pairs{size}-{dtype}-{backend}-requests{combined}"
            result.append(dict(baseline, name=name + ("-compiled" if compiled else ""),
                               pair_batch_size=size, max_batch_tokens=max_batch_tokens,
                               group_by_length=True, weight_dtype=dtype,
                               attn_implementation=backend, compile_model=compiled,
                               pad_to_multiple_of=pad_to_multiple_of, requests_per_batch=combined))
    return [result[0], *sorted(result[1:], key=lambda options: options["compile_model"])]


def load_workload(path, *, request_input=False, decisions_per_request=5,
                  max_requests=100, max_candidates=16):
    """Read a stable prefix; group prepared decisions without changing their text."""
    if decisions_per_request < 1 or max_requests < 0:
        raise ValueError("decisions_per_request must be positive and max_requests nonnegative")
    records = read_rows(path)
    if request_input:
        requests = []
        for request in islice(records, max_requests or None):
            if not isinstance(request, dict) or "state" not in request or "questions" not in request:
                raise ValueError("Each request JSONL record must contain state and questions")
            rows, _ = parse_questions(request["state"], request["questions"], max_candidates)
            requests.append(rows)
    else:
        rows = []
        for row in islice(records, max_requests * decisions_per_request if max_requests else None):
            if not isinstance(row, dict) or "target" not in row:
                raise ValueError("--data requires prepared rows with target distributions")
            validate_row(row, max_candidates)
            rows.append(dict(row, kind=row.get("kind", "choice")))
        requests = [rows[start:start + decisions_per_request]
                    for start in range(0, len(rows), decisions_per_request)]
    if not requests:
        raise ValueError("Benchmark input is empty")
    return requests


def probability_delta(reference, actual):
    total = maximum = variation = changes = count = 0
    if not reference:
        raise ValueError("Reference probabilities are empty")
    for before, after in zip(reference, actual, strict=True):
        differences = [abs(a - b) for a, b in zip(before, after, strict=True)]
        maximum = max(maximum, max(differences))
        total += sum(differences)
        variation += sum(differences) / 2
        count += len(differences)
        changes += max(range(len(before)), key=before.__getitem__) != max(
            range(len(after)), key=after.__getitem__)
    return {"max_abs": maximum, "mean_abs": total / count,
                "mean_total_variation": variation / len(reference), "argmax_changes": changes,
                "argmax_change_rate": changes / len(reference)}


def calibration_metrics(rows, probabilities):
    metrics = Metrics()
    for row, values in zip(rows, probabilities, strict=True):
        if "target" in row:
            metrics.update(row, values)
    return metrics.result() if metrics.count else None


def benchmark_model(model, workload, requests_per_batch, cuda, *, warmup_passes=1,
                    repeats=3, clock=time.perf_counter):
    """Synchronize each timed call; the first full pass covers all encountered shapes."""
    if requests_per_batch < 1 or warmup_passes < 0 or repeats < 1:
        raise ValueError("Invalid benchmark batching, warmup or repeat count")
    if not workload or any(not request for request in workload):
        raise ValueError("Benchmark requests must be nonempty")
    batches = [[row for request in workload[start:start + requests_per_batch] for row in request]
               for start in range(0, len(workload), requests_per_batch)]

    def run_pass(capture=False):
        elapsed, predictions = [], []
        for rows in batches:
            cuda.synchronize(model.device)
            start = clock()
            logits = model.score_rows(rows)
            cuda.synchronize(model.device)
            elapsed.append(clock() - start)
            if len(logits) != len(rows) or any(len(values) != len(row["candidates"])
                                              for row, values in zip(rows, logits, strict=True)):
                raise ValueError("Model returned the wrong number of scores")
            if capture:
                predictions.extend(logits)
        return elapsed, predictions

    cuda.reset_peak_memory_stats(model.device)
    cold, _ = run_pass()
    warmup = []
    for _ in range(warmup_passes):
        elapsed, _ = run_pass()
        warmup.extend(elapsed)
    startup_peak = cuda.max_memory_allocated(model.device)
    cuda.reset_peak_memory_stats(model.device)
    times, logits = [], []
    for repeat in range(repeats):
        elapsed, captured = run_pass(capture=repeat == 0)
        times.extend(elapsed)
        if repeat == 0:
            logits = captured
    total = sum(times)
    if total <= 0:
        raise ValueError("Benchmark clock did not advance")
    rows = [row for request in workload for row in request]
    report = {"first_call_ms": cold[0] * 1000, "first_pass_ms": sum(cold) * 1000,
                  "additional_warmup_ms": sum(warmup) * 1000, "warmup_passes": warmup_passes,
                  "repeats": repeats, "batches_per_pass": len(batches), "measured_batches": len(times),
                  "measured_seconds": total,
                  "combined_batch_latency_ms": {"median": statistics.median(times) * 1000,
                      "p95": sorted(times)[math.ceil(len(times) * .95) - 1] * 1000},
                  "amortized_ms_per_request": total * 1000 / (len(workload) * repeats),
                  "requests_per_second": len(workload) * repeats / total,
                  "decisions_per_second": len(rows) * repeats / total,
                  "candidate_pairs_per_second": sum(len(row["candidates"]) for row in rows) * repeats / total,
                  "startup_peak_allocated_gib": startup_peak / 2**30,
                  "steady_peak_allocated_gib": cuda.max_memory_allocated(model.device) / 2**30,
                  "steady_peak_reserved_gib": cuda.max_memory_reserved(model.device) / 2**30}
    return report, logits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local final/ export; no downloads")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL: one {state, questions} request per line")
    source.add_argument("--data", help="Prepared labelled JSONL; no calibration is fitted")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--request-batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--max-batch-tokens", type=int, default=DEFAULT_MAX_BATCH_TOKENS)
    parser.add_argument("--pad-to-multiple-of", type=int, default=DEFAULT_PAD_MULTIPLE)
    parser.add_argument("--decisions-per-request", type=int, default=5)
    parser.add_argument("--max-requests", type=int, default=100, help="Stable input prefix; 0 selects all")
    parser.add_argument("--warmup-passes", type=int, default=1, help="Extra full passes after the first pass")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="Also compile the largest pair batch")
    parser.add_argument("--compile-mode", default="default",
                        choices=COMPILE_MODES)
    parser.add_argument("--flash-attention", action="store_true", help="Also test BF16 FlashAttention 2")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Report already exists; choose a fresh --output path")
    if args.warmup_passes < 0 or args.repeats < 1:
        parser.error("Warmup passes must be nonnegative and repeats positive")
    path = Path(args.model).resolve()
    if not (path / "kev_config.json").is_file():
        parser.error("--model must be a local Kev export containing kev_config.json")
    config = load_config(path / "kev_config.json")
    try:
        variants = configurations(args.batch_sizes, args.request_batch_sizes,
            max_batch_tokens=args.max_batch_tokens, pad_to_multiple_of=args.pad_to_multiple_of,
            compile_model=args.compile, compile_mode=args.compile_mode, flash_attention=args.flash_attention)
        workload = load_workload(args.input or args.data, request_input=bool(args.input),
            decisions_per_request=args.decisions_per_request, max_requests=args.max_requests,
            max_candidates=config["max_candidates"])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    device = require_cuda()
    import torch  # noqa: PLC0415 - optional GPU runtime, absent from the offline test environment

    rows = [row for request in workload for row in request]
    results = [{"options": options, "status": "pending"} for options in variants]
    report = {"checkpoint": str(path), "kev_config_sha256": file_sha256(path / "kev_config.json"),
                  "data_sha256": file_sha256(args.input or args.data), "input_kind": "requests" if args.input else "prepared_rows",
                  "requests": len(workload), "decisions": len(rows), "labelled_decisions": sum("target" in row for row in rows),
                  "candidate_pairs": sum(len(row["candidates"]) for row in rows),
                  "max_requests": args.max_requests, "decisions_per_request": args.decisions_per_request,
                  "temperature": float(config.get("temperature", 1)), "gpu": torch.cuda.get_device_name(device),
                  "torch": torch.__version__, "transformers": version("transformers"), "cuda": torch.version.cuda,
                  "results": results, "notes": [
                      "Baseline uses this runtime with eight pairs, FP32/SDPA, no length grouping and exact padding; it includes consolidated score transfer and is not the old implementation.",
                      "Timing includes score_rows tokenization, GPU execution and CPU scores; excludes HTTP queueing/network.",
                      "Combined batch latency is shared by its requests; amortized latency is not individual request latency.",
                      "Every configuration uses identical rows in input order and the export's unchanged temperature.",
                      "The first full pass covers encountered shapes; further compilation during measurement is included.",
                      "Compiler state is reset between configurations; on-disk kernel caches and CUDA context remain warm.",
                      "Peak memory includes weights. FP32/BF16 labels describe stored weights, retaining CUDA autocast policy.",
                      "Probability/calibration comparisons use the first measured pass.",
                      "ECE uses top-candidate target mass; accuracy accepts any positive target. No test calibration is fitted."]}
    baseline = None
    write_json(args.output, report)
    for result in results:
        options, model = result["options"], None
        print(f"Benchmarking {options['name']}", flush=True)  # noqa: T201
        result["status"] = "running"
        write_json(args.output, report)
        try:
            if options["compile_model"]:
                from torch._dynamo.utils import counters  # noqa: PLC0415 - optional compiler diagnostics
                counters.clear()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            model = DecisionModel.from_pretrained(path, **{key: value for key, value in options.items()
                                                          if key not in {"name", "requests_per_batch"}})
            torch.cuda.synchronize(device)
            result["load_seconds"] = time.perf_counter() - start
            result["load_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            result["resolved_weight_dtype"] = str(model.weight_dtype)
            timing, logits = benchmark_model(model, workload, options["requests_per_batch"], torch.cuda,
                                            warmup_passes=args.warmup_passes, repeats=args.repeats)
            probabilities = [softmax(values, model.temperature) for values in logits]
            reference_probabilities = probabilities if baseline is None else baseline
            result.update(status="ok", timing=timing,
                          probability_delta=probability_delta(reference_probabilities, probabilities),
                          calibrated=calibration_metrics(rows, probabilities),
                          uncalibrated=calibration_metrics(rows, [softmax(values) for values in logits]))
            if options["compile_model"]:
                result["compiler"] = {"unique_graphs": counters["stats"]["unique_graphs"],
                                      "graph_breaks": dict(counters["graph_break"]),
                                      "unimplemented": dict(counters["unimplemented"])}
            reference = results[0].get("calibrated")
            if reference:
                result["calibration_delta"] = {key: result["calibrated"][key] - reference[key]
                    for key in ("accuracy", "log_loss", "brier_score", "ece_15", "ordinal_mae")
                    if reference[key] is not None}
            if baseline is None:
                baseline = probabilities
        except Exception as error:
            result.update(status="error", error={"type": type(error).__name__, "message": str(error)})
            print(f"{options['name']}: {type(error).__name__}: {error}", flush=True)  # noqa: T201
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            write_json(args.output, report)
            return 130
        finally:
            write_json(args.output, report)
            del model
            gc.collect()
            torch.compiler.reset()
            torch.cuda.empty_cache()
        if baseline is None:
            for pending in results[1:]:
                pending.update(status="skipped", reason="The FP32 reference failed")
            write_json(args.output, report)
            break
        write_json(args.output, report)
    print(f"Wrote {args.output}")  # noqa: T201
    return int(any(result["status"] != "ok" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
