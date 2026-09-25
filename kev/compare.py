"""Pod-only paired comparison of a Kev export and the English Laya checkpoint."""

import argparse
from collections import Counter
import gc
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from .core import read_rows, require_cuda, softmax, validate_row, write_json
from .evaluate import Metrics, file_sha256

LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
LAYA_SOURCE = "970dc8c5f63d7b886a68409493f37d569424f933"


def question_for(row):
    validate_row(row)
    kind, candidates = row["kind"], row["candidates"]
    if kind == "choice":
        # None makes Laya render exactly the candidate text, without an added key prefix.
        criteria = dict.fromkeys(candidates)
        keys = candidates
    elif kind == "score":
        criteria, keys = candidates, [str(i) for i in range(len(candidates))]
    else:
        criteria = dict(zip(("false", "true"), candidates))
        keys = ["false", "true"]
    return {"type": kind, "instructions": row["instructions"], "criteria": criteria}, keys


def probabilities_for(answer, kind, keys):
    values = ([1 - float(answer["noul"]), float(answer["noul"])] if kind == "noul"
              else [float(answer["probabilities"][key]) for key in keys])
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid prediction probabilities")
    total = sum(values)
    if not math.isclose(total, 1, abs_tol=0.002):
        raise ValueError("Prediction probabilities do not sum to one")
    # Laya's public API rounds probabilities to four decimals.
    return [p / total for p in values]


def sample_rows(path, per_source, seed):
    rng, groups, seen = random.Random(seed), {}, Counter()
    for row in read_rows(path):
        validate_row(row)
        row = {k: row[k] for k in ("state", "instructions", "kind", "candidates", "target", "source", "task")}
        source = row["source"]
        seen[source] += 1
        group = groups.setdefault(source, [])
        if per_source == 0 or len(group) < per_source:
            group.append(row)
        else:
            index = rng.randrange(seen[source])
            if index < per_source:
                group[index] = row
    rows = [row for source in sorted(groups) for row in groups[source]]
    rng.shuffle(rows)
    if not rows:
        raise ValueError("Empty evaluation data")
    return rows


def laya_clips(row, agent, render_options):
    """Mirror the pinned runtime's budgets, detecting any discarded content."""
    question, _ = question_for(row)
    q = {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}
    tok = agent.tok
    encode = lambda text: tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    head = len(encode(f'{q["t"]} question: {q["ins"]}'))
    original_options = [len(encode(" " + option)) for option in render_options(q)]
    options = [1 + min(48, n) for n in original_options]
    budget = agent.cfg.get("head_max_len", 192)
    allowance = budget - sum(options)
    if allowance < 16:
        per = max(4, (budget - 16) // len(options))
        options = [min(n, per) for n in options]
        allowance = budget - sum(options)
    kept_head = min(head, max(8, allowance))
    room = max(0, agent.cfg.get("max_len", 512) - kept_head - sum(options) - 4)
    return (head > kept_head or any(n + 1 > kept for n, kept in zip(original_options, options))
            or len(encode(row["state"])) > room
            or any(tok.mask_token in text for text in [row["state"], row["instructions"], *row["candidates"]]))


def summarize(rows, predictions, clipped):
    groups = {}
    for row, probabilities, clip in zip(rows, predictions, clipped, strict=True):
        keys = ["overall", "source/" + row["source"], "kind/" + row["kind"]]
        if not clip:
            keys.append("shared_untruncated")
        for key in keys:
            groups.setdefault(key, Metrics()).update(row, probabilities)
    return {key: metric.result() for key, metric in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-model", required=True, help="Local final/ export on the pod")
    parser.add_argument("--data", required=True, help="Held-out test.jsonl, never training data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-source", type=int, default=100, help="0 means all rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_source < 0:
        parser.error("--per-source must be nonnegative")
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Output directory already exists; choose a fresh directory")
    os.environ.setdefault("USE_TF", "0")
    device = require_cuda()
    import torch
    import laya
    from huggingface_hub import snapshot_download
    from laya.common import render_options
    from .inference import DecisionModel

    installed = json.loads(distribution("laya").read_text("direct_url.json") or "{}")
    if installed.get("vcs_info", {}).get("commit_id") != LAYA_SOURCE:
        raise RuntimeError(f"Install pinned Laya source with --no-deps: git+https://github.com/NandhaKishorM/laya.git@{LAYA_SOURCE}")
    rows = sample_rows(args.data, args.per_source, args.seed)
    out.mkdir(parents=True)
    with (out / "sample.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"data_sha256": file_sha256(args.data), "sample_sha256": file_sha256(out / "sample.jsonl"),
              "seed": args.seed, "per_source": args.per_source, "count": len(rows),
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "laya_version": laya.__version__, "laya_source_expected": LAYA_SOURCE,
              "laya_revision": LAYA_REVISION, "kev_model": str(Path(args.kev_model).resolve()),
              "notes": ["Paired test of Kev's dataset mixture, not an independent generalization benchmark; Laya training overlap unknown.",
                        "Native shipped calibration and precision; no temperature fitted on test data. Kev had in-domain calibration.",
                        "Laya uses its native input limits. shared_untruncated excludes Laya clipping/mask replacement; Kev rejects overlength inputs.",
                        "Laya API probabilities rounded to 4 decimals and renormalized. Log loss floors zero at 1e-300.",
                        "ECE uses target mass, accuracy accepts any positive target; ordinal MAE is in level indices.",
                        "Latency: one decision/request, three warmups, synchronized GPU, excludes downloads/load/filtering; native runtimes, not equal precision."]}
    all_predictions, clipped = {}, [False] * len(rows)
    for name in ("kev", "laya"):
        if name == "kev":
            model = DecisionModel.from_pretrained(args.kev_model)
            report["kev_temperature"] = model.temperature
            report["kev_config_sha256"] = file_sha256(Path(args.kev_model) / "kev_config.json")
            report["kev_weights_sha256"] = file_sha256(Path(args.kev_model) / "model.safetensors")
            predict = lambda row: softmax(model.score_rows([row])[0], model.temperature)
        else:
            path = snapshot_download("convaiinnovations/laya", revision=LAYA_REVISION,
                                     allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
            model = laya.load(path, device="cuda", fast=False)
            report["laya_config"] = model.cfg
            clipped = [laya_clips(row, model, render_options) for row in rows]
            def predict(row):
                question, keys = question_for(row)
                result = model.predict(row["state"], {"decision": question})
                if model.device.type != "cuda":
                    raise RuntimeError("Laya fell back to CPU; comparison stopped")
                return probabilities_for(result["answers"]["decision"], row["kind"], keys)
        for row in rows[:3]:
            predict(row)
        times, predictions = [], []
        torch.cuda.reset_peak_memory_stats()
        with (out / f"{name}-predictions.jsonl").open("w") as stream:
            for index, row in enumerate(rows):
                torch.cuda.synchronize()
                start = time.perf_counter()
                probabilities = predict(row)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                predictions.append(probabilities)
                stream.write(json.dumps({"index": index, "probabilities": probabilities, "seconds": elapsed}) + "\n")
                if (index + 1) % 100 == 0:
                    print(f"{name}: {index + 1}/{len(rows)}", flush=True)
        report[name] = {"latency_median_ms": statistics.median(times) * 1000,
                        "latency_p95_ms": sorted(times)[math.ceil(len(times) * .95) - 1] * 1000,
                        "decisions_per_second": len(times) / sum(times),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        all_predictions[name] = predictions
        del predict, model
        gc.collect()
        torch.cuda.empty_cache()
    report["laya_altered_input_count"] = sum(clipped)
    report["laya_altered_input_by_source"] = dict(Counter(row["source"] for row, clip in zip(rows, clipped) if clip))
    for name in all_predictions:
        report[name]["metrics"] = summarize(rows, all_predictions[name], clipped)
    write_json(out / "comparison.json", report)
    print(json.dumps({name: report[name]["metrics"]["overall"] for name in all_predictions}, indent=2))
    print(f"Full report: {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
