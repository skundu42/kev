"""Pod-only, isolated end-to-end readiness check using already prepared examples."""

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from kev.core import load_config, read_rows, require_cuda, write_json
from kev.hub import SPLITS, validate_training_data


ROOT = Path(__file__).resolve().parents[1]


def check_idle_gpu():
    """Do not compete with an existing long training run for GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot verify the GPU is idle with nvidia-smi; resolve this before check-run") from error
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError("Cannot interpret nvidia-smi compute-process output; inspect nvidia-smi before check-run")
    active = sorted({int(line) for line in lines} - {os.getpid()})
    if active:
        raise RuntimeError(f"GPU compute processes are already active (PIDs {active}). "
                           "Run readiness on an idle pod; do not run it alongside long training.")


def make_sample(data_dir, output_dir, config, manifest):
    """Cover actual maximum padded lengths for each candidate count over three passes."""
    if config.get("dataloader_drop_last", False):
        raise ValueError("Readiness requires dataloader_drop_last=false to cover every selected row")
    batch = config.get("per_device_train_batch_size", 1)
    accumulation = config.get("gradient_accumulation_steps", 1)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (batch, accumulation)):
        raise ValueError("Training batch size and accumulation must be positive integers")
    # Hold only a small prefix and at most one extreme per candidate count in RAM.
    prefixes, extremes = {}, {}
    for split in SPLITS:
        prefix = []
        for row in read_rows(Path(data_dir) / f"{split}.jsonl"):
            if len(prefix) < (128 if split == "train" else 16):
                prefix.append(row)
            if split in {"train", "validation"}:
                count = len(row["input_ids"])
                length = max(map(len, row["input_ids"]))
                if count not in extremes or length > extremes[count][0]:
                    extremes[count] = (length, row, split)
            elif len(prefix) == 16:
                break
        if not prefix:
            raise ValueError(f"Prepared {split} partition is empty")
        prefixes[split] = prefix
    effective_batch = batch * accumulation
    # Repeat a complete, small epoch so extreme shapes run again after Adam's
    # persistent optimizer state has been allocated by the first update.
    steps_per_epoch = max(1, math.ceil(len(extremes) / effective_batch))
    capacity = steps_per_epoch * effective_batch
    steps = 3 * steps_per_epoch
    selected = [extremes[count][1] for count in sorted(extremes)]
    selected.extend(itertools.islice(itertools.cycle(prefixes["train"]), capacity - len(selected)))
    prefixes["train"] = selected
    output = Path(output_dir)
    output.mkdir()
    for split, rows in prefixes.items():
        with (output / f"{split}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    sample_manifest = {**manifest, "counts": {key: len(rows) for key, rows in prefixes.items()},
                       "readiness_only": True,
                       "readiness_source": str(Path(data_dir).resolve())}
    write_json(output / "manifest.json", sample_manifest)
    small_config = {**config, "max_steps": steps, "eval_steps": 1, "save_steps": 1,
                    "logging_steps": 1, "save_total_limit": 2, "report_to": "none"}
    shapes = [{"candidates": count, "padded_length": entry[0], "source_split": entry[2]}
              for count, entry in sorted(extremes.items())]
    return small_config, shapes, sample_manifest["counts"]


def run_check(config_path, data_dir, output_dir):
    config = load_config(config_path)
    manifest = validate_training_data(data_dir, config)
    output = Path(output_dir).resolve()
    # Never mix temporary readiness checkpoints with a real training run.
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "data_dir": str(Path(data_dir).resolve()),
              "timings_seconds": {}, "note": "Readiness metrics are not model-quality measurements."}
    report_path = output / "readiness.json"
    write_json(report_path, report)

    def invoke(stage, arguments, capture=False):
        print(f"Readiness: {stage}", flush=True)
        started = time.monotonic()
        result = subprocess.run([sys.executable, *map(str, arguments)], check=True, cwd=ROOT,
                                text=True, stdout=subprocess.PIPE if capture else None)
        report["timings_seconds"][stage] = round(time.monotonic() - started, 3)
        return result

    try:
        sample_dir = output / "data"
        small_config, shapes, counts = make_sample(data_dir, sample_dir, config, manifest)
        config_file = output / "config.yaml"
        write_json(config_file, small_config)
        report.update({"counts": counts, "optimizer_steps": small_config["max_steps"],
                       "complete_passes": 3, "extreme_shapes": shapes,
                       "shape_coverage": "Every selected extreme row is trained in three complete passes, "
                                         "including after optimizer-state allocation. With batch size >1, "
                                         "this does not test every possible mixed-batch padding combination."})
        run_dir = output / "run"
        invoke("real_training", ["-m", "src.cli", "launch", "kev", config_file, "--root", ROOT,
                                  "--data-dir", sample_dir, "--output-dir", run_dir])
        target_step = small_config["max_steps"]
        if load_config(run_dir / "trainer_state.json")["global_step"] != target_step:
            raise ValueError("Readiness training did not finish its configured optimizer steps")
        report["training_metrics"] = load_config(run_dir / "train_results.json")
        report["gpu_memory"] = load_config(run_dir / "gpu_memory.json")
        resumable = []
        for checkpoint in run_dir.glob("checkpoint-*"):
            if all((checkpoint / name).is_file() for name in
                   ("trainer_state.json", "optimizer.pt", "scheduler.pt")):
                step = load_config(checkpoint / "trainer_state.json")["global_step"]
                if 0 < step < target_step:
                    resumable.append((step, checkpoint))
        if not resumable:
            raise ValueError("Readiness requires an earlier retained checkpoint with optimizer and scheduler state")
        resumed_step, checkpoint = max(resumable, key=lambda item: item[0])
        final = run_dir / "final"
        # Only this dedicated check directory is modified; preserve the first export.
        final.rename(output / "first-export")
        invoke("resume", ["-m", "src.cli", "launch", "kev", config_file, "--root", ROOT,
                          "--data-dir", sample_dir, "--output-dir", run_dir,
                          "--resume-from-checkpoint", checkpoint])
        if load_config(run_dir / "trainer_state.json")["global_step"] != target_step:
            raise ValueError("Readiness checkpoint resume did not reach the configured optimizer steps")
        report["resumed_from_step"] = resumed_step
        invoke("calibration", ["-m", "kev.calibrate", "--model", final,
                                "--data", sample_dir / "calibration.jsonl"])
        invoke("evaluation", ["-m", "kev.evaluate", "--model", final,
                               "--data", sample_dir / "test.jsonl", "--output", run_dir / "evaluation.json"])
        result = invoke("prediction", ["-m", "kev.inference", "--model", final,
                                       "--input", ROOT / "examples/request.json"], capture=True)
        predictions = json.loads(result.stdout)
        kinds = {answer["type"] for answer in predictions["answers"].values()}
        if kinds != {"choice", "noul", "score"}:
            raise ValueError("Readiness prediction did not return all three decision types")
        write_json(output / "predictions.json", predictions)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(report_path, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Fresh, dedicated readiness directory")
    args = parser.parse_args()
    check_idle_gpu()
    require_cuda()
    run_check(args.config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
