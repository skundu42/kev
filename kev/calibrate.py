"""Fit one temperature on the calibration partition; never on final evaluation data."""

import argparse
import math
from pathlib import Path

from .core import load_config, write_json
from .evaluate import file_sha256


def log_loss(examples, temperature):
    if not examples:
        raise ValueError("Calibration partition is empty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    total = 0.0
    for logits, target in examples:
        if len(logits) != len(target) or not logits:
            raise ValueError("Invalid calibration example")
        peak = max(logits)
        scaled = [(x - peak) / temperature for x in logits]
        normalizer = math.log(sum(math.exp(x) for x in scaled))
        total += normalizer - sum(y * x for y, x in zip(target, scaled))
    return total / len(examples)


def fit_temperature(examples):
    if not examples:
        raise ValueError("Calibration partition is empty")
    for logits, target in examples:
        if (len(logits) < 2 or len(logits) != len(target)
                or any(not math.isfinite(x) for x in logits)
                or any(not math.isfinite(y) or y < 0 for y in target)
                or not math.isclose(sum(target), 1.0, abs_tol=1e-6)):
            raise ValueError("Calibration requires finite logits and normalized target distributions")
    # One bounded scalar optimization needs no optimizer dependency.
    low, high = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    fleft, fright = log_loss(examples, math.exp(left)), log_loss(examples, math.exp(right))
    for _ in range(64):
        if fleft < fright:
            high, right, fright = right, left, fleft
            left = high - ratio * (high - low)
            fleft = log_loss(examples, math.exp(left))
        else:
            low, left, fleft = left, right, fright
            right = low + ratio * (high - low)
            fright = log_loss(examples, math.exp(right))
    candidates = [1.0, 0.05, 20.0, math.exp((low + high) / 2)]
    return min(candidates, key=lambda t: log_loss(examples, t))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    if Path(args.data).name != "calibration.jsonl":
        parser.error("Use the prepared calibration.jsonl partition, not validation or test")
    from .inference import DecisionModel

    model = DecisionModel.from_pretrained(args.model)
    examples = [(logits, row["target"]) for row, logits in model.iter_scores(args.data)]
    temperature = fit_temperature(examples)
    report = {"temperature": temperature, "count": len(examples),
              "temperature_bounds": [0.05, 20.0],
              "log_loss_before": log_loss(examples, 1.0),
              "log_loss_after": log_loss(examples, temperature),
              "data_sha256": file_sha256(args.data)}
    path = Path(args.model) / "kev_config.json"
    config = load_config(path)
    config["temperature"] = temperature
    config["calibration"] = report
    write_json(path, config)
    write_json(Path(args.model) / "calibration.json", report)
    print(f"Temperature={temperature:.6g}; calibration NLL "
          f"{report['log_loss_before']:.6g} -> {report['log_loss_after']:.6g}")


if __name__ == "__main__":
    main()
