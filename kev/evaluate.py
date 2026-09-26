"""Evaluate an exported Kev checkpoint on a prepared JSONL partition, on the pod."""

import argparse
import hashlib
import math
from pathlib import Path

from .core import softmax, validate_row, write_json
from .inference import DecisionModel, add_inference_arguments, inference_kwargs


class Metrics:
    def __init__(self):
        self.count = self.correct = self.nll = self.brier = 0
        self.score_count = self.score_error = 0
        self.bins = [[0, 0.0, 0.0] for _ in range(15)]

    def update(self, row, probabilities):
        validate_row(row, max_candidates=len(row["candidates"]))
        target = row["target"]
        if len(target) != len(probabilities):
            raise ValueError("Prediction and target lengths differ")
        top = max(range(len(probabilities)), key=probabilities.__getitem__)
        self.count += 1
        self.correct += target[top] > 0
        self.nll -= sum(y * math.log(max(p, 1e-300)) for y, p in zip(target, probabilities, strict=True))
        self.brier += sum((p - y) ** 2 for p, y in zip(probabilities, target, strict=True))
        bucket = self.bins[min(14, int(probabilities[top] * 15))]
        bucket[0] += 1
        bucket[1] += probabilities[top]
        bucket[2] += target[top]
        if row["kind"] == "score":
            self.score_count += 1
            self.score_error += abs(sum(i * (p - y) for i, (p, y) in
                                        enumerate(zip(probabilities, target, strict=True))))

    def result(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty partition")
        return {"count": self.count, "accuracy": self.correct / self.count,
                "log_loss": self.nll / self.count, "brier_score": self.brier / self.count,
                "ece_15": sum(abs(confidence - gold) for n, confidence, gold in self.bins if n)
                          / self.count,
                "ordinal_mae": self.score_error / self.score_count if self.score_count else None,
                "ordinal_count": self.score_count}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(scored_rows, temperature):
    stages = {"uncalibrated": {}, "calibrated": {}}
    for row, logits in scored_rows:
        for name, temp in (("uncalibrated", 1.0), ("calibrated", temperature)):
            probabilities = softmax(logits, temp)
            # Source, kind and task reports avoid hiding a small source in the global mean.
            keys = ["overall", "source/" + row["source"], "kind/" + row["kind"],
                    "task/" + row["source"] + "/" + row["task"]]
            for key in keys:
                stages[name].setdefault(key, Metrics()).update(row, probabilities)
    if not stages["uncalibrated"]:
        raise ValueError("Evaluation partition is empty")
    return {stage: {key: metric.result() for key, metric in groups.items()}
            for stage, groups in stages.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    add_inference_arguments(parser)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Report already exists; choose another --output path")
    model = DecisionModel.from_pretrained(args.model, **inference_kwargs(args))
    report = evaluate(model.iter_scores(args.data), model.temperature)
    report.update({"temperature": model.temperature, "data_sha256": file_sha256(args.data),
                   "inference": {**inference_kwargs(args), "weight_dtype": model.weight_dtype},
                   "checkpoint": str(Path(args.model).resolve()),
                   "notes": "Accuracy accepts any positive target. ECE uses gold probability mass, "
                            "not any-positive accuracy. Ordinal MAE uses zero-based level indices."})
    write_json(args.output, report)
    print(f"Wrote {args.output}")  # noqa: T201 - CLI result


if __name__ == "__main__":
    main()
