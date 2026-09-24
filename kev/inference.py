"""Pod-only checkpoint inference. Importing this module performs no downloads."""

import argparse
import json
import sys
from pathlib import Path

from .core import (answer, load_config, parse_questions, read_rows, require_cuda,
                   softmax, tokenize_row)


class DecisionModel:
    def __init__(self, model, tokenizer, config, device, pair_batch_size=8):
        if pair_batch_size < 1:
            raise ValueError("pair_batch_size must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pair_batch_size = pair_batch_size
        self.temperature = float(config.get("temperature", 1.0))
        softmax([0.0, 0.0], self.temperature)

    @classmethod
    def from_pretrained(cls, path, pair_batch_size=8):
        device = require_cuda()
        # Inference only accepts an exported checkpoint; it cannot fetch a model ID.
        path = Path(path).resolve()
        if not (path / "kev_config.json").is_file():
            raise ValueError(f"{path} is not a Kev export (missing kev_config.json)")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = load_config(path / "kev_config.json")
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            attn_implementation="sdpa", dtype=torch.float32,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False)
        if model.config.num_labels != 1:
            raise ValueError("Expected an exported single-score candidate model")
        return cls(model, tokenizer, config, device, pair_batch_size)

    def score_rows(self, rows):
        import torch

        flat, counts = [], []
        for row in rows:
            encoded = tokenize_row(row, self.tokenizer, self.config["max_length"],
                                   self.config["max_candidates"])
            counts.append(len(row["candidates"]))
            # Targets never enter the single-output regression head.
            flat.extend({key: encoded[key][i] for key in ("input_ids", "attention_mask")}
                        for i in range(len(row["candidates"])))
        values = []
        with torch.inference_mode():
            for start in range(0, len(flat), self.pair_batch_size):
                batch = self.tokenizer.pad(flat[start:start + self.pair_batch_size],
                                           padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_bf16_supported()):
                    scores = self.model(**batch).logits.squeeze(-1).float()
                values.extend(scores.cpu().tolist())
        result, offset = [], 0
        for count in counts:
            result.append(values[offset:offset + count])
            offset += count
        return result

    def predict(self, state, questions):
        rows, metadata = parse_questions(state, questions, self.config["max_candidates"])
        answers = {}
        for row, (question_id, keys), logits in zip(rows, metadata, self.score_rows(rows), strict=True):
            answers[question_id] = answer(row["kind"], row["candidates"], keys,
                                          softmax(logits, self.temperature))
        return {"answers": answers}

    def iter_scores(self, path, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batch = []
        for row in read_rows(path):
            batch.append(row)
            if len(batch) == batch_size:
                yield from zip(batch, self.score_rows(batch), strict=True)
                batch = []
        if batch:
            yield from zip(batch, self.score_rows(batch), strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="-", help="JSON request file, or - for stdin")
    parser.add_argument("--pair-batch-size", type=int, default=8)
    args = parser.parse_args()
    request = json.load(sys.stdin) if args.input == "-" else load_config(args.input)
    if "state" not in request or "questions" not in request:
        parser.error("Request must contain state and questions")
    model = DecisionModel.from_pretrained(args.model, args.pair_batch_size)
    print(json.dumps(model.predict(request["state"], request["questions"]),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
