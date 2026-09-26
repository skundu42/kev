"""Pod-only checkpoint inference. Importing this module performs no downloads."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .core import answer, load_config, parse_questions, read_rows, require_cuda, softmax, tokenize_row

DEFAULT_PAIR_BATCH_SIZE = 8
DEFAULT_MAX_BATCH_TOKENS = 8192
DEFAULT_PAD_MULTIPLE = 1
WEIGHT_DTYPES = ("float32", "bfloat16", "auto")
ATTENTION_BACKENDS = ("sdpa", "flash_attention_2")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")


def validate_batch_options(pair_batch_size, max_batch_tokens, pad_to_multiple_of):
    for name, value in (("pair_batch_size", pair_batch_size),
                        ("pad_to_multiple_of", pad_to_multiple_of)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_batch_tokens is not None and (isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int) or max_batch_tokens < 1):
        raise ValueError("max_batch_tokens must be a positive integer or None")


def plan_batches(lengths, pair_batch_size, max_batch_tokens, max_length,
                 pad_to_multiple_of, group_by_length=True, fixed_batch_shapes=False):
    """Bound actual padded tokens, including duplicated rows used for compilation."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__) if group_by_length else range(len(lengths))
    indices, width, padded_count = [], 0, 0
    for index in order:
        length = min(max_length, ((lengths[index] + pad_to_multiple_of - 1)
                                 // pad_to_multiple_of) * pad_to_multiple_of)
        next_width = max(width, length)
        count = len(indices) + 1
        next_count = min(pair_batch_size, 1 << (count - 1).bit_length()) if fixed_batch_shapes else count
        if indices and (count > pair_batch_size or (max_batch_tokens is not None
                                                   and next_count * next_width > max_batch_tokens)):
            yield indices, width, padded_count
            indices, next_width, next_count = [], length, 1
        if max_batch_tokens is not None and next_width * next_count > max_batch_tokens:
            raise ValueError("max_batch_tokens cannot fit one padded candidate pair")
        indices.append(index)
        width, padded_count = next_width, next_count
    if indices:
        yield indices, width, padded_count


@dataclass
class PreparedRequest:
    rows: list
    metadata: list
    pairs: list
    counts: list


class DecisionModel:
    def __init__(self, model, tokenizer, config, device, pair_batch_size=DEFAULT_PAIR_BATCH_SIZE,
                 *, max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS, group_by_length=True,
                 pad_to_multiple_of=DEFAULT_PAD_MULTIPLE, compile_model=False):
        validate_batch_options(pair_batch_size, max_batch_tokens, pad_to_multiple_of)
        if max_batch_tokens is not None and max_batch_tokens < config["max_length"]:
            raise ValueError("max_batch_tokens must be at least the export's max_length")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pair_batch_size = pair_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.group_by_length = group_by_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.compile_model = compile_model
        self.temperature = float(config.get("temperature", 1.0))
        softmax([0.0, 0.0], self.temperature)

    @classmethod
    def from_pretrained(cls, path, pair_batch_size=DEFAULT_PAIR_BATCH_SIZE, *,
                        max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS, group_by_length=True,
                        weight_dtype="float32", attn_implementation="sdpa", compile_model=False,
                        compile_mode="default", pad_to_multiple_of=DEFAULT_PAD_MULTIPLE):
        validate_batch_options(pair_batch_size, max_batch_tokens, pad_to_multiple_of)
        if weight_dtype not in WEIGHT_DTYPES:
            raise ValueError(f"weight_dtype must be one of {WEIGHT_DTYPES}")
        if attn_implementation not in ATTENTION_BACKENDS:
            raise ValueError(f"attn_implementation must be one of {ATTENTION_BACKENDS}")
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
        device = require_cuda()
        # Inference only accepts an exported checkpoint; it cannot fetch a model ID.
        path = Path(path).resolve()
        if not (path / "kev_config.json").is_file():
            raise ValueError(f"{path} is not a Kev export (missing kev_config.json)")
        import torch  # noqa: PLC0415 - optional pod dependencies
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        config = load_config(path / "kev_config.json")
        if max_batch_tokens is not None and max_batch_tokens < config["max_length"]:
            raise ValueError("max_batch_tokens must be at least the export's max_length")
        bf16_supported = torch.cuda.is_bf16_supported()
        if weight_dtype == "auto":
            weight_dtype = "bfloat16" if bf16_supported else "float32"
        if weight_dtype == "bfloat16" and not bf16_supported:
            raise ValueError("bfloat16 weights require a CUDA device with BF16 support")
        if attn_implementation == "flash_attention_2" and weight_dtype != "bfloat16":
            raise ValueError("flash_attention_2 requires --weight-dtype bfloat16 (or supported auto)")
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            attn_implementation=attn_implementation, dtype=getattr(torch, weight_dtype),
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False)
        if model.config.num_labels != 1:
            raise ValueError("Expected an exported single-score candidate model")
        model.eval()
        if compile_model:
            model = torch.compile(model, mode=compile_mode, dynamic=False)
        runtime = cls(model, tokenizer, config, device, pair_batch_size,
                      max_batch_tokens=max_batch_tokens, group_by_length=group_by_length,
                      pad_to_multiple_of=pad_to_multiple_of, compile_model=compile_model)
        runtime.weight_dtype = weight_dtype
        runtime.attn_implementation = attn_implementation
        return runtime

    def _encode_rows(self, rows):
        flat, counts = [], []
        for row in rows:
            encoded = tokenize_row(row, self.tokenizer, self.config["max_length"],
                                   self.config["max_candidates"])
            counts.append(len(row["candidates"]))
            # Targets never enter the single-output regression head.
            flat.extend({key: encoded[key][i] for key in ("input_ids", "attention_mask")}
                        for i in range(len(row["candidates"])))
        return flat, counts

    def _score_pairs(self, flat, counts):
        if not flat:
            return []
        import torch  # noqa: PLC0415 - optional pod dependency

        order, tensors = [], []
        batches = plan_batches([len(pair["input_ids"]) for pair in flat], self.pair_batch_size,
                               self.max_batch_tokens, self.config["max_length"],
                               self.pad_to_multiple_of, self.group_by_length, self.compile_model)
        with torch.inference_mode():
            for indices, width, padded_count in batches:
                pairs = [flat[index] for index in indices]
                # Duplicating a valid pair avoids all-masked rows in attention kernels.
                pairs.extend([pairs[-1]] * (padded_count - len(pairs)))
                batch = self.tokenizer.pad(pairs, padding="max_length", max_length=width,
                                           return_tensors="pt")
                batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_bf16_supported()):
                    scores = self.model(**batch).logits.squeeze(-1).float()
                # Compiled CUDA graphs can reuse output storage on the next forward.
                tensors.append(scores[:len(indices)].clone())
                order.extend(indices)
            # One device-to-host synchronization for the entire request group.
            packed = torch.cat(tensors).cpu().tolist()
        values = [0.0] * len(flat)
        for index, value in zip(order, packed, strict=True):
            values[index] = value
        result, offset = [], 0
        for count in counts:
            result.append(values[offset:offset + count])
            offset += count
        return result

    def score_rows(self, rows):
        return self._score_pairs(*self._encode_rows(rows))

    def prepare_request(self, state, questions):
        rows, metadata = parse_questions(state, questions, self.config["max_candidates"])
        flat, counts = self._encode_rows(rows)
        return PreparedRequest(rows, metadata, flat, counts)

    def predict_prepared(self, requests):
        pairs = [pair for request in requests for pair in request.pairs]
        counts = [count for request in requests for count in request.counts]
        scores = iter(self._score_pairs(pairs, counts))
        results = []
        for request in requests:
            answers = {}
            for row, (question_id, keys) in zip(request.rows, request.metadata, strict=True):
                answers[question_id] = answer(row["kind"], row["candidates"], keys,
                                              softmax(next(scores), self.temperature))
            results.append({"answers": answers})
        return results

    def predict_many(self, requests):
        return self.predict_prepared([self.prepare_request(request["state"], request["questions"])
                                      for request in requests])

    def predict(self, state, questions):
        return self.predict_many([{"state": state, "questions": questions}])[0]

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


def add_inference_arguments(parser):
    parser.add_argument("--pair-batch-size", type=int, default=DEFAULT_PAIR_BATCH_SIZE)
    parser.add_argument("--max-batch-tokens", type=int, default=DEFAULT_MAX_BATCH_TOKENS,
                        help="Maximum padded tokens per GPU forward (must fit max_length)")
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-dtype", choices=WEIGHT_DTYPES, default="float32")
    parser.add_argument("--attn-implementation", choices=ATTENTION_BACKENDS, default="sdpa")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="Opt in to torch.compile; first use of each shape has warmup cost")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="default")
    parser.add_argument("--pad-to-multiple-of", type=int, default=DEFAULT_PAD_MULTIPLE,
                        help="Sequence-length bucket width (capped by the export's max_length)")


def inference_kwargs(args):
    return {name: getattr(args, name) for name in (
        "pair_batch_size", "max_batch_tokens", "group_by_length", "weight_dtype",
        "attn_implementation", "compile_model", "compile_mode", "pad_to_multiple_of")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="-", help="JSON request file, or - for stdin")
    add_inference_arguments(parser)
    args = parser.parse_args()
    request = json.load(sys.stdin) if args.input == "-" else load_config(args.input)
    if "state" not in request or "questions" not in request:
        parser.error("Request must contain state and questions")
    model = DecisionModel.from_pretrained(args.model, **inference_kwargs(args))
    sys.stdout.write(json.dumps(model.predict(request["state"], request["questions"]),
                               indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
