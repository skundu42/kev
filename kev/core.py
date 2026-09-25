"""Shared formatting and validation. Standard library only; safe to test offline."""

import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


class InputTooLongError(ValueError):
    """A decision exceeds the configured token limit without truncation."""


def load_config(path):
    # JSON is also YAML; keeping our configs in this subset needs no local PyYAML.
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object (also valid YAML).")
    return config


def require_cuda():
    """Guard GPU model commands before importing Hugging Face libraries."""
    if sys.platform != "linux":
        raise RuntimeError("Run this command inside the Runpod Halo GPU container, not locally.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command on the GPU pod.")
    root = Path(os.environ.get("KEV_WORKDIR", "/workspace/kev-run"))
    os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    return torch.device("cuda")


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def validate_row(row, max_candidates=16):
    for name in ("state", "instructions"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= max_candidates:
        raise ValueError(f"Expected 2–{max_candidates} candidates")
    if any(not isinstance(c, str) or not c.strip() for c in candidates):
        raise ValueError("Candidates must be nonempty strings")
    if len({c.strip() for c in candidates}) != len(candidates):
        raise ValueError("Candidates must be distinct")
    if row.get("kind", "choice") not in {"choice", "noul", "score"}:
        raise ValueError("Unknown decision kind")
    if row.get("kind") == "noul" and len(candidates) != 2:
        raise ValueError("Noul requires two candidates in false/true order")
    if "target" in row:
        target = row["target"]
        if not isinstance(target, list) or len(target) != len(candidates):
            raise ValueError("Target and candidate counts differ")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in target):
            raise ValueError("Targets must be finite nonnegative numbers")
        if not math.isclose(sum(target), 1.0, abs_tol=1e-6):
            raise ValueError("Target probabilities must sum to one")
    return row


def format_prompt(state, instructions, candidates, kind="choice"):
    # Include the complete criteria so references to other answers remain meaningful.
    return (f"Decision type: {kind}\nInstructions: {instructions}\n"
            f"State:\n{state}\nCriteria:\n"
            + "\n".join(f"{i}: {candidate}" for i, candidate in enumerate(candidates)))


def tokenize_row(row, tokenizer, max_length=1024, max_candidates=16):
    validate_row(row, max_candidates)
    if not isinstance(max_length, int) or not 1 <= max_length <= 7999:
        raise ValueError("max_length must be between 1 and Ettin's 7999-token limit")
    prompt = format_prompt(row["state"], row["instructions"], row["candidates"],
                           row.get("kind", "choice"))
    encoded = tokenizer([prompt] * len(row["candidates"]), row["candidates"],
                        truncation=False, padding=False, return_token_type_ids=False)
    if any(len(ids) > max_length for ids in encoded["input_ids"]):
        raise InputTooLongError(f"overlength: decision exceeds {max_length} tokens; no input was truncated")
    result = {key: encoded[key] for key in ("input_ids", "attention_mask")}
    if "target" in row:
        result["labels"] = list(row["target"])
    return result


def softmax(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if len(logits) < 2 or any(not math.isfinite(x) for x in logits):
        raise ValueError("Expected at least two finite logits")
    peak = max(logits)
    values = [math.exp((x - peak) / temperature) for x in logits]
    total = sum(values)
    return [x / total for x in values]


def pad_targets(targets):
    """Validate soft targets and mark padded candidates without a tensor dependency."""
    if not targets:
        raise ValueError("Cannot pad an empty decision batch")
    for target in targets:
        if not isinstance(target, list) or len(target) < 2:
            raise ValueError("Each decision requires at least two target probabilities")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in target):
            raise ValueError("Targets must be finite nonnegative numbers")
        if not math.isclose(sum(target), 1.0, abs_tol=1e-6):
            raise ValueError("Target probabilities must sum to one")
    width = max(map(len, targets))
    labels = [[float(v) for v in row] + [0.0] * (width - len(row)) for row in targets]
    mask = [[True] * len(row) + [False] * (width - len(row)) for row in targets]
    return labels, mask


def parse_questions(state, questions, max_candidates=16):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    rows, metadata = [], []
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id or not isinstance(question, dict):
            raise ValueError("Each question needs a nonempty string identifier and object value")
        kind = question.get("type")
        instructions = question.get("instructions")
        if instructions is None or not text(instructions).strip():
            raise ValueError(f"Question {question_id}: instructions are required")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or any(not isinstance(k, str) or not k.strip() for k in criteria):
                raise ValueError("Choice criteria must map nonempty option names to descriptions")
            keys = list(criteria)
            candidates = [k if v is None else f"{k}: {text(v)}" for k, v in criteria.items()]
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError("Score criteria must be an ordered array")
            candidates = [text(v) for v in criteria]
            keys = [str(i) for i in range(len(candidates))]
        elif kind == "noul":
            criteria = {"false": "No", "true": "Yes"} if criteria is None else criteria
            if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
                raise ValueError("Noul criteria must contain exactly false and true")
            keys = ["false", "true"]
            candidates = [text(criteria[k]) for k in keys]
        else:
            raise ValueError(f"Unknown question type: {kind}")
        row = {"state": text(state), "instructions": text(instructions),
               "candidates": candidates, "kind": kind}
        validate_row(row, max_candidates)
        rows.append(row)
        metadata.append((question_id, keys))
    return rows, metadata


def answer(kind, candidates, keys, probabilities):
    if len(probabilities) != len(candidates) or len(keys) != len(candidates):
        raise ValueError("Probability and candidate counts differ")
    validate_row({"state": "answer", "instructions": "answer", "kind": kind,
                  "candidates": candidates, "target": probabilities}, len(candidates))
    if kind == "noul":
        return {"type": kind, "noul": probabilities[1]}
    count = len(probabilities)
    output = {"type": kind, "probabilities": dict(zip(keys, probabilities)),
              "confidence": max(0.0, min(1.0, (count * max(probabilities) - 1) / (count - 1)))}
    if kind == "choice":
        output["choice"] = keys[max(range(count), key=probabilities.__getitem__)]
    else:
        output["score"] = sum(i * p for i, p in enumerate(probabilities))
        output["legend"] = dict(zip(keys, candidates))
    return output


def read_rows(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def staged_export(destination):
    """Publish final/ only after weights, tokenizer, and metadata are complete."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Export already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        yield staging
        staging.rename(destination)


def validate_run_directory(output_dir, config, provenance, resume=None):
    """Resume interrupted runs only, retaining completed exports and their reports."""
    output = Path(output_dir).resolve()
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory is not empty; choose another or resume its checkpoint")
        return None
    checkpoint = Path(resume).resolve()
    if checkpoint.parent != output:
        raise ValueError("Resume checkpoint must belong to the selected output directory")
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError("Resume requires a checkpoint containing trainer_state.json")
    if (output / "final").exists() or (output / "evaluation.json").exists():
        raise FileExistsError("This run already has exported results. Preserve them and start a new run; "
                              "resume is reserved for interrupted runs without a final export.")
    previous_config = load_config(output / "training_config.json")
    fields = ("model_name_or_path", "model_revision", "max_length", "max_candidates", "seed")
    if any(previous_config.get(key) != config.get(key) for key in fields):
        raise ValueError("The resumed run uses a different model, input format, or seed")
    # Trainer uses these settings to skip consumed batches and restore its schedule.
    # A checkpoint resumes an interrupted run; it is not a new training recipe.
    training_fields = (
        "per_device_train_batch_size", "gradient_accumulation_steps", "dataloader_drop_last",
        "num_train_epochs", "max_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "warmup_steps", "lr_scheduler_type", "lr_scheduler_kwargs",
        "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm", "bf16", "fp16",
    )
    changed = [key for key in training_fields if previous_config.get(key) != config.get(key)]
    if changed:
        raise ValueError("Resume requires the original training settings; changed: " + ", ".join(changed))
    if load_config(output / "provenance.json") != provenance:
        raise ValueError("The resumed run uses different prepared data")
    return str(checkpoint)
