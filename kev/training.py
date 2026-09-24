"""Candidate scoring trained by Halo; import this module only inside the GPU pod."""

from __future__ import annotations

import math
import os
from dataclasses import fields
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from src.configs.classification_config import ClassificationConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.reward.classification import ClassificationTrainer

from kev.core import load_config, pad_targets, require_cuda, validate_run_directory, write_json
from kev.hub import validate_dataset


class ChoiceCollator:
    """Keep the decision batch dimension visible to Trainer, including during evaluation."""

    required_dataset_columns = ("input_ids", "attention_mask", "labels")

    def __init__(self, tokenizer, max_length=1024, max_candidates=16):
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define a pad token.")
        self.tokenizer = tokenizer
        # len(tokenizer) can rebuild the full vocabulary; never call it per token.
        self.vocab_size = len(tokenizer)
        self.max_length = max_length
        self.max_candidates = max_candidates

    def __call__(self, rows):
        if not rows:
            raise ValueError("Cannot collate an empty decision batch.")
        for row in rows:
            count = len(row["input_ids"])
            if not 2 <= count <= self.max_candidates:
                raise ValueError(f"Expected 2..{self.max_candidates} candidates, got {count}.")
            if len(row["labels"]) != count or len(row["attention_mask"]) != count:
                raise ValueError("Candidate inputs, masks, and labels must have matching counts.")
        padded_targets, candidate_masks = pad_targets([row["labels"] for row in rows])
        labels = torch.tensor(padded_targets, dtype=torch.float32)
        valid = torch.tensor(candidate_masks, dtype=torch.bool)
        flat, positions = [], []
        for i, row in enumerate(rows):
            for j, (ids, mask) in enumerate(zip(row["input_ids"], row["attention_mask"], strict=True)):
                if not 1 <= len(ids) <= self.max_length or len(ids) != len(mask):
                    raise ValueError("Each candidate needs equal, nonempty token/mask lengths within max_length.")
                if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < self.vocab_size for x in ids):
                    raise ValueError("Candidate token IDs must be integers within the tokenizer vocabulary.")
                if any(x not in (0, 1) for x in mask) or not any(mask):
                    raise ValueError("Each attention mask must contain only zero/one and at least one real token.")
                flat.append({"input_ids": ids, "attention_mask": mask})
                positions.append((i, j))
        padded = self.tokenizer.pad(flat, padding=True, return_tensors="pt")
        shape = (*labels.shape, padded["input_ids"].shape[-1])
        ids = torch.full(shape, self.tokenizer.pad_token_id, dtype=torch.long)
        masks = torch.zeros(shape, dtype=torch.long)
        for k, (i, j) in enumerate(positions):
            ids[i, j] = padded["input_ids"][k]
            masks[i, j] = padded["attention_mask"][k]
        return {"input_ids": ids, "attention_mask": masks, "labels": labels, "candidate_mask": valid}


def choice_loss(model, inputs):
    """One equally weighted soft-target cross-entropy per decision."""
    valid = inputs["candidate_mask"]
    scores = model(
        input_ids=inputs["input_ids"][valid],
        attention_mask=inputs["attention_mask"][valid],
        return_dict=True,
    ).logits.squeeze(-1).float()
    # Finite FP32 masking keeps zero-mass padded targets from producing 0 * -inf.
    logits = scores.new_full(valid.shape, torch.finfo(torch.float32).min)
    logits[valid] = scores
    loss = -(inputs["labels"].float() * logits.log_softmax(-1)).sum(-1).mean()
    return loss, logits


class ChoiceTrainer(ClassificationTrainer):
    """Reuse Halo's optimizer, evaluation, checkpointing, and training loop."""

    _supports_ep = False
    _supports_cp = False
    _supports_tp = False
    _supports_pp = False
    _pp_unsupported_reason = "This demo trains grouped decisions on one GPU."
    _loss_is_own_mean = True
    _loss_outside_model_forward = True

    def _compute_loss_inner(self, model, inputs, return_outputs):
        loss, logits = choice_loss(model, inputs)
        return (loss, {"logits": logits}) if return_outputs else loss

    def _extract_document_lengths(self, inputs):
        return inputs["attention_mask"].sum(-1).flatten()


def build_training_args(config, output_dir):
    accepted = {field.name for field in fields(ClassificationConfig)}
    values = {key: value for key, value in config.items() if key in accepted}
    # Transformers 5.16 folds fractional warmup into warmup_steps.
    if "warmup_ratio" in config:
        ratio = config["warmup_ratio"]
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio < 1:
            raise ValueError("warmup_ratio must be a finite number in [0, 1).")
        if "warmup_steps" in config:
            raise ValueError("Specify either warmup_ratio or warmup_steps, not both.")
        values["warmup_steps"] = float(ratio)
    values.update(
        output_dir=str(output_dir),
        remove_unused_columns=False,
        prediction_loss_only=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        use_liger_kernel=False,
        optim="adamw_torch",
    )
    values.setdefault("eval_strategy", "steps")
    values.setdefault("save_strategy", "steps")
    values.setdefault("report_to", "none")
    args = ClassificationConfig(**values)
    args._n_gpu = 1
    return args


def make_trainer(model, tokenizer, config, output_dir, train_dataset, eval_dataset):
    if model.config.num_labels != 1:
        raise ValueError("Candidate scoring requires exactly one output logit per pair.")
    # Keep FP32 master/AdamW weights and Accelerate BF16 autocast. Halo's default
    # disables native_amp because its usual loader stores model weights in BF16.
    return ChoiceTrainer(
        model=model,
        processing_class=tokenizer,
        args=build_training_args(config, output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ChoiceCollator(tokenizer, config["max_length"], config["max_candidates"]),
        parallelism_config=ParallelismConfig(
            use_grouped_gemm=False,  # Halo defaults to MoE wrappers even on one GPU.
            bf16_optimizer=False,
            fp32_output_conversion=True,
        ),
        is_binary=False,
    )


def train(config_path, data_dir, output_dir, resume_from_checkpoint=None):
    config = load_config(config_path)
    require_cuda()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The demo supports one GPU/process; do not use torchrun or -n > 1.")
    torch.cuda.set_device(0)
    if config.get("bf16", False) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a GPU with BF16 support.")
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    provenance = validate_dataset(data_dir, config)
    if (data_dir / "hub.json").is_file():
        provenance["prepared_dataset_hub"] = load_config(data_dir / "hub.json")
    resume_from_checkpoint = validate_run_directory(output_dir, config, provenance, resume_from_checkpoint)
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], revision=config["model_revision"], trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"],
        revision=config["model_revision"],
        num_labels=1,
        dtype=torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if config["max_length"] > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the base model's context window.")
    datasets = load_dataset(
        "json", data_files={split: str(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    )
    if any(len(datasets[split]) == 0 for split in ("train", "validation")):
        raise ValueError("Training and validation splits must both contain decisions.")
    trainer = make_trainer(model, tokenizer, config, output_dir, datasets["train"], datasets["validation"])
    write_json(output_dir / "training_config.json", config)
    write_json(output_dir / "provenance.json", provenance)
    torch.cuda.reset_peak_memory_stats(0)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    write_json(output_dir / "gpu_memory.json", {
        "gpu": torch.cuda.get_device_name(0),
        "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
        "note": "PyTorch allocator peaks during training and validation; excludes other processes and some CUDA allocations.",
    })
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", metrics)
    metadata = {key: config[key] for key in ("max_length", "max_candidates", "model_name_or_path", "model_revision")}
    metadata["temperature"] = 1.0
    write_json(final_dir / "kev_config.json", metadata)
    write_json(final_dir / "training_config.json", config)
    write_json(final_dir / "provenance.json", provenance)
    trainer.cleanup_ep()
    print(f"Saved model and tokenizer to {final_dir}")
