"""Pod-only Halo check: tiny random weights and synthetic data; never accesses the Hub."""

import json
import math
import os
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from kev.core import require_cuda

require_cuda()

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForSequenceClassification, ModernBertConfig, PreTrainedTokenizerFast, set_seed
from transformers.trainer_utils import get_last_checkpoint

from kev.training import ChoiceCollator, choice_loss, make_trainer


def main():
    require_cuda()
    set_seed(42)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(16)}, unk_token="t3")),
        pad_token="t0", cls_token="t1", sep_token="t2", unk_token="t3", model_max_length=64,
    )
    model_config = ModernBertConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64, local_attention=16,
        layer_types=["full_attention", "sliding_attention"], classifier_pooling="mean",
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        num_labels=1, classifier_dropout=0.0, attention_dropout=0.0,
    )
    model = AutoModelForSequenceClassification.from_config(model_config, attn_implementation="sdpa").cuda()
    rows = [
        {
            "input_ids": [[1, 6, 2, 4 + j, 2] for j in range(count)],
            "attention_mask": [[1] * 5 for _ in range(count)],
            "labels": ([0.75, 0.25] + [0.0] * (count - 2)) if count == 5 else [1.0] + [0.0] * (count - 1),
        }
        for count in (2, 5, 3)
    ]
    collator = ChoiceCollator(tokenizer, max_length=64, max_candidates=5)
    batch = {key: value.cuda() for key, value in collator(rows).items()}
    assert batch["input_ids"].shape == (3, 5, 5)
    loss, logits = choice_loss(model, batch)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.base_model.parameters()), "No backbone gradient"
    assert model.classifier.weight.grad is not None and model.classifier.weight.grad.abs().sum() > 0, "No head gradient"
    model.zero_grad(set_to_none=True)
    config = {
        "max_length": 64, "max_candidates": 5, "seed": 42,
        "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1, "max_steps": 12, "learning_rate": 0.005,
        "lr_scheduler_type": "constant", "warmup_ratio": 0.0, "weight_decay": 0.0,
        "bf16": False, "gradient_checkpointing": False,
        "eval_steps": 4, "save_steps": 4, "logging_steps": 4,
        "save_total_limit": 2, "disable_tqdm": True, "report_to": "none",
    }
    dataset = Dataset.from_list(rows)
    with tempfile.TemporaryDirectory(prefix="kev-gpu-check-") as temporary:
        output = Path(temporary) / "run"
        trainer = make_trainer(model, tokenizer, config, output, dataset, dataset)
        model.eval()
        with torch.no_grad():
            expected = sum(choice_loss(model, {k: v.cuda() for k, v in collator([row]).items()})[0].item() for row in rows) / len(rows)
        before = trainer.evaluate()["eval_loss"]
        assert math.isclose(before, expected, rel_tol=1e-5, abs_tol=1e-6), (before, expected)
        trainer.train()
        after = trainer.evaluate()["eval_loss"]
        assert math.isfinite(after) and after < before, (before, after)
        export = Path(temporary) / "export"
        trainer.save_model(str(export))
        tokenizer.save_pretrained(export)
        reloaded = AutoModelForSequenceClassification.from_pretrained(
            export, local_files_only=True, attn_implementation="sdpa"
        ).cuda().eval()
        model.eval()
        with torch.no_grad():
            original_scores = choice_loss(model, batch)[1]
            restored_scores = choice_loss(reloaded, batch)[1]
        torch.testing.assert_close(original_scores, restored_scores)
        checkpoint = get_last_checkpoint(str(output))
        assert checkpoint and (Path(checkpoint) / "optimizer.pt").is_file()
        resumed = make_trainer(reloaded, tokenizer, {**config, "max_steps": 16}, output, dataset, dataset)
        resumed.train(resume_from_checkpoint=checkpoint)
        assert resumed.state.global_step == 16
        optimizer_steps = [state["step"].item() for state in resumed.optimizer.state.values() if "step" in state]
        assert optimizer_steps and max(optimizer_steps) == 16, optimizer_steps
        print(json.dumps({"status": "passed", "initial_loss": before, "trained_loss": after, "resumed_step": 16}))
        resumed.cleanup_ep()
        trainer.cleanup_ep()


if __name__ == "__main__":
    main()
