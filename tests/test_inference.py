"""Offline inference regressions; these stand-ins do not benchmark model performance."""

import json
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kev.core import InputTooLongError
from kev.inference import DecisionModel, plan_batches


class TensorStub:
    def __init__(self, values, transfers):
        self.values, self.transfers = values, transfers

    def to(self, device, *, non_blocking=False):
        return self

    def squeeze(self, dimension):
        return self

    def float(self):
        return self

    def __getitem__(self, index):
        return TensorStub(self.values[index], self.transfers)

    def clone(self):
        return TensorStub(list(self.values), self.transfers)

    def cpu(self):
        self.transfers.append(len(self.values))
        return self

    def tolist(self):
        return self.values


class TokenizerStub:
    def __init__(self, candidates, transfers):
        self.candidates, self.transfers, self.batches = candidates, transfers, []

    def __call__(self, prompts, candidates, **kwargs):
        assert not kwargs["truncation"] and not kwargs["padding"]
        ids = [[self.candidates[c][0]] * self.candidates[c][1] for c in candidates]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    def pad(self, pairs, *, padding, max_length, return_tensors):
        assert padding == "max_length" and return_tensors == "pt"
        self.batches.append((len(pairs), max_length))
        assert all(set(pair) == {"input_ids", "attention_mask"} for pair in pairs)
        assert all(0 < len(pair["input_ids"]) <= max_length for pair in pairs)
        return {key: TensorStub([pair[key] + [0] * (max_length - len(pair[key]))
                                 for pair in pairs], self.transfers)
                for key in ("input_ids", "attention_mask")}


def model_stub(transfers):
    def forward(*, input_ids, attention_mask):
        assert all(any(mask) for mask in attention_mask.values)
        return SimpleNamespace(logits=TensorStub([row[0] for row in input_ids.values], transfers))
    model = Mock(side_effect=forward, config=SimpleNamespace(num_labels=1))
    model.eval.return_value = model.to.return_value = model
    return model


def torch_stub(transfers):
    return SimpleNamespace(
        inference_mode=nullcontext, autocast=lambda *args, **kwargs: nullcontext(),
        cuda=SimpleNamespace(is_bf16_supported=Mock(return_value=True)),
        bfloat16="bf16", float32="fp32",
        cat=lambda tensors: TensorStub([v for tensor in tensors for v in tensor.values], transfers),
    )


class InferenceTests(unittest.TestCase):
    def test_budget_counts_rounded_width_and_compilation_padding(self):
        cases = (
            ([17, 17, 17], 32, 64, 64, 16, False, [([0, 1], 32, 2), ([2], 32, 1)]),
            ([17, 17, 17, 17], 32, 96, 64, 16, True, [([0, 1], 32, 2), ([2, 3], 32, 2)]),
            ([17, 17, 17], 32, 128, 64, 16, True, [([0, 1, 2], 32, 4)]),
            ([65, 65], 32, 130, 65, 64, False, [([0, 1], 65, 2)]),
            ([2] * 7, 3, None, 64, 8, True,
             [([0, 1, 2], 8, 3), ([3, 4, 5], 8, 3), ([6], 8, 1)]),
        )
        for lengths, pairs, tokens, maximum, multiple, fixed, expected in cases:
            with self.subTest(tokens=tokens, fixed=fixed, pairs=pairs):
                actual = list(plan_batches(lengths, pairs, tokens, maximum, multiple,
                                           fixed_batch_shapes=fixed))
                self.assertEqual(actual, expected)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            list(plan_batches([17], 16, 31, 64, 16))

    def test_grouping_preserves_requests_questions_and_candidates_with_one_transfer(self):
        transfers = []
        tokenizer = TokenizerStub({"east": (2, 29), "west": (4, 8), "No": (5, 17),
                                   "Yes": (3, 31), "low": (1, 9), "mid": (2, 25),
                                   "high": (0, 6)}, transfers)
        model = model_stub(transfers)
        runtime = DecisionModel(model, tokenizer, {"max_length": 32, "max_candidates": 4},
                                "cuda", 4, max_batch_tokens=128, pad_to_multiple_of=8,
                                compile_model=True)
        requests = [
            {"state": "first", "questions": {
                "direction": {"type": "choice", "instructions": "Pick", "criteria": {"east": None, "west": None}},
                "truth": {"type": "noul", "instructions": "Check"}}},
            {"state": "second", "questions": {
                "direction": {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]}}},
        ]
        with patch.dict(sys.modules, {"torch": torch_stub(transfers)}):
            result = runtime.predict_many(requests)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result[0]["answers"]), ["direction", "truth"])
        choice = result[0]["answers"]["direction"]
        self.assertEqual(list(choice["probabilities"]), ["east", "west"])
        self.assertEqual(choice["choice"], "west")
        self.assertAlmostEqual(choice["probabilities"]["east"], 1 / (1 + math.exp(2)))
        self.assertAlmostEqual(result[0]["answers"]["truth"]["noul"], 1 / (1 + math.exp(2)))
        score = result[1]["answers"]["direction"]
        self.assertEqual(score["legend"], {"0": "low", "1": "mid", "2": "high"})
        denominator = math.exp(1) + math.exp(2) + 1
        self.assertAlmostEqual(score["score"], (math.exp(2) + 2) / denominator)
        self.assertEqual(tokenizer.batches, [(4, 24), (4, 32)])
        self.assertEqual(model.call_count, 2)
        self.assertEqual(transfers, [7])

    def test_no_grouping_retains_input_batches(self):
        lengths = [29, 8, 17, 31, 9, 25, 6]
        batches = list(plan_batches(lengths, 3, None, 32, 8, group_by_length=False))
        self.assertEqual(batches, [([0, 1, 2], 32, 3), ([3, 4, 5], 32, 3), ([6], 8, 1)])

    def test_exact_token_limit_is_accepted_and_overflow_rejected_before_forward(self):
        tokenizer = TokenizerStub({"No": (1, 65), "Yes": (2, 1)}, [])
        model = model_stub([])
        runtime = DecisionModel(model, tokenizer, {"max_length": 65, "max_candidates": 2},
                                "cuda", max_batch_tokens=130)
        questions = {"answer": {"type": "noul", "instructions": "Check"}}
        prepared = runtime.prepare_request("state", questions)
        self.assertEqual([len(pair["input_ids"]) for pair in prepared.pairs], [65, 1])
        tokenizer.candidates["No"] = (1, 66)
        with self.assertRaisesRegex(InputTooLongError, "no input was truncated"):
            runtime.prepare_request("state", questions)
        model.assert_not_called()

    def test_invalid_options_fail_before_cuda_or_loading(self):
        options = [{key: value} for key in ("pair_batch_size", "max_batch_tokens", "pad_to_multiple_of")
                   for value in (False, 0, -1, 1.5)]
        options.extend([{"weight_dtype": "float16"}, {"attn_implementation": "invalid"},
                        {"compile_mode": "invalid"}])
        with patch("kev.inference.require_cuda") as cuda:
            for kwargs in options:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    DecisionModel.from_pretrained("unused", **kwargs)
            cuda.assert_not_called()

    def test_weight_dtype_attention_and_compile_are_passed_to_local_loader(self):
        model = model_stub([])
        torch = torch_stub([])
        torch.compile = Mock(return_value=model)
        transformers = SimpleNamespace(
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
        )
        loader = transformers.AutoModelForSequenceClassification.from_pretrained
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                patch("kev.inference.require_cuda", return_value="cuda"):
            path = Path(directory)
            (path / "kev_config.json").write_text(json.dumps({"max_length": 64, "max_candidates": 4}))
            for dtype, supported, expected in (("bfloat16", True, "bfloat16"), ("auto", True, "bfloat16"),
                                                ("auto", False, "float32"), ("float32", True, "float32")):
                with self.subTest(dtype=dtype, supported=supported):
                    torch.cuda.is_bf16_supported.return_value = supported
                    runtime = DecisionModel.from_pretrained(path, weight_dtype=dtype)
                    self.assertEqual(runtime.weight_dtype, expected)
                    loader.assert_called_with(path, local_files_only=True, trust_remote_code=False,
                                              attn_implementation="sdpa", dtype=getattr(torch, expected))
            torch.cuda.is_bf16_supported.return_value = True
            DecisionModel.from_pretrained(path, weight_dtype="bfloat16", attn_implementation="flash_attention_2",
                                          compile_model=True, compile_mode="reduce-overhead")
            loader.assert_called_with(path, local_files_only=True, trust_remote_code=False,
                                      attn_implementation="flash_attention_2", dtype="bf16")
            torch.compile.assert_called_once_with(model, mode="reduce-overhead", dynamic=False)
            transformers.AutoTokenizer.from_pretrained.assert_called_with(
                path, local_files_only=True, trust_remote_code=False)
            loader.reset_mock()
            with self.assertRaisesRegex(ValueError, "requires.*bfloat16"):
                DecisionModel.from_pretrained(path, attn_implementation="flash_attention_2", weight_dtype="float32")
            torch.cuda.is_bf16_supported.return_value = False
            with self.assertRaisesRegex(ValueError, "BF16 support"):
                DecisionModel.from_pretrained(path, weight_dtype="bfloat16")
            with self.assertRaisesRegex(ValueError, "max_length"):
                DecisionModel.from_pretrained(path, max_batch_tokens=63)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
