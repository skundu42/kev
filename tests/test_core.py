import json
import math
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

from kev.calibrate import fit_temperature, log_loss
from kev.core import InputTooLongError, answer, pad_targets, parse_questions, require_cuda, softmax, tokenize_row, validate_row
from kev.evaluate import Metrics, evaluate


class FakeTokenizer:
    def __call__(self, prompts, candidates, **kwargs):
        assert kwargs["truncation"] is False
        ids = [[1] + [3] * len(p.split()) + [2] + [4] * len(c.split()) + [2]
               for p, c in zip(prompts, candidates)]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def row(**changes):
    result = {"state": "A cyclist is moving.", "instructions": "Is someone moving?",
              "candidates": ["No", "Yes"], "target": [0.0, 1.0], "kind": "noul",
              "source": "synthetic", "task": "movement"}
    result.update(changes)
    return result


class CoreTests(unittest.TestCase):
    def test_padded_targets_exclude_only_absent_candidates(self):
        labels, valid = pad_targets([[0.0, 1.0], [0.5, 0, 0, 0, 0.5], [0, 1, 0]])
        self.assertEqual(labels[0], [0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(valid[0], [True, True, False, False, False])
        self.assertEqual(valid[1], [True] * 5)
        self.assertEqual(valid[2], [True, True, True, False, False])
        for targets in ([], [[1.0]], [[-1, 2]], [[float("nan"), 1]]):
            with self.assertRaises(ValueError):
                pad_targets(targets)

    def test_invalid_targets_and_candidates(self):
        for changes in ({"target": [0.0, float("nan")]}, {"target": [-1.0, 2.0]},
                        {"target": [1.0]}, {"target": [0.2, 0.2]},
                        {"candidates": ["same", "same"]}, {"state": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_row(row(**changes))

    def test_tokenization_preserves_candidates_and_soft_targets(self):
        example = row(candidates=["One", "Two", "Three"], target=[0.5, 0.0, 0.5], kind="choice")
        result = tokenize_row(example, FakeTokenizer(), 100)
        self.assertEqual(result["labels"], example["target"])
        self.assertEqual(len(result["input_ids"]), 3)
        with self.assertRaisesRegex(InputTooLongError, "overlength"):
            tokenize_row(example, FakeTokenizer(), 4)

    def test_all_question_types(self):
        request = json.loads((Path(__file__).resolve().parents[1] / "examples/request.json").read_text())
        rows, metadata = parse_questions(request["state"], request["questions"])
        answers = [answer(r["kind"], r["candidates"], keys, softmax([0.0] * len(keys)))
                   for r, (_, keys) in zip(rows, metadata)]
        self.assertEqual(answers[0]["confidence"], 0)
        self.assertAlmostEqual(answers[1]["noul"], 0.5)
        self.assertAlmostEqual(answers[2]["score"], 1)
        self.assertEqual(answers[2]["legend"]["2"], "Very frustrated")

    def test_question_validation_and_softmax_stability(self):
        for question in ({"type": "other", "instructions": "x"},
                         {"type": "noul", "instructions": "x", "criteria": {"yes": "x"}},
                         {"type": "choice", "instructions": "x", "criteria": {"a": "x"}}):
            with self.assertRaises(ValueError):
                parse_questions("state", {"test": question})
        self.assertEqual(softmax([10000.0, 10000.0]), [0.5, 0.5])
        for invalid in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                softmax([0, 1], invalid)

    def test_metrics_and_calibration(self):
        metric = Metrics()
        metric.update(row(kind="score"), [0.25, 0.75])
        result = metric.result()
        self.assertAlmostEqual(result["brier_score"], 0.125)
        self.assertAlmostEqual(result["log_loss"], -math.log(0.75))
        self.assertAlmostEqual(result["ordinal_mae"], 0.25)
        self.assertAlmostEqual(result["ece_15"], 0.25)
        examples = [([0.0, 8.0], [0.0, 1.0]), ([0.0, 8.0], [1.0, 0.0])]
        temperature = fit_temperature(examples)
        self.assertGreater(temperature, 1)
        self.assertLessEqual(log_loss(examples, temperature), log_loss(examples, 1))
        for invalid in ([], [([float("nan"), 1], [0, 1])], [([0, 1], [-1, 2])]):
            with self.assertRaises(ValueError):
                fit_temperature(invalid)
        report = evaluate([(row(), [0.0, 1.0])], 2)
        self.assertEqual(report["calibrated"]["source/synthetic"]["count"], 1)

    def test_local_guard_runs_before_torch_import(self):
        with patch.object(sys, "platform", "darwin"), self.assertRaisesRegex(RuntimeError, "not locally"):
            require_cuda()
        self.assertNotIn("torch", sys.modules)

    def test_offline_runner_blocks_network(self):
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            socket.create_connection(("example.com", 443))


if __name__ == "__main__":
    unittest.main()
