import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from kev.compare import question_for, probabilities_for, sample_rows, summarize, laya_clips


class ComparisonTests(unittest.TestCase):
    def row(self, kind="choice"):
        return dict(state="example", instructions="choose", kind=kind,
                    candidates=["No", "Yes"], target=[0., 1.], source="synthetic", task="demo")

    def test_mapping_and_rounding(self):
        row = self.row()
        q, keys = question_for(row)
        self.assertEqual(q["criteria"], {"No": None, "Yes": None})
        self.assertEqual(keys, row["candidates"])
        q, keys = question_for(self.row("noul"))
        self.assertEqual(q["criteria"], {"false": "No", "true": "Yes"})
        self.assertEqual(probabilities_for({"noul": .8}, "noul", keys), [1-.8, .8])
        q, keys = question_for(self.row("score"))
        self.assertEqual(keys, ["0", "1"])
        p = probabilities_for({"probabilities": {"0": .3, "1": .6999}}, "score", keys)
        self.assertAlmostEqual(sum(p), 1.)
        for invalid in (float("nan"), -1, 2):
            with self.assertRaises(ValueError):
                probabilities_for({"noul": invalid}, "noul", [])

    def test_sampling_is_repeatable_and_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            rows = [dict(self.row(), state=str(i), source=str(i % 2)) for i in range(30)]
            path.write_text("\n".join(map(json.dumps, rows)))
            a = sample_rows(path, 3, 42)
            self.assertEqual(a, sample_rows(path, 3, 42))
            self.assertEqual(len(a), 6)
            self.assertEqual(sum(row["source"] == "0" for row in a), 3)
            self.assertEqual(len(sample_rows(path, 0, 42)), 30)

    def test_shared_subset_and_multitarget_metrics(self):
        rows = [self.row(), dict(self.row(), target=[.5, .5])]
        result = summarize(rows, [[.1, .9], [.9, .1]], [False, True])
        self.assertEqual(result["overall"]["accuracy"], 1.)
        self.assertEqual(result["shared_untruncated"]["count"], 1)

    def test_clipping_of_state_instruction_and_options(self):
        class Tokenizer:
            mask_token = "[MASK]"
            def __call__(self, text, **kwargs):
                return {"input_ids": text.split()}
        agent = SimpleNamespace(tok=Tokenizer(), cfg={"max_len": 40, "head_max_len": 25})
        render = lambda q: list(q["crit"])
        row = self.row()
        self.assertFalse(laya_clips(row, agent, render))
        for field in ("state", "instructions"):
            self.assertTrue(laya_clips(dict(row, **{field: "word " * 60}), agent, render))
        self.assertTrue(laya_clips(dict(row, candidates=["word " * 60, "Yes"]), agent, render))
        self.assertTrue(laya_clips(dict(row, state="a [MASK] b"), agent, render))
