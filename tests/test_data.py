"""Small handwritten source fixtures; no models, datasets, or dependencies."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from kev.core import validate_row
from kev.data import (NLI, RATINGS, SkipRow, adapt_row, content_key,
                      excluded_task, split_for)
from kev.prepare import HashSample, prepare


SPECS = json.loads((Path(__file__).resolve().parents[1] / "configs" / "data.json").read_text())["sources"]


def adapt(adapter, raw, config="default", split="train", dataset=None):
    spec = next(item for item in SPECS if item["adapter"] == adapter
                and (dataset is None or item["dataset"] == dataset))
    return validate_row(adapt_row(spec, config, split, raw))


class DataTests(unittest.TestCase):
    def test_all_ten_sources_have_exact_pins(self):
        self.assertEqual(len(SPECS), 10)
        self.assertEqual(len({spec["dataset"] for spec in SPECS}), 10)
        for spec in SPECS:
            self.assertRegex(spec["revision"], r"^[a-f0-9]{40}$")

    def test_nli_native_label_mappings_and_provenance(self):
        for dataset in ("nyu-mll/multi_nli", "facebook/anli", "tasksource/FOL-nli"):
            label = "neutral" if dataset.endswith("FOL-nli") else 1
            row = adapt("nli", {"premise": "A child runs.", "hypothesis": "A child sings.", "label": label}, dataset=dataset)
            self.assertEqual(row["candidates"], NLI)
            self.assertEqual(row["target"], [0, 1, 0])
            self.assertEqual(row["source"], dataset)
            self.assertEqual(row["original_split"], "train")
            self.assertEqual(len(row["id"]), 64)

    def test_zero_shot_does_not_narrow_label_space(self):
        row = adapt("zero_shot", {"premise": "An enjoyable book.", "hypothesis": "This example is positive.", "labels": 1, "task": "custom/sentiment"})
        self.assertEqual(row["target"], [0, 1, 0])
        self.assertEqual(row["task"], "custom/sentiment")

    def test_aggregate_overlap_aliases(self):
        for task in ("glue/mnli", "anli/a1", "super_glue/boolq", "hellaswag", "defeasible-nli/atomic",
                     "FOL-nli", "doc-nli", "bigbench/arithmetic", "yelp_review_full/yelp_review_full",
                     "tweet_eval/sentiment", "boolq-natural-perturbations", "robust_nli/IS_CS"):
            self.assertTrue(excluded_task(task), task)
        self.assertFalse(excluded_task("tweet_eval/emotion"))
        with self.assertRaisesRegex(SkipRow, "overlapping_task_family"):
            adapt("zero_shot", {"premise": "P", "hypothesis": "H", "labels": 0, "task": "glue/mnli"})

    def test_doc_nli_keeps_non_entailment(self):
        row = adapt("doc_nli", {"premise": "P", "hypothesis": "H", "label": "not_entailment"})
        self.assertEqual(row["candidates"], ["entailment", "not_entailment"])
        self.assertEqual(row["target"], [0, 1])
        nli = adapt("nli", {"premise": " p  ", "hypothesis": "h", "label": 1})
        self.assertEqual(content_key(row), content_key(nli))

    def test_defeasible_keeps_update_semantics_and_groups(self):
        raw = {"Premise": "The floor is wet.", "Hypothesis": "It rained.", "Update": "A pipe burst.", "UpdateType": "weakener"}
        row = adapt("defeasible", raw, config="atomic")
        self.assertEqual(row["candidates"], ["strengthener", "weakener"])
        self.assertEqual(row["target"], [0, 1])
        other = adapt("defeasible", {**raw, "Update": "Clouds covered the sky."}, config="atomic")
        self.assertEqual(row["group_id"], other["group_id"])
        self.assertNotEqual(content_key(row), content_key(other))
        social = adapt("defeasible", {key: value for key, value in raw.items() if key != "Premise"}, config="social")
        self.assertNotIn("Premise:", social["state"])

    def test_boolq_uses_passage_and_native_boolean(self):
        row = adapt("boolq", {"passage": "Birds have feathers.", "question": "Do birds have feathers?", "answer": True})
        self.assertEqual(row["kind"], "noul")
        self.assertEqual(row["candidates"], ["No", "Yes"])
        self.assertEqual(row["target"], [0, 1])
        self.assertIn("Birds have feathers.", row["state"])
        with self.assertRaises(SkipRow):
            adapt("boolq", {"passage": "P", "question": "Q", "answer": "false"})

    def test_hellaswag_choices_are_continuations_not_contradictions(self):
        raw = {"ctx": "She opens the book and", "endings": ["reads it.", "swims.", "flies.", "sleeps."], "label": "0", "source_id": "clip-1"}
        row = adapt("hellaswag", raw)
        self.assertEqual(row["target"], [1, 0, 0, 0])
        self.assertEqual(row["candidates"], raw["endings"])
        with self.assertRaisesRegex(SkipRow, "unlabeled"):
            adapt("hellaswag", {**raw, "label": ""}, split="test")

    def test_curated_instruct_restores_full_five_star_scale(self):
        row = adapt("instruct", {"task": "yelp_review_full/yelp_review_full",
                     "inputs": 'With no explanation, label the following with either "3 stars", "2 star", "5 stars" or "1 star".\nThe meal was average.',
                     "targets": "3 stars."})
        self.assertEqual(row["candidates"], RATINGS)
        self.assertEqual(row["target"], [0, 0, 1, 0, 0])
        self.assertEqual(row["kind"], "score")
        self.assertEqual(row["state"], "The meal was average.")
        sentiment = adapt("instruct", {"task": "tweet_eval/sentiment",
                          "inputs": 'With no explanation, label the following with either "negative", "neutral" or "positive".\nAn ordinary day.',
                          "targets": "neutral."})
        self.assertEqual(sentiment["target"], [0, 1, 0])

    def test_instruct_rejects_unapproved_or_ambiguous_rows(self):
        with self.assertRaisesRegex(SkipRow, "outside_instruct_allowlist"):
            adapt("instruct", {"task": "bigbench/hindu_knowledge", "inputs": "Who?", "targets": "Fish."})
        with self.assertRaises(SkipRow):
            adapt("instruct", {"task": "yelp_review_full/yelp_review_full",
                  "inputs": 'With no explanation, label the following with either "1 star" or "2 star".\nReview', "targets": "5 stars."})

    def test_bigbench_multiple_correct_targets(self):
        raw = {"inputs": "Which numbers are even?", "multiple_choice_targets": ["2", "3", "4"], "multiple_choice_scores": [1, 0, 1]}
        row = adapt("bigbench", raw, config="handwritten_math")
        self.assertEqual(row["target"], [0.5, 0, 0.5])
        for scores in ([0, 0, 0], [1, 1, 1], [-1, 0, 1], [float("nan"), 0, 1], [1, 0]):
            with self.assertRaises(SkipRow):
                adapt("bigbench", {**raw, "multiple_choice_scores": scores})
        with self.assertRaisesRegex(SkipRow, "open_ended"):
            adapt("bigbench", {**raw, "multiple_choice_targets": [], "multiple_choice_scores": []})

    def test_blank_and_schema_failures_are_distinct(self):
        with self.assertRaises(SkipRow):
            adapt("nli", {"premise": "  ", "hypothesis": "H", "label": 0})
        with self.assertRaises(KeyError):
            adapt("nli", {"premise": "P", "label": 0})

    def test_group_partitions_are_stable_and_official_tests_stay_test(self):
        for group in ("group-a", "group-b", "group-c"):
            self.assertEqual(split_for("train", group, True), "train")
            self.assertEqual(split_for("test", group, True), "test")
            self.assertIn(split_for("validation", group, True), {"validation", "calibration"})
            self.assertEqual(split_for("validation", group, False), split_for("validation", group, False))
        allocated = {split_for("validation", str(i), False) for i in range(100)}
        self.assertEqual(allocated, {"validation", "calibration", "test"})

    def test_hash_sampling_is_bounded_and_order_independent(self):
        rows = [{"id": str(index), "value": index} for index in range(100)]
        first, second = HashSample(7, 42), HashSample(7, 42)
        for row in rows:
            first.add(row)
        for row in reversed(rows):
            second.add(row)
        first.add(first.rows()[0])
        self.assertEqual(first.rows(), second.rows())
        self.assertEqual(len(first.rows()), 7)

    def test_preparation_pipeline_with_handwritten_sources_and_fake_tokenizer(self):
        """Exercise partitions, coverage, dedup, and files without any HF code."""
        calls = []

        def load_dataset(source, subset, split, revision, streaming):
            self.assertTrue(calls and calls[0] == "cuda_guard")
            self.assertTrue(streaming)
            spec = next(spec for spec in SPECS if spec["dataset"] == source)
            self.assertEqual(revision, spec["revision"])
            adapter = spec["adapter"]
            for index in range(32):
                identifier = f"{source}:{subset}:{split}:{index}"
                if adapter in {"nli", "doc_nli", "zero_shot"}:
                    raw = {"premise": identifier, "hypothesis": "The claim is supported.", "label": 0}
                    if adapter == "doc_nli":
                        raw["label"] = "not_entailment"
                    if adapter == "zero_shot":
                        raw.update(labels=0, task="fixture/classification")
                elif adapter == "defeasible":
                    raw = {"Premise": identifier, "Hypothesis": identifier, "Update": "New evidence.", "UpdateType": "strengthener"}
                elif adapter == "hellaswag":
                    raw = {"ctx": identifier, "source_id": identifier, "endings": ["one", "two", "three", "four"], "label": "1"}
                elif adapter == "boolq":
                    # One deliberate train/holdout overlap must be removed.
                    passage = f"shared-boolq-{index}" if index == 0 else identifier
                    raw = {"passage": passage, "question": "Is this supported?", "answer": True}
                elif adapter == "bigbench":
                    raw = {"inputs": identifier, "multiple_choice_targets": ["yes", "no", "perhaps"], "multiple_choice_scores": [1, 0, 1]}
                else:
                    task = "yelp_review_full/yelp_review_full" if index % 2 else "tweet_eval/sentiment"
                    options = '"1 star" or "2 star"' if index % 2 else '"negative", "neutral" or "positive"'
                    raw = {"task": task,
                           "inputs": f"With no explanation, label the following with either {options}.\n{identifier}",
                           "targets": "1 star." if index % 2 else "neutral."}
                yield raw

        class Tokenizer:
            @staticmethod
            def from_pretrained(name, revision, trust_remote_code):
                self.assertEqual((name, revision), ("fixture-model", "fixture-revision"))
                self.assertFalse(trust_remote_code)
                return lambda prompts, candidates, **kwargs: {
                    "input_ids": [[1, 2] for _ in candidates],
                    "attention_mask": [[1, 1] for _ in candidates],
                }

        fake_datasets = types.SimpleNamespace(load_dataset=load_dataset,
                                             get_dataset_config_names=lambda *args, **kwargs: ["analytic_entailment"])
        config = {"max_train_per_source": 12, "max_train_per_task": 6,
                  "max_eval_per_source": 8, "max_length": 32, "max_candidates": 16,
                  "seed": 42, "model_name_or_path": "fixture-model", "model_revision": "fixture-revision",
                  "scan_limit_per_split": 32}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "datasets": fake_datasets, "transformers": types.SimpleNamespace(AutoTokenizer=Tokenizer),
        }), patch("kev.prepare.require_cuda", side_effect=lambda: calls.append("cuda_guard")), patch("builtins.print"):
            prepare(config, directory)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            train = [json.loads(line) for line in (Path(directory) / "train.jsonl").read_text().splitlines()]
            self.assertEqual({row["source"] for row in train}, {spec["dataset"] for spec in SPECS})
            self.assertEqual(manifest["source_counts"]["google/boolq"]["dropped:holdout_overlap"], 1)
            seen_groups, seen_content = {}, set()
            for split in ("test", "calibration", "validation", "train"):
                for line in (Path(directory) / f"{split}.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    self.assertEqual(row["target"], row["labels"])
                    self.assertEqual(seen_groups.setdefault(row["group_id"], split), split)
                    self.assertNotIn(content_key(row), seen_content)
                    seen_content.add(content_key(row))
            self.assertFalse((Path(directory) / ".holdout-fingerprints.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
