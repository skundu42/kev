import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from kev.core import pad_targets, staged_export, validate_run_directory, write_json


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_interrupted_export_does_not_publish_final_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "final"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with staged_export(final) as staging:
                    (staging / "weights.fixture").write_text("synthetic")
                    self.assertFalse(final.exists())
                    raise RuntimeError("interrupted")
            self.assertFalse(final.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            with staged_export(final) as staging:
                write_json(staging / "kev_config.json", {"temperature": 1.0})
                self.assertFalse(final.exists())
            self.assertTrue((final / "kev_config.json").is_file())
            with self.assertRaises(FileExistsError):
                with staged_export(final):
                    self.fail("Completed export was overwritten")

    def test_nonfinite_training_metrics_stop_the_run(self):
        tree = ast.parse((ROOT / "kev/training.py").read_text())
        callback_class = next(node for node in tree.body
                              if isinstance(node, ast.ClassDef) and node.name == "FiniteMetricsCallback")
        namespace = {"TrainerCallback": object, "math": math}
        exec(compile(ast.Module(body=[callback_class], type_ignores=[]), "metrics_callback", "exec"), namespace)
        callback = namespace["FiniteMetricsCallback"]()
        state = SimpleNamespace(global_step=10)
        callback.on_log(None, state, None, logs={"loss": 0.2, "grad_norm": 1.0, "eval_loss": 0.3})
        for metric in ("loss", "grad_norm", "eval_loss"):
            for value in (float("nan"), float("inf")):
                with self.subTest(metric=metric), self.assertRaisesRegex(FloatingPointError, "step 10"):
                    callback.on_log(None, state, None, logs={metric: value})

    def test_collator_caches_vocab_size_and_still_rejects_invalid_tokens(self):
        tree = ast.parse((ROOT / "kev/training.py").read_text())
        collator_class = next(node for node in tree.body
                              if isinstance(node, ast.ClassDef) and node.name == "ChoiceCollator")
        class ReachedPadding(Exception):
            pass
        class Tokenizer:
            pad_token_id = 0
            length_calls = 0
            def __len__(self):
                self.length_calls += 1
                return 100
            def pad(self, *args, **kwargs):
                raise ReachedPadding
        namespace = {"pad_targets": pad_targets, "torch": SimpleNamespace(
            tensor=lambda *args, **kwargs: None, float32=None, bool=None)}
        exec(compile(ast.Module(body=[collator_class], type_ignores=[]), "collator", "exec"), namespace)
        tokenizer = Tokenizer()
        collator = namespace["ChoiceCollator"](tokenizer)
        row = {"input_ids": [[1] * 1024, [99] * 1024],
               "attention_mask": [[1] * 1024, [1] * 1024], "labels": [0, 1]}
        for _ in range(2):
            with self.assertRaises(ReachedPadding):
                collator([row])
        for invalid in (-1, 100, True, 1.5):
            row["input_ids"][0][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "token IDs"):
                collator([row])
        self.assertEqual(tokenizer.length_calls, 1)

    def test_dense_trainer_disables_halo_moe_default(self):
        # Execute the real factory with lightweight stand-ins; no Torch or downloads.
        tree = ast.parse((ROOT / "kev/training.py").read_text())
        factory = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "make_trainer")
        def parallelism_config(use_grouped_gemm=True, **kwargs):
            return SimpleNamespace(needs_ep_wrappers=use_grouped_gemm, **kwargs)
        def trainer(**kwargs):
            self.assertFalse(kwargs["parallelism_config"].needs_ep_wrappers)
            return kwargs
        namespace = {
            "ChoiceTrainer": trainer, "ParallelismConfig": parallelism_config,
            "build_training_args": lambda *args: None,
            "ChoiceCollator": lambda *args: None,
            "FiniteMetricsCallback": object,
        }
        exec(compile(ast.Module(body=[factory], type_ignores=[]), "training_factory", "exec"), namespace)
        namespace["make_trainer"](
            SimpleNamespace(config=SimpleNamespace(num_labels=1)), None,
            {"max_length": 1024, "max_candidates": 16}, "unused", [], [],
        )

    def test_5090_smoke_matches_training_memory_settings(self):
        train = json.loads((ROOT / "configs/train.yaml").read_text())
        smoke = json.loads((ROOT / "configs/smoke.yaml").read_text())
        expected = {"max_length": 1024, "max_candidates": 16,
                    "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 32, "bf16": True,
                    "gradient_checkpointing": True}
        for key, value in expected.items():
            self.assertEqual(train[key], value, key)
            self.assertEqual(smoke[key], value, key)
        self.assertEqual(smoke["max_steps"], 3)

    def test_resume_preserves_other_runs_and_completed_results(self):
        config = {"model_name_or_path": "synthetic", "model_revision": "v1",
                  "max_length": 32, "max_candidates": 3, "seed": 42}
        provenance = {"status": "complete", "config": config}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            self.assertIsNone(validate_run_directory(output, config, provenance))
            checkpoint = output / "checkpoint-2"
            checkpoint.mkdir(parents=True)
            write_json(checkpoint / "trainer_state.json", {"global_step": 2})
            write_json(output / "training_config.json", config)
            write_json(output / "provenance.json", provenance)
            self.assertEqual(validate_run_directory(output, config, provenance, checkpoint), str(checkpoint.resolve()))
            with self.assertRaises(FileExistsError):
                validate_run_directory(output, config, provenance)
            with self.assertRaisesRegex(ValueError, "selected output"):
                validate_run_directory(Path(directory) / "other", config, provenance, checkpoint)
            with self.assertRaisesRegex(ValueError, "different prepared data"):
                validate_run_directory(output, config, {**provenance, "changed": True}, checkpoint)
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_run_directory(output, {**config, "seed": 7}, provenance, checkpoint)
            training_settings = {
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 32,
                "num_train_epochs": 1, "learning_rate": 2e-5, "weight_decay": 0.01,
                "warmup_ratio": 0.03, "bf16": True,
            }
            training_config = {**config, **training_settings}
            write_json(output / "training_config.json", training_config)
            self.assertEqual(validate_run_directory(output, training_config, provenance, checkpoint), str(checkpoint.resolve()))
            for key, value in training_settings.items():
                changed = False if isinstance(value, bool) else value * 2
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate_run_directory(output, {**training_config, key: changed}, provenance, checkpoint)
            write_json(output / "training_config.json", config)
            (output / "final").mkdir()
            with self.assertRaisesRegex(FileExistsError, "exported results"):
                validate_run_directory(output, config, provenance, checkpoint)

    def test_source_and_configs_parse_without_importing_training_code(self):
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(), filename=str(path))
        for path in (ROOT / "configs").glob("*.yaml"):
            config = json.loads(path.read_text())
            self.assertEqual(config["model_name_or_path"], "jhu-clsp/ettin-encoder-400m")
            self.assertEqual(len(config["model_revision"]), 40)
            self.assertLessEqual(config["max_length"], 7999)
            self.assertGreaterEqual(config["max_candidates"], 2)

    def test_preflight_version_parser(self):
        spec = importlib.util.spec_from_file_location("kev_preflight_test", ROOT / "scripts/preflight.py")
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        self.assertEqual(preflight.version_tuple("2.11.0+cu130"), (2, 11, 0))
        self.assertEqual(preflight.version_tuple("5.16.1"), (5, 16, 1))
        for version in ("bad", "2.11", "2.11.0.dev1", "2.11.0rc1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                preflight.version_tuple(version)


if __name__ == "__main__":
    unittest.main()
