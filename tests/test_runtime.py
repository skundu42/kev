import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from kev.core import validate_run_directory, write_json


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
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
