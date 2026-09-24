"""Offline readiness orchestration checks; subprocesses never load a real model."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kev.core import load_config, read_rows, write_json


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kev_check_run", ROOT / "scripts/check_run.py")
check_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_run)


def decision(count, length, identifier):
    return {"id": identifier, "state": identifier, "instructions": "Choose an answer",
            "source": "synthetic", "task": "readiness", "kind": "choice",
            "candidates": [f"answer-{index}" for index in range(count)],
            "target": [1.0] + [0.0] * (count - 1),
            "labels": [1.0] + [0.0] * (count - 1),
            "input_ids": [[1] * length for _ in range(count)],
            "attention_mask": [[1] * length for _ in range(count)]}


class CheckRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "prepared"
        self.data.mkdir()
        self.config = {"model_name_or_path": "synthetic", "model_revision": "a" * 40,
                       "max_length": 16, "max_candidates": 3, "seed": 42,
                       "per_device_train_batch_size": 1, "gradient_accumulation_steps": 2,
                       "bf16": True, "gradient_checkpointing": True, "save_total_limit": 2}
        self.config_file = self.root / "config.yaml"
        write_json(self.config_file, self.config)
        rows = {"train": [decision(2, 3, "short"), decision(2, 6, "longest-two")],
                "validation": [decision(3, 12, "longest-three")],
                "calibration": [decision(2, 2, "calibration")],
                "test": [decision(2, 2, "test")]}
        for split, examples in rows.items():
            (self.data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in examples))
        self.manifest = {"status": "complete", "format_version": 1, "config": self.config,
                         "counts": {split: len(examples) for split, examples in rows.items()}, "sources": []}
        write_json(self.data / "manifest.json", self.manifest)

    def test_gpu_guard_rejects_active_or_uninspectable_gpu(self):
        with patch.object(check_run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="")):
            check_run.check_idle_gpu()
        with patch.object(check_run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="999999999\n")), \
             self.assertRaisesRegex(RuntimeError, "already active"):
            check_run.check_idle_gpu()
        with patch.object(check_run.subprocess, "run", side_effect=FileNotFoundError), \
             self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            check_run.check_idle_gpu()

    def test_sample_covers_extremes_in_exact_optimizer_windows(self):
        before = {path.name: path.read_bytes() for path in self.data.iterdir()}
        sample = self.root / "sample"
        config, shapes, counts = check_run.make_sample(self.data, sample, self.config, self.manifest)
        rows = list(read_rows(sample / "train.jsonl"))
        self.assertEqual(counts["train"] * 3, config["max_steps"] * 2)
        self.assertEqual(config["save_total_limit"], 2)
        self.assertEqual({row["id"] for row in rows[:2]}, {"longest-two", "longest-three"})
        self.assertEqual([(shape["candidates"], shape["padded_length"]) for shape in shapes], [(2, 6), (3, 12)])
        for key in ("bf16", "gradient_checkpointing", "max_length", "max_candidates",
                    "per_device_train_batch_size", "gradient_accumulation_steps"):
            self.assertEqual(config[key], self.config[key])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.data.iterdir()})
        with self.assertRaisesRegex(ValueError, "drop_last"):
            check_run.make_sample(self.data, self.root / "unused", {**self.config, "dataloader_drop_last": True}, self.manifest)

    def command_spy(self, command, **kwargs):
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["cwd"], ROOT)
        module = command[2]
        if module == "src.cli":
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(exist_ok=True)
            (output / "final").mkdir()
            write_json(output / "trainer_state.json", {"global_step": 3})
            write_json(output / "train_results.json", {"train_runtime": 2.0})
            write_json(output / "gpu_memory.json", {"peak_allocated_gib": 1.0})
            for step in (1, 3):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir(exist_ok=True)
                write_json(checkpoint / "trainer_state.json", {"global_step": step})
                (checkpoint / "optimizer.pt").touch()
                (checkpoint / "scheduler.pt").touch()
        stdout = json.dumps({"answers": {kind: {"type": kind} for kind in ("choice", "noul", "score")}})
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    def test_check_runs_real_entrypoints_resume_then_all_final_stages(self):
        output = self.root / "check"
        with patch.object(check_run.subprocess, "run", side_effect=self.command_spy) as run, patch("builtins.print"):
            report = check_run.run_check(self.config_file, self.data, output)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([command[2] for command in commands],
                         ["src.cli", "src.cli", "kev.calibrate", "kev.evaluate", "kev.inference"])
        self.assertIn("--resume-from-checkpoint", commands[1])
        self.assertEqual(report["resumed_from_step"], 1)
        self.assertEqual(report["complete_passes"], 3)
        self.assertEqual(load_config(output / "readiness.json")["status"], "passed")
        self.assertTrue((output / "first-export").is_dir())
        self.assertTrue((output / "predictions.json").is_file())
        self.assertTrue(all("kev.prepare" not in command for command in commands))
        with patch.object(check_run.subprocess, "run") as run, self.assertRaises(FileExistsError):
            check_run.run_check(self.config_file, self.data, output)
        run.assert_not_called()

    def test_incomplete_data_or_child_failure_never_reports_passed(self):
        write_json(self.data / "manifest.json", {**self.manifest, "status": "preparing"})
        with patch.object(check_run.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "incomplete"):
            check_run.run_check(self.config_file, self.data, self.root / "not-created")
        run.assert_not_called()
        self.assertFalse((self.root / "not-created").exists())
        write_json(self.data / "manifest.json", self.manifest)
        output = self.root / "failed"
        with patch.object(check_run.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "training")), \
             patch("builtins.print"), \
             self.assertRaises(subprocess.CalledProcessError):
            check_run.run_check(self.config_file, self.data, output)
        self.assertEqual(load_config(output / "readiness.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
