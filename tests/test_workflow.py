"""Exercise shell stage boundaries with command spies, never real downloads or GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        bin_dir = self.base / "bin"
        bin_dir.mkdir()
        (self.base / "halo/.git").mkdir(parents=True)
        self.log = self.base / "calls.jsonl"
        spy = f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
with open(os.environ["TEST_CALL_LOG"], "a") as log:
    log.write(json.dumps({"tool": pathlib.Path(sys.argv[0]).name, "args": sys.argv[1:],
                          "hf_home": os.environ.get("HF_HOME"),
                          "datasets_cache": os.environ.get("HF_DATASETS_CACHE")}) + "\\n")
if pathlib.Path(sys.argv[0]).name == "git":
    if "rev-parse" in sys.argv:
        print("ffc9d46290b62e61150568f3b66b0b1f900b2598")
    elif "diff" not in sys.argv:
        sys.exit("Unexpected git operation: " + repr(sys.argv))
if os.environ.get("TEST_REJECT_VALIDATE") and "validate" in sys.argv:
    sys.exit(7)
'''
        for tool in ("python", "git"):
            executable = bin_dir / tool
            executable.write_text(spy)
            executable.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "TEST_CALL_LOG": str(self.log), "KEV_WORKDIR": str(self.base),
                    "KEV_RUN_NAME": "experiment", "KEV_DATA_DIR": str(self.base / "shared-data"),
                    "KEV_CONFIG": str(ROOT / "configs/train.yaml"),
                    "HF_HOME": str(self.base / "existing-login"),
                    "HF_DATASETS_CACHE": str(self.base / "existing-cache")}

    def run_stage(self, *args):
        self.log.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(ROOT / "scripts/runpod.sh"), *args],
                                env=self.env, text=True, capture_output=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_cpu_and_transfer_stages_never_call_gpu_preflight_or_halo(self):
        for args in (("setup-data",), ("prepare",), ("push-data", "owner/data"),
                     ("pull-data", "owner/data", "a" * 40)):
            with self.subTest(args=args):
                result, calls = self.run_stage(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(calls)
                self.assertTrue(all(call["tool"] == "python" for call in calls))
                self.assertNotIn(str(ROOT / "scripts/preflight.py"), str(calls))
                for call in calls:
                    self.assertEqual(call["hf_home"], self.env["HF_HOME"])
                    self.assertEqual(call["datasets_cache"], self.env["HF_DATASETS_CACHE"])
                if args[0] != "setup-data":
                    self.assertIn(self.env["KEV_DATA_DIR"], calls[-1]["args"])

    def test_fit_validates_before_launch_and_never_prepares(self):
        result, calls = self.run_stage("fit")
        self.assertEqual(result.returncode, 0, result.stderr)
        modules = [call["args"][1] for call in calls if call["args"][0] == "-m"]
        self.assertEqual(modules, ["kev.hub", "src.cli", "kev.calibrate", "kev.evaluate", "kev.inference"])
        validation = next(call["args"] for call in calls if "validate" in call["args"])
        self.assertIn(self.env["KEV_DATA_DIR"], validation)
        self.env["TEST_REJECT_VALIDATE"] = "1"
        result, calls = self.run_stage("fit")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("src.cli" in call["args"] for call in calls))

    def test_check_run_uses_isolated_outputs_and_existing_data(self):
        result, calls = self.run_stage("check-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[-1]["args"], [str(ROOT / "scripts/check_run.py"),
                         "--config", self.env["KEV_CONFIG"],
                         "--data-dir", self.env["KEV_DATA_DIR"],
                         "--output-dir", str(self.base / "checks/experiment")])
        self.assertFalse(any("kev.prepare" in call["args"] or "src.cli" in call["args"] for call in calls))

    def test_prepare_preserves_existing_data_and_transfer_requires_arguments(self):
        Path(self.env["KEV_DATA_DIR"]).mkdir()
        result, calls = self.run_stage("prepare")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("kev.prepare" in call["args"] for call in calls))
        for args in (("push-data",), ("pull-data", "owner/data")):
            result, calls = self.run_stage(*args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
