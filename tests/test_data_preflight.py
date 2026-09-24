from contextlib import contextmanager, redirect_stdout
import importlib.metadata
import io
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

from scripts import preflight_data


class DataPreflightTests(unittest.TestCase):
    @contextmanager
    def environment(self, versions):
        modules = {name: types.ModuleType(name) for name in
                   ("packaging", "packaging.specifiers", "datasets", "huggingface_hub", "transformers")}
        # Test the preflight's version/import flow; packaging itself is external.
        accepted = {"==5.16.1": {"5.16.1"}, "==4.8.5": {"4.8.5"}, ">=1.5.0,<2.0": {"1.5.0"}}
        modules["packaging.specifiers"].SpecifierSet = Mock(side_effect=accepted.__getitem__)
        for package, names in {
            "datasets": ("get_dataset_config_names", "load_dataset"),
            "huggingface_hub": ("HfApi", "snapshot_download"),
            "transformers": ("AutoTokenizer",),
        }.items():
            for name in names:
                setattr(modules[package], name, Mock(side_effect=AssertionError("Preflight must only import APIs")))

        def version(package):
            if package not in versions:
                raise importlib.metadata.PackageNotFoundError(package)
            return versions[package]

        with patch.dict(sys.modules, modules), patch.object(sys, "version_info", (3, 12, 3)), \
                patch.object(preflight_data.importlib.metadata, "version", side_effect=version):
            yield

    def test_cpu_preflight_needs_no_torch_or_asset_loading(self):
        self.assertNotIn("torch", sys.modules)
        output = io.StringIO()
        with self.environment({"transformers": "5.16.1", "datasets": "4.8.5", "huggingface-hub": "1.5.0"}), \
                redirect_stdout(output):
            preflight_data.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["device"], "cpu")
        self.assertFalse(report["model_weights_required"])
        for name in ("torch", "datasets", "transformers", "huggingface_hub"):
            self.assertNotIn(name, sys.modules)

    def test_bad_or_missing_dependency_fails_with_install_help(self):
        for hub_version in ("2.0.0", None):
            versions = {"transformers": "5.16.1", "datasets": "4.8.5"}
            if hub_version is not None:
                versions["huggingface-hub"] = hub_version
            with self.subTest(hub_version=hub_version), self.environment(versions), \
                    self.assertRaises(SystemExit) as error:
                preflight_data.main()
            self.assertIn("huggingface-hub", str(error.exception))
            self.assertIn("pip install -r requirements-data.txt", str(error.exception))
            self.assertIn("Do not install", str(error.exception))


if __name__ == "__main__":
    unittest.main()
