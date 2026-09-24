#!/usr/bin/env python3
"""Check the CPU data environment without downloading assets or requiring CUDA."""

import importlib.metadata
import json
import platform
import sys


REQUIREMENTS = {
    "transformers": "==5.16.1",
    "datasets": "==4.8.5",
    "huggingface-hub": ">=1.5.0,<2.0",
}
INSTALL_HELP = (
    "On the remote CPU machine, create a Python 3.12 virtual environment and run:\n"
    "  python -m pip install -r requirements-data.txt\n"
    "Do not install these requirements over the Halo GPU image's dependencies."
)


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("CPU preparation requires Python 3.12.\n" + INSTALL_HELP)
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError as exc:
        raise SystemExit("CPU preparation dependencies are missing.\n" + INSTALL_HELP) from exc

    versions, errors = {}, []
    for package, specifier in REQUIREMENTS.items():
        try:
            value = importlib.metadata.version(package)
            versions[package] = value
            if value not in SpecifierSet(specifier):
                errors.append(f"{package} {value}: expected {specifier}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is missing")
    if errors:
        raise SystemExit("CPU data dependency mismatch:\n" + "\n".join(errors) + "\n" + INSTALL_HELP)

    try:
        # Import the actual APIs; never call from_pretrained/load_dataset here.
        # These packages support a clean CPU environment without Torch installed.
        from datasets import get_dataset_config_names, load_dataset  # noqa: F401
        from huggingface_hub import HfApi, snapshot_download  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
    except (ImportError, RuntimeError, OSError) as exc:
        raise SystemExit(f"CPU data imports failed: {exc}\n" + INSTALL_HELP) from exc

    print(json.dumps({
        "python": platform.python_version(),
        "packages": versions,
        "device": "cpu",
        "model_weights_required": False,
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
