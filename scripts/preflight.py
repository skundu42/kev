#!/usr/bin/env python3
"""Check the supplied Halo image without installing or downloading anything."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


VERSIONS = {
    "torch": ((2, 11, 0), (2, 12, 0)),
    "transformers": ((5, 16, 1), (5, 17, 0)),
    "trl": ((1, 6, 0), (1, 7, 0)),
    "accelerate": ((1, 11, 0), (1, 12, 0)),
    "datasets": ((4, 8, 5), (5, 0, 0)),
    "peft": ((0, 18, 1), (0, 19, 0)),
}


def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\+[^ ]+)?$", value)
    if not match:
        raise ValueError(f"Expected a stable package version, got {value!r}")
    return tuple(map(int, match.groups()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-only", action="store_true", help="Check Python/CUDA/package versions before cloning Halo")
    args = parser.parse_args()
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Run this on the RunPod Linux pod in the Halo image with Python 3.12.")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is missing; use the documented Halo image. No packages were installed.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; attach a GPU to the Halo pod before setup.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The selected GPU must support BF16 for these training profiles.")
    # Exercise the image's actual CUDA kernels; recognizing a 5090 is not enough.
    try:
        query = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        attention.float().square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(attention).all() or not torch.isfinite(query.grad).all():
            raise RuntimeError("Nonfinite BF16 SDPA output or gradient")
        del query, attention
    except RuntimeError as exc:
        raise SystemExit("CUDA BF16/SDPA execution failed. Use a Halo Blackwell training image "
                         "and a Runpod driver supporting the RTX 5090. " + str(exc)) from exc
    gpu = torch.cuda.get_device_properties(0)
    report = {
        "python": platform.python_version(),
        "gpu": gpu.name,
        "gpu_memory_gib": round(gpu.total_memory / 2**30, 1),
        "cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_sdpa_backward": "passed",
    }
    versions = {}
    errors = []
    for package, (minimum, maximum) in VERSIONS.items():
        try:
            value = importlib.metadata.version(package)
            versions[package] = value
            if not minimum <= version_tuple(value) < maximum:
                errors.append(f"{package} {value}: expected >= {'.'.join(map(str, minimum))}, < {'.'.join(map(str, maximum))}")
        except (importlib.metadata.PackageNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise SystemExit("Image dependency mismatch:\n" + "\n".join(errors) + "\nUse the documented image; this script never installs packages.")
    report["packages"] = versions
    if not args.runtime_only:
        halo_root = (Path(os.environ.get("KEV_WORKDIR", "/workspace/kev-run")) / "halo").resolve()
        source = importlib.import_module("src")
        source_path = getattr(source, "__file__", None)
        if not source_path or not Path(source_path).resolve().is_relative_to(halo_root):
            raise SystemExit(f"Halo imported from {source_path!r}; expected the pinned checkout at {halo_root}. Check PYTHONPATH.")
        revision = subprocess.check_output(["git", "-C", str(halo_root), "rev-parse", "HEAD"], text=True).strip()
        if revision != "ffc9d46290b62e61150568f3b66b0b1f900b2598":
            raise SystemExit(f"Unexpected Halo revision: {revision}")
        for module in ("transformers", "trl", "accelerate", "datasets", "peft"):
            importlib.import_module(module)
        from transformers import ModernBertModel  # noqa: F401
        from src.configs.classification_config import ClassificationConfig  # noqa: F401
        from src.distributed.parallelism_config import ParallelismConfig  # noqa: F401
        from src.trainers.reward.classification import ClassificationTrainer  # noqa: F401

        report.update(halo_root=str(halo_root), halo_revision=revision)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
