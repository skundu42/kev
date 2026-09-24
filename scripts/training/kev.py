"""Run with halo launch kev CONFIG --root REPO --data-dir DATA --output-dir OUTPUT."""

import argparse
import sys
from pathlib import Path

# The entrypoint is named kev.py; prefer the package over this script's directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kev.core import require_cuda


def main():
    parser = argparse.ArgumentParser(description="Train the Ettin candidate scorer with Halo on one GPU.")
    parser.add_argument("config")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    require_cuda()
    from kev.training import train

    train(args.config, args.data_dir, args.output_dir, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
