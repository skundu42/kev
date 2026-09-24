#!/usr/bin/env python3
"""Run the stdlib-only suite with network disabled. No installation is needed."""

import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def blocked(*args, **kwargs):
    raise RuntimeError("Network access is forbidden in Kev's offline tests")


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", blocked), \
         patch.object(socket.socket, "connect_ex", blocked), \
         patch.object(socket, "create_connection", blocked), \
         patch.object(socket, "getaddrinfo", blocked):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    unexpected = {"torch", "transformers", "datasets"}.intersection(sys.modules)
    if unexpected:
        sys.exit(f"Offline tests imported training dependencies: {sorted(unexpected)}")
    sys.exit(not result.wasSuccessful())
