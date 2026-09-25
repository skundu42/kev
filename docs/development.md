# Development and validation

[Project overview](../README.md)

Run the dependency-free checks locally:

```bash
python3 scripts/test_offline.py
bash -n scripts/runpod.sh
```

These exercise adapters, split/overlap logic, request validation, probability outputs, CPU/GPU command boundaries, and mocked Hub transfers with checksum and interruption checks. They do not download dependencies, weights, tokenizers, or data. Actual CPU package execution and Hub transfers must still be verified on the remote machines. `scripts/check_gpu.py` and the complete `smoke` command are the pod validation path. The optional HTTP server uses FastAPI/Uvicorn; the training and CLI paths do not require them.

When the API requirements and `httpx` are already available, also run the in-process ASGI tests. They inject a fake model, block network access, and reject training-library imports:

```bash
KEV_TEST_API=1 python3 scripts/test_offline.py
```

## Repository layout

```text
kev/           Data adapters, training, inference, API, and evaluation
configs/       Pinned model/data revisions and training profiles
scripts/       Workflow entrypoint and environment/readiness checks
examples/      JSON requests for choice, noul, and score
tests/         Offline tests with synthetic fixtures
docs/          Setup, usage, and methodology guides
```

## Changes and validation

Keep configuration defaults and CLI examples consistent with the code. For changes to data adapters, preserve source labels and partition boundaries. For changes to inference, preserve the Python, CLI, and HTTP request/response formats. Use synthetic fixtures for local tests; perform real model and dataset checks on the remote machine.

See [training readiness](training.md#check-readiness) for GPU checks and [HTTP API](inference.md#http-api) for serving checks.
