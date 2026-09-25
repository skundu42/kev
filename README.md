# Kev

A 400M-parameter encoder for decisions over text: choose between options, estimate yes/no probabilities, and score against an ordered rubric.

Kev fine-tunes [Ettin Encoder 400M](https://huggingface.co/jhu-clsp/ettin-encoder-400m) with [Halo](https://github.com/whitecircle/halo). This repository includes data preparation, training, calibration, evaluation, and inference through Python, a JSON CLI, or an authenticated HTTP API.

**Hugging Face:** [Model weights](https://huggingface.co/skundu42/kev) · [Prepared dataset](https://huggingface.co/datasets/skundu42/kev-prepared)

## What it does

Supply text and criteria at inference time; the model scores the candidates you provide.

| Decision | Output | Example |
|---|---|---|
| `choice` | Selected option and probabilities | Route a support request |
| `noul` | Probability of yes | Detect whether escalation is needed |
| `score` | Distribution and expected zero-based rubric index | Rate sentiment or frustration |

The default profile supports **2–16 candidates** and **1,024 tokens per encoded prompt/candidate pair**, including instructions and criteria. Overlength requests are rejected rather than truncated.

## Quickstart

Training and inference run on a Linux CUDA GPU pod with Python 3.12 and the [configured Halo environment](docs/training.md#gpu-and-container-setup). The default training profile targets an RTX 5090 with 32 GB VRAM. Data preparation runs separately on a CPU machine.

**Train a model:** [prepare and publish the dataset](docs/data.md), then [train on Runpod](docs/training.md). The training workflow exports weights, a tokenizer, and calibration settings to `runs/<run-name>/final/`.

**Use an existing export:** from the repository root on your GPU pod, replace the model path below with your completed export:

```python
from kev.inference import DecisionModel

model = DecisionModel.from_pretrained("/workspace/kev-run/runs/demo/final")
result = model.predict(
    state="I was charged twice for my subscription.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Choose the team that should handle this request.",
            "criteria": {
                "billing": "Payments, invoices, and refunds",
                "technical": "Bugs and product support",
                "sales": "Plans and purchasing",
            },
        }
    },
)
print(result)
```

See the [inference guide](docs/inference.md) for all decision types, JSON CLI usage, and HTTP API deployment. A complete request is available in [examples/request.json](examples/request.json).

## Documentation

| Guide | Contents |
|---|---|
| [Data and model design](docs/data.md) | CPU preparation, Hub transfer, datasets, label mappings, and split policy |
| [Training](docs/training.md) | GPU setup, readiness checks, configuration, checkpoints, and resume |
| [Inference and API](docs/inference.md) | Python, CLI, request formats, authentication, and serving |
| [Evaluation](docs/evaluation.md) | Temperature calibration, metrics, and paired Laya comparison |
| [Development](docs/development.md) | Repository layout, offline tests, and validation |

## Development

Run the offline checks without installing training dependencies:

```bash
python3 scripts/test_offline.py
bash -n scripts/runpod.sh
```

The tests use synthetic fixtures and block network access. They do not download models, tokenizers, or datasets. Real GPU validation runs on the pod; see the [development guide](docs/development.md).

## Scope

Kev uses supervised decision training with temperature calibration. It does not claim to reproduce Jev's proprietary training recipe. Model and dataset revisions are pinned in the [training](configs/train.yaml) and [data](configs/data.json) configurations; source licenses and terms continue to apply.
