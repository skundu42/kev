# Kev

A 400M-parameter encoder for decisions over text: choose between options, estimate yes/no probabilities, and score against an ordered rubric.

Kev fine-tunes [Ettin Encoder 400M](https://huggingface.co/jhu-clsp/ettin-encoder-400m) with [Halo](https://github.com/whitecircle/halo). This repository includes data preparation, training, calibration, evaluation, and inference through Python, a JSON CLI, or an authenticated HTTP API.

**Hugging Face:** [Model card and weights](https://huggingface.co/skundu42/kev) · [Prepared dataset](https://huggingface.co/datasets/skundu42/kev-prepared)

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

## Benchmarks

Kev performs well on its own held-out task mixture, but the external phishing and typed-decision evaluations show weak generalization. **These results do not support a universal “Kev is better than Jev” claim.** All completed benchmarks, per-source results, metric definitions, revisions and aggregate JSON reports are in the [benchmark record](docs/benchmarks/README.md).

### Completed comparisons on Kev’s held-out mixture

| Experiment | Decisions | Kev accuracy | Comparator accuracy | Evidence |
|---|---:|---:|---:|---|
| Kev vs Laya English | 17,476 | 79.93% | 51.63% | Published paired RTX 4090 run |
| Kev vs Jev 1.13 pilot | 100 | 80.00% | 81.00% | Completed source-balanced pilot |
| Kev vs Jev 1.13 full | 17,476 | 79.96% | 73.54% | Completed paired M3 Pro / OpenRouter run |

The pilot is part of the full test pool, not independent confirmation. The historical standalone export evaluation recorded **79.86%** accuracy; it is a separate run from these comparisons. Kev’s task families and calibration are in-domain; comparator training overlap is unknown. The Laya comparison altered 1,597 inputs under Laya’s native limits; on the shared 15,879 unaltered rows, accuracy was 79.44% versus 52.93%.

| Full Kev/Jev run | Kev | Jev |
|---|---:|---:|
| Brier score ↓ | 0.2769 | 0.3823 |
| ECE-15, target mass ↓ | 0.0151 | 0.0979 |
| Ordinal MAE, 997 scoring decisions ↓ | 0.3923 | 0.3564 |
| Median request latency | 132.7 ms | 388.4 ms |
| p95 request latency | 607.3 ms | 499.5 ms |

Kev’s accuracy lead was **6.41 percentage points**, but Jev had lower ordinal error and won on HellaSwag, ANLI, BoolQ and defeasible NLI. Much of Kev’s aggregate advantage came from FOL-NLI and zero-shot-label-NLI. See the [per-source breakdown and probability-rounding caveats](docs/benchmarks/README.md#paired-kev-vs-jev).

Latency is not a pure model-speed comparison: Kev ran sequentially in FP32 on one Apple M3 Pro GPU, while Jev used four concurrent remote requests and includes network time. Kev’s summed request time was **64.29 minutes** (221 ms mean, 54,402 candidate pairs). Model loading and warmups were excluded.

### External evaluation: Luni benchmark suite

Kev was tested sequentially with the prompts from [Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark/tree/d75081b2a4b2ad772793d6a7f5f5b4fdca00d557), without fine-tuning, recalibration, truncation or dropped cases. **Jev was not called for these tests**; its numbers below are the repository’s published reference figures, not a new paired comparison.

| Benchmark | Kev measured | Jev published reference |
|---|---:|---:|
| PhishNChips core: 2,000 emails, accuracy | 50.25% | 62.6% |
| Phishing AUROC | 0.3482 | 0.689 |
| Typed decisions: 400 cases / 2,000 decisions, accuracy | 44.15% | 72.7% |

Kev flagged **995 of 1,000 legitimate emails as phishing**. Typed scoring MAE was **0.6105** across 800 score decisions. Median latency was **123.4 ms/email** and **2,008.5 ms/five-question case**; measured inference totaled approximately **18 minutes**. Small behavioral diagnostics passed grounding **4/5**, complement consistency **3/3** under the specified tolerance, and routing variants **1/3**. These probes are not a broad benchmark.

The mixture accuracy rule accepts any positive-target candidate; external typed accuracy uses exact gold labels. The suites therefore measure different tasks and label conventions. [Full external results and methodology](docs/benchmarks/README.md#external-evaluation-luni-benchmark-suite).

## Documentation

The [published model card](https://huggingface.co/skundu42/kev) includes checkpoint details, Hub download instructions, training provenance, evaluation results, and limitations. The benchmark record above includes newer local evaluations and should be read alongside it.

| Guide | Contents |
|---|---|
| [Data and model design](docs/data.md) | CPU preparation, Hub transfer, datasets, label mappings, and split policy |
| [Training](docs/training.md) | GPU setup, readiness checks, configuration, checkpoints, and resume |
| [Inference and API](docs/inference.md) | Python, CLI, request formats, authentication, and serving |
| [Evaluation](docs/evaluation.md) | Temperature calibration, metrics, and paired Laya comparison |
| [Benchmark record](docs/benchmarks/README.md) | Complete results, external failures, run status, revisions, and methodology |
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
