# Kev: an Ettin decision-model training demo

Fine-tune [`jhu-clsp/ettin-encoder-400m`](https://huggingface.co/jhu-clsp/ettin-encoder-400m) into a Jev-like model that chooses among runtime criteria, estimates yes/no probabilities, and rates text on an ordered scale. Training runs through [Halo](https://github.com/whitecircle/halo), using its classification trainer with a custom grouped candidate loss.

This is a supervised demonstration, not a reproduction of Jev's private training recipe or RLCD. It does not generate free-form answers. All model/data downloads and GPU work run **inside the RunPod pod**; the local repository contains code, configs, and offline checks only. A GPU training run has not been performed while creating this repository.

## Run on RunPod

The default profiles target **one RTX 5090 with 32 GB VRAM**. Create a GPU pod with a persistent volume mounted at `/workspace`, select `public.ecr.aws/whitecircle/halo:blackwell-1.0.0`, and set its start command to `sleep infinity`:

| Pod GPU | Container image |
|---|---|
| **RTX 5090 32 GB (default target)** | `public.ecr.aws/whitecircle/halo:blackwell-1.0.0` |
| H100 / H200 | `public.ecr.aws/whitecircle/halo:hopper-1.0.0` |
| B200 / B300 | `public.ecr.aws/whitecircle/halo:blackwell-1.0.0` |

The architecture-specific image tags come from [Halo's installation instructions](https://github.com/whitecircle/halo/blob/ffc9d46290b62e61150568f3b66b0b1f900b2598/human-docs/installation.md). This repository does not start Docker inside the pod, build an image, or install packages. GPU memory requirements and throughput must be measured with the smoke run; no fit or speed claims are made here.

Clone this repository and run these commands **in the pod terminal**:

```bash
git clone https://github.com/skundu42/kev.git /workspace/kev
cd /workspace/kev
bash scripts/runpod.sh setup
bash scripts/runpod.sh smoke
bash scripts/runpod.sh all
```

`setup` checks Linux, Python 3.12, package versions, and imports, then executes a small BF16 SDPA forward/backward pass to check the installed CUDA kernels on the actual GPU. It reports compute capability and does not load a model or train. It clones Halo into `/workspace/kev-run/halo`, checks out `ffc9d46290b62e61150568f3b66b0b1f900b2598`, and refuses a different revision or tracked modifications. This checkout remains available even when the volume hides the image's own `/workspace` contents. All commands resolve paths from the script location and set `PYTHONPATH` to the Halo checkout and this repository.

The expected image environment is PyTorch 2.11.x, Transformers `>=5.16.1,<5.17`, TRL 1.6.x, Accelerate 1.11.x, Datasets `>=4.8.5,<5.0`, and PEFT `>=0.18.1,<0.19`. A mismatch fails with a diagnostic; setup never silently upgrades the image. Full preflight verifies the imported Halo module belongs to the pinned checkout and prints its commit.

`smoke` first checks a tiny random ModernBERT model, then prepares a small mixture, trains the real Ettin model for three optimizer steps using the same 1,024-token limit, batch size 1, and accumulation 32 as the training profile, then calibrates, evaluates, and predicts. It downloads the real tokenizer, model weights, and required dataset shards on the pod. It limits scanning to 2,000 rows per source split, keeps at most 32 training and 16 evaluation examples per source, and selects three BIG-bench configurations. Its metrics check the pipeline, not model quality. Inspect `runs/smoke/gpu_memory.json` for measured peak allocated/reserved VRAM before the longer run. Actual sample lengths and candidate counts vary, so the sampled smoke run does not guarantee the worst-case 16-candidate memory fit.

`all` uses the larger demo profile and runs setup → prepare → train → calibrate → evaluate → predict. The training profile caps each source at 50,000 training decisions, each aggregate subtask at 5,000, and each source in each evaluation partition at 2,000. Caps are maxima, not guaranteed final counts. It scans full source splits to sample deterministically; downloading/preparing the mixture can take substantially longer than the smoke run. This is a bounded demo mixture, not a claim to train on every row.

Separate stages and resume:

Resume is for interrupted runs and requires a checkpoint inside the selected run directory with matching data provenance. Completed `final/` exports and evaluation reports are protected; start a fresh run for a new experiment.

```bash
bash scripts/runpod.sh prepare
bash scripts/runpod.sh train
bash scripts/runpod.sh calibrate
bash scripts/runpod.sh evaluate
bash scripts/runpod.sh predict examples/request.json
bash scripts/runpod.sh resume /workspace/kev-run/runs/demo/checkpoint-500
# Equivalent:
bash scripts/runpod.sh train --resume-from-checkpoint /workspace/kev-run/runs/demo/checkpoint-500
```

Use `export KEV_WORKDIR=/workspace/another-volume/kev-run` to change storage, `export KEV_RUN_NAME=experiment-2` for a fresh run, and `export KEV_CONFIG=configs/train.yaml` for a different profile. `smoke` defaults to run name `smoke`; other commands default to `demo`. To inspect its output, use `KEV_RUN_NAME=smoke bash scripts/runpod.sh predict`. Existing prepared directories and training runs are preserved; use a new name or an explicit checkpoint to resume. Evaluation does not overwrite an existing report.

Storage defaults:

```text
/workspace/kev-run/
  halo/                   pinned Halo checkout
  hf-cache/               model/tokenizer/dataset cache
  data/demo/              JSONL splits, tokenized data, preparation manifest
  runs/demo/              resumable checkpoint-* directories
    final/                exported model, tokenizer, Kev config and temperature
    evaluation.json       held-out metrics
    gpu_memory.json       observed PyTorch peak VRAM during training/validation
```

## Model and data

The model scores each `(state + instructions + complete criteria, candidate)` pair with the same Ettin encoder and scalar head. A masked softmax compares only that example's valid candidates. Cross-entropy trains against a one-hot or normalized soft target, so a candidate's position is not a fixed output class. There are 2–16 candidates per decision. The initial scalar head is new; Ettin supplies the pretrained encoder weights.

The RTX 5090 configuration uses 1,024 tokens, BF16 autocast, SDPA attention, gradient checkpointing, training/evaluation batch size 1, accumulation 32 (32 decisions per full optimizer step), learning rate `2e-5`, and one epoch. A single decision can contain up to 16 candidate pairs. Model weights and AdamW states remain FP32; all model parameters are trained. Inference and calibration process at most eight candidate pairs per forward pass by default. The token limit applies to each encoded prompt/candidate pair. Preparation drops overlength examples and records them instead of silently truncating evidence or candidate descriptions. Inference rejects overlength requests. Increasing `max_length` requires preparing fresh data and more GPU memory.

Every requested dataset has an explicit adapter and a pinned revision in [`configs/data.json`](configs/data.json):

| Dataset | Decision supervision |
|---|---|
| [zero-shot-label-nli](https://huggingface.co/datasets/tasksource/zero-shot-label-nli) | Existing premise/hypothesis relation labels; retain task identity and remove known overlapping task families. |
| [tasksource-instruct-v0](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) | Curated Yelp stars and tweet sentiment tasks provide ordered scores; other instruction tasks are counted and excluded. |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | Entailment / neutral / contradiction; matched and mismatched validation retain their source split names. |
| [ANLI](https://huggingface.co/datasets/facebook/anli) | Three-way NLI from all three rounds; dev/test remain separate from training. |
| [defeasible-nli](https://huggingface.co/datasets/tasksource/defeasible-nli) | Whether an update strengthens or weakens a hypothesis; include atomic, snli, and social configurations. |
| [FOL-nli](https://huggingface.co/datasets/tasksource/FOL-nli) | Three-way NLI using natural-language premises and hypotheses; proofs are not model inputs. |
| [doc-nli](https://huggingface.co/datasets/tasksource/doc-nli) | Binary entailment / not_entailment; never relabel non-entailment as contradiction. |
| [BIG-bench](https://huggingface.co/datasets/tasksource/bigbench) | Discover configurations; normalize informative, nonnegative multiple-choice scores into soft targets. Count/drop open-ended and unsuitable rows. |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | Select one of four continuations. The unlabeled test split is excluded. |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | Yes/no supervision over passage and question, used for `noul` probabilities. |

Defeasible strengthener/weakener labels describe a change in plausibility, not entailment/contradiction. Its social configuration has no premise. Instruction text is not presumed to be arbitrary classification data: only the documented allowlist is parsed into ordered labels. Source dataset licenses and terms continue to apply; prepared data and model weights are not committed here.

Preparation preserves native split roles, groups related source content, removes exact normalized content overlap across splits, and excludes known aggregate/native duplicate families. Native validation is split deterministically into validation/calibration; when no labeled native test exists, it supplies validation/calibration/test with a 50/25/25 split. For sources with a labeled native test the validation/calibration split is 50/50. This is exact-content protection, not semantic deduplication or a guarantee against all upstream contamination. BIG-bench is training data in this recipe, so its used tasks are not a clean external benchmark.

The preparation manifest records pinned sources, configs, counts, exclusions, and preparation settings. Inspect it before a long training run. No dataset is downloaded locally by the repository's tests.

## Inference, calibration, evaluation

The direct CLI reads one JSON request from a file or stdin:

```bash
python -m kev.inference --model /workspace/kev-run/runs/demo/final --input examples/request.json
cat examples/request.json | python -m kev.inference --model /workspace/kev-run/runs/demo/final
```

The same interface is available from Python on the pod:

```python
import json
from pathlib import Path
from kev.inference import DecisionModel

request = json.loads(Path("examples/request.json").read_text())
model = DecisionModel.from_pretrained("/workspace/kev-run/runs/demo/final")
result = model.predict(request["state"], request["questions"])
print(json.dumps(result, indent=2))
```

Requests contain `state` and a `questions` object. Each question has `type`, `instructions`, and criteria:

```json
{
  "state": "The delivery arrived two days early and everything worked.",
  "questions": {
    "sentiment": {
      "type": "choice",
      "instructions": "Classify the review sentiment.",
      "criteria": {"negative": null, "neutral": null, "positive": null}
    },
    "satisfied": {
      "type": "noul",
      "instructions": "Is the reviewer satisfied?",
      "criteria": {"false": "No", "true": "Yes"}
    },
    "rating": {
      "type": "score",
      "instructions": "Rate the review sentiment.",
      "criteria": ["negative", "neutral", "positive"]
    }
  }
}
```

- `choice`: returns the selected criterion name and probabilities over the supplied criteria.
- `noul`: returns the probability of `true`, between 0 and 1. Its criteria are exactly `false` and `true`; omitting them defaults to No/Yes.
- `score`: returns the expected **zero-based criterion index** and its distribution, not an independently learned continuous scale. A five-point rubric has scores from 0 to 4.
- For choice/score, `confidence = (K × max_probability − 1) / (K − 1)`: zero at a uniform distribution and one at certainty. This is a confidence summary, not a validated probability of correctness.

Calibration fits one positive temperature on the separate calibration split and saves it with the final model. It adjusts the softmax distribution without changing the winning candidate. Calibration is approximate: hard labels and limited training sources do not make arbitrary yes/no answers trustworthy probabilities.

```bash
python -m kev.calibrate --model /workspace/kev-run/runs/demo/final --data /workspace/kev-run/data/demo/calibration.jsonl
python -m kev.evaluate --model /workspace/kev-run/runs/demo/final --data /workspace/kev-run/data/demo/test.jsonl --output /workspace/kev-run/runs/demo/evaluation.json
```

Evaluation emits both uncalibrated and calibrated metrics, grouped overall and by source, kind, and task:

| Metric | Definition |
|---|---|
| `accuracy` | Fraction whose winning candidate has positive target mass; this accepts any source-approved answer in soft-target tasks. |
| `log_loss` | Mean cross-entropy against the complete target distribution. |
| `brier_score` | Mean of the sum of squared probability errors across candidates. |
| `ece_15` | Fifteen-bin calibration gap between maximum predicted probability and target mass at the winning candidate. |
| `ordinal_mae` | For score rows, mean absolute error between predicted and target expected zero-based indices; `ordinal_count` gives the sample count. |

Temperature fitting searches `[0.05, 20]` on calibration data and includes temperature 1 as a baseline. Test data is not used to fit it or select training checkpoints. Small smoke results are not performance estimates.

## Development and checks

Run the dependency-free checks locally:

```bash
python3 scripts/test_offline.py
bash -n scripts/runpod.sh
```

These exercise adapters, split/overlap logic, request validation, and probability outputs without downloading dependencies, weights, tokenizers, or data. `scripts/check_gpu.py` and the complete `smoke` command are the pod validation path. There is no UI, serving framework, RL environment, or local training environment to maintain.
