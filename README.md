# Kev: an Ettin decision-model training demo

Fine-tune [`jhu-clsp/ettin-encoder-400m`](https://huggingface.co/jhu-clsp/ettin-encoder-400m) into a Jev-like model that chooses among runtime criteria, estimates yes/no probabilities, and rates text on an ordered scale. Training runs through [Halo](https://github.com/whitecircle/halo), using its classification trainer with a custom grouped candidate loss.

## Prepare once on a CPU machine

Use a remote Linux machine with Python 3.12, sufficient RAM and disk for preparation, and internet access. No CUDA, GPU, Halo checkout, Torch, or model weights are needed. Preparation downloads source data and the pinned tokenizer. Install the CPU dependencies into a separate virtual environment **on that remote machine**:

```bash
git clone https://github.com/skundu42/kev.git "$HOME/kev"
cd "$HOME/kev"
python3.12 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt

export KEV_WORKDIR="$HOME/kev-run"
export KEV_DATA_DIR="$KEV_WORKDIR/data/mixture-v1"
export HF_HOME="$KEV_WORKDIR/hf-cache"
hf auth login
bash scripts/runpod.sh setup-data
bash scripts/runpod.sh prepare &&
  bash scripts/runpod.sh push-data skundu42/kev-prepared
```

`setup-data` checks the CPU dependencies without installing anything. `prepare` scans, filters, partitions, deduplicates, and tokenizes the mixture. The larger profile scans full source splits before emitting its capped sample, so preparation still takes time on a CPU machine. It does not need a paid GPU while doing this work. `push-data` uploads the completed prepared artifact to a Hugging Face **dataset** repository, creates new repositories as private, refuses existing public repositories, and prints its commit SHA. Keep that SHA for the GPU download. Authentication needs write access to that repository. Source licenses and terms still apply; uploading the mixture does not grant a new license to its contents.

Use a fresh `KEV_DATA_DIR` if a previous preparation was interrupted. Existing output directories are protected, and preparation does not resume midway. The four prepared JSONL splits and `manifest.json` must remain together. To reuse completed data from a previous GPU preparation, set `KEV_DATA_DIR` to that directory and run `push-data` directly; no rescan is required. The returned repository and commit are also saved in `published.json` alongside the prepared files.

## Train separately on RunPod

The default profiles target **one RTX 5090 with 32 GB VRAM** and also support an RTX 6000 Ada with 48 GB. Create a GPU pod with persistent storage mounted at `/workspace`, select the appropriate Halo image, and set its start command to `sleep infinity`:

| Pod GPU | Container image |
|---|---|
| RTX 5090 32 GB / RTX 6000 Ada 48 GB | `public.ecr.aws/whitecircle/halo:blackwell-1.0.0` |
| H100 / H200 | `public.ecr.aws/whitecircle/halo:hopper-1.0.0` |
| B200 / B300 | `public.ecr.aws/whitecircle/halo:blackwell-1.0.0` |

The architecture-specific image selection follows [Halo's installation instructions](https://github.com/whitecircle/halo/blob/ffc9d46290b62e61150568f3b66b0b1f900b2598/human-docs/installation.md). This repository does not start Docker inside the pod, build an image, or replace the container's Torch/CUDA stack.

Run these commands **in the GPU pod**, replacing the revision placeholder with the exact 40-character SHA printed by `push-data`:

```bash
git clone https://github.com/skundu42/kev.git /workspace/kev
cd /workspace/kev
export KEV_DATA_DIR=/workspace/kev-run/data/mixture-v1
export KEV_RUN_NAME=demo
export HF_HOME=/workspace/kev-run/hf-cache
DATA_REVISION=REPLACE_WITH_40_CHARACTER_COMMIT_FROM_PUSH
hf auth login
bash scripts/runpod.sh setup &&
  bash scripts/runpod.sh pull-data skundu42/kev-prepared "$DATA_REVISION" &&
  bash scripts/runpod.sh fit
```

`pull-data` downloads only the prepared artifact at the specified commit, checks its file hashes and row counts, and records the Hub revision in `hub.json`. Interrupted transfers can resume; requesting the same downloaded revision reuses it, while other existing output directories are protected. `fit` checks the GPU environment, validates the prepared files and their compatibility with the training config, then runs **train → calibrate → evaluate → predict**. It never prepares or rescans source datasets. The base model weights are downloaded on the GPU pod when training starts. The final model is saved in `/workspace/kev-run/runs/demo/final/`.

Before committing to a long run, run the real-model check on an **idle GPU** with the same prepared data:

```bash
export KEV_DATA_DIR=/workspace/kev-run/data/mixture-v1
KEV_RUN_NAME="readiness-$(date +%Y%m%d-%H%M%S)" bash scripts/runpod.sh check-run
```

`check-run` verifies dataset checksums, scans the prepared training/validation files for the longest row at each candidate count, and copies a small test subset. It never repeats source preparation. With the default batch size 1 and accumulation 32, it trains 32 decisions for three complete passes, exercising the largest actual shapes again after Adam state allocation. It retains the configured BF16, gradient checkpointing, token and candidate limits. It then resumes a real-model checkpoint and runs native export reload, calibration, evaluation, and choice/noul/score prediction. With larger batch sizes, this cannot cover every possible mixed-batch padding combination.

Read `checks/<name>/readiness.json`: only `status: passed` means every stage finished. It records timings, extreme shapes and peak allocator VRAM; these are software checks, not model-quality metrics or a full-run ETA. The check has its own output directory and preserves existing training runs. It rejects an already-busy GPU. Reserve roughly 20 GB of additional disk space for its full-model checkpoints and two exports. Do not run it alongside an existing training job; stop only after securing a checkpoint if you need to check an ongoing run.

The new check has been tested with offline command mocks; real GPU execution must happen on the pod. A pass reduces the risk of startup, memory, checkpoint and export failures but cannot guarantee an uninterrupted long run.

The training config must match the prepared model/tokenizer identity and revision, token limit, and candidate limit. Training settings such as batch size and learning rate can change without preparing again. Source caps, split seed, and other preparation settings are recorded in the artifact's manifest; changing them in a training config does not alter already prepared data. Keep the pinned dataset revision to reproduce the same training inputs.

`setup` checks Linux, Python 3.12, package versions and imports, then runs a small BF16 SDPA forward/backward pass on the GPU. It clones Halo into `/workspace/kev-run/halo`, checks out `ffc9d46290b62e61150568f3b66b0b1f900b2598`, and refuses a different revision or tracked modifications. This checkout remains available even when the volume hides the image's bundled `/workspace` contents. All commands resolve paths from the script location and configure imports themselves.

The expected image environment is PyTorch 2.11.x, Transformers `>=5.16.1,<5.17`, TRL 1.6.x, Accelerate 1.11.x, Datasets `>=4.8.5,<5.0`, and PEFT `>=0.18.1,<0.19`. A mismatch fails with a diagnostic; setup never silently upgrades the image. Full preflight verifies the imported Halo module belongs to the pinned checkout and prints its commit. The user has verified the tiny random-model GPU check on RTX 6000 Ada, including checkpoint save/reload/resume; the full Ettin workflow has not been benchmarked or verified by the local offline tests.

## Individual stages and existing workflows

```bash
bash scripts/runpod.sh setup-data
bash scripts/runpod.sh prepare
bash scripts/runpod.sh push-data OWNER/DATASET
bash scripts/runpod.sh pull-data OWNER/DATASET COMMIT_SHA
bash scripts/runpod.sh train
bash scripts/runpod.sh calibrate
bash scripts/runpod.sh evaluate
bash scripts/runpod.sh predict examples/request.json
bash scripts/runpod.sh resume /workspace/kev-run/runs/demo/checkpoint-500
# Equivalent resume:
bash scripts/runpod.sh train --resume-from-checkpoint /workspace/kev-run/runs/demo/checkpoint-500
```

`train` runs training only; `fit` runs training and the following calibration/evaluation/prediction stages. Resume is for interrupted training and requires a checkpoint inside the selected run directory with matching data provenance. After resumed training succeeds, run `calibrate`, `evaluate`, and `predict` individually. Completed `final/` exports and evaluation reports are protected; start a fresh run for a new experiment.

Training rechecks bundled dataset hashes before loading the model. Resume also requires the original batch, accumulation, precision, optimizer and scheduler settings so consumed-example positions and optimizer state stay consistent. Nonfinite logged loss, gradient norm or validation loss stops the run. Exports are assembled in a temporary sibling directory and published as `final/` only when weights, tokenizer and inference metadata are complete, so a failed export does not block checkpoint recovery.

The original single-pod commands remain available for convenience:

```bash
# Small real-data end-to-end check, including preparation on the GPU pod:
KEV_RUN_NAME=smoke KEV_DATA_DIR=/workspace/kev-run/data/smoke bash scripts/runpod.sh smoke
# Full preparation and training on the same GPU pod:
KEV_RUN_NAME=combined KEV_DATA_DIR=/workspace/kev-run/data/combined bash scripts/runpod.sh all
```

`smoke` first checks a tiny random ModernBERT model, then prepares a small mixture and trains the real Ettin model for three optimizer steps before calibration, evaluation, and predictions. It uses the same 1,024-token limit, batch size 1 and accumulation 32 as the training profile. It limits scanning to 2,000 rows per source split, keeps at most 32 training and 16 evaluation examples per source, and selects three BIG-bench configurations. Inspect `runs/smoke/gpu_memory.json` for measured peak allocated/reserved VRAM. Its sampled lengths and candidate counts do not guarantee the worst-case 16-candidate memory fit, and its metrics do not measure model quality.

`all` runs setup → prepare → train → calibrate → evaluate → predict on one GPU pod. Prefer the separate CPU/GPU workflow to avoid GPU charges during preparation. The full preparation profile caps each source at 50,000 training decisions, each aggregate subtask at 5,000, and each source in each evaluation partition at 2,000. Caps are maxima, not guaranteed final counts. This is a bounded demo mixture, not a claim to train on every source row.

Use `KEV_WORKDIR` to change storage, `KEV_DATA_DIR` to select a reusable prepared artifact independently of the run, `KEV_RUN_NAME` for a fresh training output directory, and `KEV_CONFIG` for a different profile. `KEV_WORKDIR` and `KEV_DATA_DIR` must be absolute paths. `smoke` defaults to the run name `smoke`; other commands default to `demo`. `HF_HOME` and `HF_DATASETS_CACHE` are respected when supplied; otherwise they default under `KEV_WORKDIR`. Use the same `HF_HOME` for `hf auth login` as for the scripts. Existing prepared directories and training runs are preserved; evaluation does not overwrite an existing report.

Default storage layout (the examples above select `data/mixture-v1` explicitly):

```text
/workspace/kev-run/
  halo/                   pinned Halo checkout, GPU stages only
  hf-cache/               Hugging Face cache and saved authentication
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

These exercise adapters, split/overlap logic, request validation, probability outputs, CPU/GPU command boundaries, and mocked Hub transfers with checksum and interruption checks. They do not download dependencies, weights, tokenizers, or data. Actual CPU package execution and Hub transfers must still be verified on the remote machines. `scripts/check_gpu.py` and the complete `smoke` command are the pod validation path. There is no UI, serving framework, RL environment, or local training environment to maintain.
