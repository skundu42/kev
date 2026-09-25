# Training on Runpod

[Project overview](../README.md)

## GPU and container setup

The default profiles target **one RTX 5090 with 32 GB VRAM** and also support an RTX 6000 Ada with 48 GB. Create a GPU pod with persistent storage mounted at `/workspace`, select the appropriate Halo image, and set its start command to `sleep infinity`:

| Pod GPU | Container image |
|---|---|
| RTX 5090 32 GB / RTX 6000 Ada 48 GB | `public.ecr.aws/whitecircle/halo:blackwell-1.0.0` |
| H100 / H200 | `public.ecr.aws/whitecircle/halo:hopper-1.0.0` |
| B200 / B300 | `public.ecr.aws/whitecircle/halo:blackwell-1.0.0` |

The architecture-specific image selection follows [Halo's installation instructions](https://github.com/whitecircle/halo/blob/ffc9d46290b62e61150568f3b66b0b1f900b2598/human-docs/installation.md). This repository does not start Docker inside the pod, build an image, or replace the container's Torch/CUDA stack.

## Train from a prepared artifact

Complete [data preparation](data.md) first. Run these commands **in the GPU pod**, replacing `YOUR_USERNAME/kev-prepared` with your dataset repository and the revision placeholder with the exact 40-character SHA printed by `push-data`:

```bash
git clone https://github.com/skundu42/kev.git /workspace/kev
cd /workspace/kev
export KEV_DATA_DIR=/workspace/kev-run/data/mixture-v1
export KEV_RUN_NAME=demo
export HF_HOME=/workspace/kev-run/hf-cache
DATA_REVISION=REPLACE_WITH_40_CHARACTER_COMMIT_FROM_PUSH
hf auth login
bash scripts/runpod.sh setup &&
  bash scripts/runpod.sh pull-data YOUR_USERNAME/kev-prepared "$DATA_REVISION"
```

`pull-data` downloads only the prepared artifact at the specified commit, checks its file hashes and row counts, and records the Hub revision in `hub.json`. Interrupted transfers can resume; requesting the same downloaded revision reuses it, while other existing output directories are protected.

## Check readiness

Before starting a long run, run the real-model check on an **idle GPU** with the same prepared data:

```bash
export KEV_DATA_DIR=/workspace/kev-run/data/mixture-v1
KEV_RUN_NAME="readiness-$(date +%Y%m%d-%H%M%S)" bash scripts/runpod.sh check-run
```

`check-run` verifies dataset checksums, scans the prepared training/validation files for the longest row at each candidate count, and copies a small test subset. It never repeats source preparation. With the default batch size 1 and accumulation 32, it trains 32 decisions for three complete passes, exercising the largest actual shapes again after Adam state allocation. It retains the configured BF16, gradient checkpointing, token and candidate limits. It then resumes a real-model checkpoint and runs native export reload, calibration, evaluation, and choice/noul/score prediction. With larger batch sizes, this cannot cover every possible mixed-batch padding combination.

Read `checks/<name>/readiness.json`: only `status: passed` means every stage finished. It records timings, extreme shapes and peak allocator VRAM; these are software checks, not model-quality metrics or a full-run ETA. The check has its own output directory and preserves existing training runs. It rejects an already-busy GPU. Reserve roughly 20 GB of additional disk space for its full-model checkpoints and two exports. Do not run it alongside an existing training job; stop only after securing a checkpoint if you need to check an ongoing run.

The readiness check has been tested with offline command mocks; real GPU execution must happen on the pod. A pass reduces the risk of startup, memory, checkpoint and export failures but cannot guarantee an uninterrupted long run.

The training config must match the prepared model/tokenizer identity and revision, token limit, and candidate limit. Training settings such as batch size and learning rate can change without preparing again. Source caps, split seed, and other preparation settings are recorded in the artifact's manifest; changing them in a training config does not alter already prepared data. Keep the pinned dataset revision to reproduce the same training inputs.

## Start training

After the readiness check passes, run this in the same shell with the environment above:

```bash
bash scripts/runpod.sh fit
```

`fit` checks the GPU environment, validates the prepared files and their compatibility with the training config, then runs **train → calibrate → evaluate → predict**. It never prepares or rescans source datasets. Model weights are downloaded on the GPU pod as needed, including during the readiness check. The final model is saved in `/workspace/kev-run/runs/demo/final/`.

## Environment validation

`setup` checks Linux, Python 3.12, package versions and imports, then runs a small BF16 SDPA forward/backward pass on the GPU. It clones Halo into `/workspace/kev-run/halo`, checks out `ffc9d46290b62e61150568f3b66b0b1f900b2598`, and refuses a different revision or tracked modifications. This checkout remains available even when the volume hides the image's bundled `/workspace` contents. All commands resolve paths from the script location and configure imports themselves.

The expected image environment is PyTorch 2.11.x, Transformers `>=5.16.1,<5.17`, TRL 1.6.x, Accelerate 1.11.x, Datasets `>=4.8.5,<5.0`, and PEFT `>=0.18.1,<0.19`. A mismatch fails with a diagnostic; setup never silently upgrades the image. Full preflight verifies the imported Halo module belongs to the pinned checkout and prints its commit. The tiny random-model GPU check has been exercised on RTX 6000 Ada, including checkpoint save/reload/resume. Local offline tests do not verify GPU execution; run the readiness check on your target pod.

## Individual stages and checkpoint resume

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

## Single-pod workflows

Preparation and training can also run on the same pod:

```bash
# Small real-data end-to-end check, including preparation on the GPU pod:
KEV_RUN_NAME=smoke KEV_DATA_DIR=/workspace/kev-run/data/smoke bash scripts/runpod.sh smoke
# Full preparation and training on the same GPU pod:
KEV_RUN_NAME=combined KEV_DATA_DIR=/workspace/kev-run/data/combined bash scripts/runpod.sh all
```

`smoke` first checks a tiny random ModernBERT model, then prepares a small mixture and trains the real Ettin model for three optimizer steps before calibration, evaluation, and predictions. It uses the same 1,024-token limit, batch size 1 and accumulation 32 as the training profile. It limits scanning to 2,000 rows per source split, keeps at most 32 training and 16 evaluation examples per source, and selects three BIG-bench configurations. Inspect `runs/smoke/gpu_memory.json` for measured peak allocated/reserved VRAM. Its sampled lengths and candidate counts do not guarantee the worst-case 16-candidate memory fit, and its metrics do not measure model quality.

`all` runs setup → prepare → train → calibrate → evaluate → predict on one GPU pod. Prefer the separate CPU/GPU workflow to avoid GPU charges during preparation. The full preparation profile caps each source at 50,000 training decisions, each aggregate subtask at 5,000, and each source in each evaluation partition at 2,000. Caps are maxima, not guaranteed final counts. This is a bounded demo mixture, not a claim to train on every source row.

## Storage and configuration

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

Use the [inference guide](inference.md) to run or serve the exported model, and the [evaluation guide](evaluation.md) to inspect its quality.
