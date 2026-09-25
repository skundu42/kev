# Data preparation and model design

[Project overview](../README.md)

Prepare the mixture on a remote CPU machine, then transfer the completed artifact to the GPU pod through the Hugging Face Hub.

## Prepare and publish

Replace `YOUR_USERNAME/kev-prepared` with a dataset repository you can write to.

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
  bash scripts/runpod.sh push-data YOUR_USERNAME/kev-prepared
```

`setup-data` checks the CPU dependencies without installing anything. `prepare` scans, filters, partitions, deduplicates, and tokenizes the mixture. The larger profile scans full source splits before emitting its capped sample, so preparation still takes time on a CPU machine. It does not need a paid GPU while doing this work. `push-data` uploads the completed prepared artifact to a Hugging Face **dataset** repository, creates new repositories as private, refuses existing public repositories, and prints its commit SHA. Keep that SHA for the GPU download. Authentication needs write access to that repository. Source licenses and terms still apply; uploading the mixture does not grant a new license to its contents.

Use a fresh `KEV_DATA_DIR` if a previous preparation was interrupted. Existing output directories are protected, and preparation does not resume midway. The four prepared JSONL splits and `manifest.json` must remain together. To reuse completed data from a previous GPU preparation, set `KEV_DATA_DIR` to that directory and run `push-data` directly; no rescan is required. The returned repository and commit are also saved in `published.json` alongside the prepared files.

## Model and datasets

The model scores each `(state + instructions + complete criteria, candidate)` pair with the same Ettin encoder and scalar head. A masked softmax compares only that example's valid candidates. Cross-entropy trains against a one-hot or normalized soft target, so a candidate's position is not a fixed output class. There are 2–16 candidates per decision. The initial scalar head is new; Ettin supplies the pretrained encoder weights.

The RTX 5090 configuration uses 1,024 tokens, BF16 autocast, SDPA attention, gradient checkpointing, training/evaluation batch size 1, accumulation 32 (32 decisions per full optimizer step), learning rate `2e-5`, and one epoch. A single decision can contain up to 16 candidate pairs. Model weights and AdamW states remain FP32; all model parameters are trained. Inference and calibration process at most eight candidate pairs per forward pass by default. The token limit applies to each encoded prompt/candidate pair. Preparation drops overlength examples and records them instead of silently truncating evidence or candidate descriptions. Inference rejects overlength requests. Increasing `max_length` requires preparing fresh data and more GPU memory.

Every source dataset has an explicit adapter and a pinned revision in [`configs/data.json`](../configs/data.json):

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

Continue with the [training guide](training.md) using the commit SHA printed by `push-data`.
