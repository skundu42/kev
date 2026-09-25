# Calibration and evaluation

[Project overview](../README.md)

Run evaluation from the repository root on the GPU pod after training. Commands below assume a completed `runs/demo/final` export and prepared data in `data/mixture-v1`; adjust the paths for your setup. The `fit` workflow already runs calibration and evaluation, so these commands are also available as separate stages. Evaluation will not overwrite an existing report; choose a new output path to repeat it.

## Calibration and held-out metrics

Calibration fits one positive temperature on the separate calibration split and saves it with the final model. It adjusts the softmax distribution without changing the winning candidate. Calibration is approximate: hard labels and limited training sources do not make arbitrary yes/no answers trustworthy probabilities.

```bash
python -m kev.calibrate --model /workspace/kev-run/runs/demo/final --data /workspace/kev-run/data/mixture-v1/calibration.jsonl
python -m kev.evaluate --model /workspace/kev-run/runs/demo/final --data /workspace/kev-run/data/mixture-v1/test.jsonl --output /workspace/kev-run/runs/demo/evaluation.json
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

## Compare with Laya

The paired benchmark uses your prepared **test** partition and the English root checkpoint
`convaiinnovations/laya`, pinned to `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
It does not route to multilingual or typed-decisions variants. Run after training finishes,
inside tmux on an idle GPU. Install only the pinned Laya runtime into a separate environment
that reuses the Halo container's dependencies; do not replace Torch/CUDA:

```bash
cd /workspace/kev
git pull --ff-only
python -m venv --system-site-packages /workspace/kev-run/venv-compare
source /workspace/kev-run/venv-compare/bin/activate
python -m pip install --no-deps 'git+https://github.com/NandhaKishorM/laya.git@970dc8c5f63d7b886a68409493f37d569424f933'
export HF_HOME=/workspace/kev-run/hf-cache
export USE_TF=0
export PYTHONUNBUFFERED=1
python -m kev.compare \
  --kev-model /workspace/kev-run/runs/demo/final \
  --data /workspace/kev-run/data/mixture-v1/test.jsonl \
  --per-source 100 \
  --output-dir /workspace/kev-run/comparisons/laya-sample
```

The default sample selects up to 100 decisions per source deterministically. For the whole
test set, use `--per-source 0` and a new output directory. Both models process identical rows
sequentially on the same GPU. `comparison.json` includes per-source/per-kind accuracy, log loss,
Brier score, ECE, ordinal MAE, synchronized single-request latency and peak allocated VRAM.
`sample.jsonl` and per-model prediction files preserve the exact paired inputs and outputs.
Existing output directories are protected; use a fresh directory after an interrupted run.

This measures native deployed behavior: each model uses its shipped calibration, formatting,
precision and limits. Kev has in-domain calibration; Laya's calibration and training overlap
are not controlled. Laya's default English context/head budgets can discard state, instructions
or options. The report counts affected rows and separately evaluates `shared_untruncated`
for **both** models on the same unaffected subset. Kev rejects overlength input. Laya's public
API rounds probabilities to four decimals; these are renormalized and zero probabilities use
the existing evaluation log-loss floor of 1e-300. Thus log loss can be sensitive to rounding.
The runtime's CPU fallback aborts the benchmark, rather than silently corrupting GPU timings.
No model is recalibrated or trained on the test set. This is a comparison on Kev's dataset
mixture, not proof of superiority on unseen task families. Local validation uses handwritten
fixtures with network access blocked; validate GPU execution on your target pod.
