#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND="${1:-help}"
if (($#)); then shift; fi

usage() {
  cat <<'EOF'
Usage: bash scripts/runpod.sh COMMAND [arguments]
  setup-data                    Check CPU preparation dependencies (no installation)
  prepare                       Prepare data on a CPU machine; no CUDA or Halo needed
  push-data REPO_ID              Upload complete prepared data to a private HF dataset repo
  pull-data REPO_ID REVISION     Download prepared data at an exact Hub commit SHA
  setup                         Validate the pod image and pin Halo (no training)
  check-run                     Check real-model training/resume/export on prepared data
  fit                           Train, calibrate, evaluate, and predict from existing data
  train [trainer arguments]     Train through Halo (pass --resume-from-checkpoint PATH to resume)
  resume CHECKPOINT             Resume the selected run without replacing it
  calibrate                     Fit temperature on the separate calibration split
  evaluate                      Write held-out test metrics to RUN_DIR/evaluation.json
  predict [INPUT.json|-]         Read a request (defaults to examples/request.json)
  serve [--host HOST --port N]    Serve RUN_DIR/final via authenticated HTTP (KEV_API_KEY)
  smoke                         Setup and run every stage with the three-step smoke profile
  all                           Setup and run every stage with the training profile

Environment: KEV_WORKDIR=/workspace/kev-run; KEV_RUN_NAME=demo (smoke: smoke).
KEV_DATA_DIR overrides the prepared data path independently of the run name.
KEV_CONFIG may select a different config file; HF_HOME and HF_DATASETS_CACHE are respected.
Existing data/runs are not overwritten. Prepare on a remote CPU machine; train on a GPU pod.
EOF
}

case "$COMMAND" in
  help|-h|--help) usage; exit 0 ;;
  setup-data|prepare|push-data|pull-data|setup|check-run|fit|train|resume|calibrate|evaluate|predict|serve|smoke|all) ;;
  *) usage >&2; exit 2 ;;
esac

WORKDIR="${KEV_WORKDIR:-/workspace/kev-run}"
RUN_NAME="${KEV_RUN_NAME:-demo}"
CONFIG="${KEV_CONFIG:-$REPO_ROOT/configs/train.yaml}"
if [[ "$COMMAND" == smoke ]]; then
  RUN_NAME="${KEV_RUN_NAME:-smoke}"
  CONFIG="${KEV_CONFIG:-$REPO_ROOT/configs/smoke.yaml}"
fi
[[ "$WORKDIR" == /* ]] || { echo 'KEV_WORKDIR must be absolute.' >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'KEV_RUN_NAME must be a simple directory name.' >&2; exit 2; }
[[ "$CONFIG" == /* ]] || CONFIG="$REPO_ROOT/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

HALO_REVISION=ffc9d46290b62e61150568f3b66b0b1f900b2598
HALO_ROOT="$WORKDIR/halo"
DATA_DIR="${KEV_DATA_DIR:-$WORKDIR/data/$RUN_NAME}"
[[ "$DATA_DIR" == /* ]] || { echo 'KEV_DATA_DIR must be absolute.' >&2; exit 2; }
RUN_DIR="$WORKDIR/runs/$RUN_NAME"
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export KEV_WORKDIR="$WORKDIR"
export PYTHONPATH="$HALO_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"

runtime() {
  python "$REPO_ROOT/scripts/preflight.py" --runtime-only >&2
}

setup_data() {
  python "$REPO_ROOT/scripts/preflight_data.py" >&2
}

verify_halo() {
  [[ -d "$HALO_ROOT/.git" ]] || { echo 'Run setup first: the pinned Halo checkout is missing.' >&2; exit 1; }
  [[ "$(git -C "$HALO_ROOT" rev-parse HEAD)" == "$HALO_REVISION" ]] || { echo 'Halo revision differs from the required pin; use a new KEV_WORKDIR.' >&2; exit 1; }
  git -C "$HALO_ROOT" diff --quiet --ignore-submodules -- || { echo 'Halo has tracked modifications; use a clean checkout.' >&2; exit 1; }
  git -C "$HALO_ROOT" diff --cached --quiet --ignore-submodules -- || { echo 'Halo has staged modifications; use a clean checkout.' >&2; exit 1; }
}

setup() {
  runtime
  command -v git >/dev/null || { echo 'git is missing from the pod image.' >&2; exit 1; }
  if [[ ! -e "$HALO_ROOT" ]]; then
    mkdir -p "$WORKDIR"
    git clone --filter=blob:none --no-checkout https://github.com/whitecircle/halo.git "$HALO_ROOT"
    git -C "$HALO_ROOT" checkout --detach "$HALO_REVISION"
  fi
  verify_halo
  python "$REPO_ROOT/scripts/preflight.py" >&2
}

prepare() {
  setup_data
  [[ ! -e "$DATA_DIR" ]] || { echo "Data already exists: $DATA_DIR. Choose a new KEV_DATA_DIR or KEV_RUN_NAME." >&2; exit 1; }
  python -m kev.prepare --config "$CONFIG" --output-dir "$DATA_DIR"
}

train() {
  runtime
  verify_halo
  python "$REPO_ROOT/scripts/preflight.py" >&2
  python -m kev.hub validate --data-dir "$DATA_DIR" --config "$CONFIG"
  local resuming=false
  local argument
  for argument in "$@"; do
    if [[ "$argument" == --resume-from-checkpoint || "$argument" == --resume-from-checkpoint=* ]]; then resuming=true; fi
  done
  if [[ -e "$RUN_DIR" && "$resuming" == false ]]; then
    echo "Run already exists: $RUN_DIR. Use resume or choose a new KEV_RUN_NAME." >&2
    exit 1
  fi
  python -m src.cli launch kev "$CONFIG" --root "$REPO_ROOT" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" "$@"
}

calibrate() {
  runtime
  python -m kev.calibrate --model "$RUN_DIR/final" --data "$DATA_DIR/calibration.jsonl"
}

evaluate() {
  runtime
  [[ ! -e "$RUN_DIR/evaluation.json" ]] || { echo 'evaluation.json already exists; preserve or move it before another evaluation.' >&2; exit 1; }
  python -m kev.evaluate --model "$RUN_DIR/final" --data "$DATA_DIR/test.jsonl" --output "$RUN_DIR/evaluation.json"
}

predict() {
  runtime
  python -m kev.inference --model "$RUN_DIR/final" --input "${1:-$REPO_ROOT/examples/request.json}"
}

case "$COMMAND" in
  setup-data) setup_data ;;
  push-data)
    [[ $# -eq 1 ]] || { echo 'Usage: push-data OWNER/DATASET' >&2; exit 2; }
    python -m kev.hub push --data-dir "$DATA_DIR" --repo-id "$1"
    ;;
  pull-data)
    [[ $# -eq 2 ]] || { echo 'Usage: pull-data OWNER/DATASET COMMIT_SHA' >&2; exit 2; }
    python -m kev.hub pull --data-dir "$DATA_DIR" --repo-id "$1" --revision "$2"
    ;;
  setup) setup ;;
  check-run)
    setup
    python "$REPO_ROOT/scripts/check_run.py" --config "$CONFIG" --data-dir "$DATA_DIR" --output-dir "$WORKDIR/checks/$RUN_NAME"
    ;;
  prepare) prepare ;;
  train) train "$@" ;;
  resume)
    [[ $# -ge 1 && -f "$1/trainer_state.json" ]] || { echo 'resume requires a checkpoint directory containing trainer_state.json.' >&2; exit 2; }
    train --resume-from-checkpoint "$1" "${@:2}"
    ;;
  calibrate) calibrate ;;
  evaluate) evaluate ;;
  predict) predict "$@" ;;
  serve) python -m kev.serve --model "$RUN_DIR/final" "$@" ;;
  fit) setup; train; calibrate; evaluate; predict ;;
  smoke) setup; python "$REPO_ROOT/scripts/check_gpu.py"; prepare; train; calibrate; evaluate; predict ;;
  all) setup; prepare; train; calibrate; evaluate; predict ;;
esac
