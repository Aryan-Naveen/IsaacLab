#!/usr/bin/env bash
# One-shot multi-eval for Docker: exports paths, then runs multi_eval_all.sh.
#
#   chmod +x run_multi_eval.sh
#   ./run_multi_eval.sh
#   ./run_multi_eval.sh -- --headless --enable_cameras
#
# Override before running (optional):
#   export ISAACLAB_ROOT=/other/path
#   export MULTI_EVAL_MANIFEST=/path/to/manifest.csv
#   export MULTI_EVAL_OUT_DIR=/path/to/output
#   export MASTER_SEED=42
#   export CKPTS="best_agent.pt agent_250000.pt"
#
set -euo pipefail

export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
export ISAACLAB_SH="${ISAACLAB_SH:-$ISAACLAB_ROOT/isaaclab.sh}"
export MASTER_SEED="${MASTER_SEED:-42}"
export CKPTS="${CKPTS:-best_agent.pt agent_250000.pt}"

SKRL_CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MULTI_EVAL_MANIFEST="${MULTI_EVAL_MANIFEST:-$SKRL_CTRL/multi_eval_manifest_legtrain_seeds.csv}"
export MULTI_EVAL_OUT_DIR="${MULTI_EVAL_OUT_DIR:-$ISAACLAB_ROOT/runs/eval/multi_eval_legtrain}"

if [[ ! -d "$ISAACLAB_ROOT" ]]; then
  echo "[run_multi_eval] ISAACLAB_ROOT is not a directory: $ISAACLAB_ROOT" >&2
  echo "[run_multi_eval] Set ISAACLAB_ROOT to your Isaac Lab repo root." >&2
  exit 1
fi
if [[ ! -f "$ISAACLAB_SH" ]]; then
  echo "[run_multi_eval] Missing: $ISAACLAB_SH" >&2
  exit 1
fi
if [[ ! -f "$MULTI_EVAL_MANIFEST" ]]; then
  echo "[run_multi_eval] Manifest not found: $MULTI_EVAL_MANIFEST" >&2
  exit 1
fi

mkdir -p "$MULTI_EVAL_OUT_DIR"/{bundles,csv,coeffs}

echo "[run_multi_eval] ISAACLAB_ROOT=$ISAACLAB_ROOT"
echo "[run_multi_eval] MANIFEST=$MULTI_EVAL_MANIFEST"
echo "[run_multi_eval] OUT_DIR=$MULTI_EVAL_OUT_DIR"
echo "[run_multi_eval] MASTER_SEED=$MASTER_SEED CKPTS=$CKPTS"
echo

cd "$ISAACLAB_ROOT"
exec bash "$SKRL_CTRL/multi_eval_all.sh" "$MULTI_EVAL_MANIFEST" "$MULTI_EVAL_OUT_DIR" "$@"
