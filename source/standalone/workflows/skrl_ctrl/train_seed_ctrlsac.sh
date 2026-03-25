#!/usr/bin/env bash
# Train SAC, CTRLSAC (single-task), and CTRLSAC-multi across the same five seeds.
# Sets RNG seed, W&B run name (title), and a distinct checkpoint directory per run.
#
# Usage:
#   ./train_seed_sweep.sh [extra args for every train invocation, e.g. --headless --enable_cameras]
#
# Env:
#   ENV_VERSION   — task suffix (default: legtrain)
#   SEEDS         — space-separated list (default: 42 43 44 45 46)
#   ISAACLAB_SH   — launcher (default: /workspace/isaaclab/isaaclab.sh for Docker)
#   WANDB_PROJECT — optional; passed via a temp experiment JSON merged into defaults

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
ENV_VERSION="${ENV_VERSION:-legtrain}"
SEEDS="${SEEDS:-42 43 44 45 46}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

TASK="Isaac-Quadcopter-${ENV_VERSION}-Trajectory-Direct-v0"
BASE_TORCH="runs/torch/${TASK}"

EXTRA=( "$@" )
TEMPS=()
cleanup() {
  local f
  for f in "${TEMPS[@]:-}"; do
    rm -f "$f"
  done
}
trap cleanup EXIT

exp_json() {
  # Optional wandb project override (otherwise default_experiment.json / factory default CDC)
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/train_seed_sweep_XXXXXX.json")"
  TEMPS+=("$tmp")
  if [[ -n "$WANDB_PROJECT" ]]; then
    python3 -c 'import json,sys; json.dump({"wandb_project": sys.argv[1]}, open(sys.argv[2], "w"))' \
      "$WANDB_PROJECT" "$tmp"
  else
    echo '{}' >"$tmp"
  fi
  echo "$tmp"
}

run_sac() {
  local seed="$1"
  local ej
  ej="$(exp_json)"
  echo "[train-seed-sweep] SAC seed=${seed}" >&2
  "$ISAACLAB_SH" -p "$SCRIPT_DIR/train_sac.py" \
    --env_version "$ENV_VERSION" \
    --seed "$seed" \
    --wandb_name "SAC_seed${seed}" \
    --experiment_directory "${BASE_TORCH}/SAC/seed_${seed}/" \
    --experiment_json "$ej" \
    "${EXTRA[@]}"
}

run_ctrlsac() {
  local seed="$1"
  local ej
  ej="$(exp_json)"
  echo "[train-seed-sweep] CTRLSAC seed=${seed}" >&2
  "$ISAACLAB_SH" -p "$SCRIPT_DIR/train.py" \
    --env_version "$ENV_VERSION" \
    --seed "$seed" \
    --wandb_name "CTRLSAC_seed${seed}" \
    --experiment_directory "${BASE_TORCH}/CTRL-SAC-False/seed_${seed}/" \
    --experiment_json "$ej" \
    "${EXTRA[@]}"
}

run_ctrlsac_multi() {
  local seed="$1"
  local ej
  ej="$(exp_json)"
  echo "[train-seed-sweep] CTRLSAC-multi seed=${seed}" >&2
  "$ISAACLAB_SH" -p "$SCRIPT_DIR/train.py" \
    --env_version "$ENV_VERSION" \
    --multitask \
    --seed "$seed" \
    --wandb_name "CTRLSAC-multi_seed${seed}" \
    --experiment_directory "${BASE_TORCH}/CTRL-SAC-True/seed_${seed}/" \
    --experiment_json "$ej" \
    "${EXTRA[@]}"
}

read -r -a SEED_ARR <<< "$SEEDS"
total=$((${#SEED_ARR[@]} * 3))
echo "[train-seed-sweep] ${total} jobs (${#SEED_ARR[@]} seeds × 3 agents), ENV_VERSION=${ENV_VERSION}" >&2

done=0
for seed in "${SEED_ARR[@]}"; do
  for runner in run_ctrlsac; do
    ((++done)) || true
    echo "[${done}/${total}]" >&2
    "$runner" "$seed"
    echo "    finished OK" >&2
  done
done

echo "[train-seed-sweep] all ${total} runs finished. Check runs/torch/${TASK}/ and W&B." >&2
