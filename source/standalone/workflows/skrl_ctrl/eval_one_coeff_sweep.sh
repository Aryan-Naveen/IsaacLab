#!/usr/bin/env bash
# Sequential eval-one over every *.json in a directory (CTRLSAC-multi, CTRLSAC, SAC per file).
# Agent-first: all coeffs for multi, then all for single CTRLSAC, then all for SAC.
#
# Usage:
#   ./eval_one_coeff_sweep.sh <coeff_json_dir> [args passed to every eval-one...]
#
# Env:
#   EACH_ROW=1  — split multi-row JSON (matrix or one flat row) using jq; requires jq.
#   ISAACLAB_SH, CLI_PY — overrides (defaults suit /workspace/isaaclab in Docker).
#   PRESET_MULTI, PRESET_CTRLSAC, PRESET_SAC — preset paths.
#
# Coeff directory should contain only trajectory coefficient JSONs. Files matching *manifest* are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
CLI_PY="${CLI_PY:-$SCRIPT_DIR/cli.py}"
PRESET_MULTI="${PRESET_MULTI:-$SCRIPT_DIR/configs/presets/eval_CTRL_multi_zero_shot.json}"
PRESET_CTRLSAC="${PRESET_CTRLSAC:-$SCRIPT_DIR/configs/presets/eval_CTRL.json}"
PRESET_SAC="${PRESET_SAC:-$SCRIPT_DIR/configs/presets/eval_SAC.json}"
EACH_ROW="${EACH_ROW:-0}"

usage() {
  echo "Usage: $0 <coeff_json_dir> [extra args for each eval-one, e.g. --headless --enable_cameras]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COEFF_DIR="$1"
shift
EXTRA=( "$@" )

if [[ ! -d "$COEFF_DIR" ]]; then
  echo "Not a directory: $COEFF_DIR" >&2
  exit 1
fi

if [[ "$EACH_ROW" == "1" ]] && ! command -v jq &>/dev/null; then
  echo "EACH_ROW=1 requires jq in PATH." >&2
  exit 1
fi

shopt -s nullglob
mapfile -t found < <(find "$COEFF_DIR" -maxdepth 1 -type f -name '*.json' ! -name '*manifest*' | LC_ALL=C sort)
shopt -u nullglob

if [[ ${#found[@]} -eq 0 ]]; then
  echo "No coefficient *.json files in $COEFF_DIR (manifest names are skipped)." >&2
  exit 1
fi

declare -a COEFF_PATH=()
declare -a COEFF_LABEL=()
declare -a TEMPS=()

cleanup() {
  local f
  for f in "${TEMPS[@]:-}"; do
    rm -f "$f"
  done
}
trap cleanup EXIT

add_jobs_for_file() {
  local f="$1"
  if [[ "$EACH_ROW" == "1" ]]; then
    local t0 n i tmp
    t0="$(jq -r '.[0] | type' "$f")"
    if [[ "$t0" == "array" ]]; then
      n="$(jq 'length' "$f")"
      for ((i = 0; i < n; i++)); do
        tmp="$(mktemp /tmp/coeff_sweep_XXXXXX.json)"
        jq -c "[.[$i]]" "$f" >"$tmp"
        TEMPS+=("$tmp")
        COEFF_PATH+=("$tmp")
        COEFF_LABEL+=("$(basename "$f")#row${i}")
      done
      return
    fi
  fi
  COEFF_PATH+=("$f")
  COEFF_LABEL+=("$(basename "$f")")
}

for f in "${found[@]}"; do
  add_jobs_for_file "$f"
done

n_coeff=${#COEFF_PATH[@]}
total=$((n_coeff * 3))
echo "[coeff-sweep] ${total} eval-one jobs (${n_coeff} coefficient jobs × 3 agents), agent-first, EACH_ROW=${EACH_ROW}" >&2

done=0
declare -a PRESETS=("$PRESET_MULTI" "$PRESET_CTRLSAC" "$PRESET_SAC")
declare -a AGENTS=("CTRLSAC-multi" "CTRLSAC" "SAC")

for pi in 0 1 2; do
  preset="${PRESETS[$pi]}"
  agent="${AGENTS[$pi]}"
  if [[ ! -f "$preset" ]]; then
    echo "Preset not found: $preset" >&2
    exit 1
  fi
  for ci in "${!COEFF_PATH[@]}"; do
    ((++done)) || true
    coeff="${COEFF_PATH[$ci]}"
    label="${COEFF_LABEL[$ci]}"
    echo "[${done}/${total}] agent=${agent} coeff=${label}" >&2
    "$ISAACLAB_SH" -p "$CLI_PY" eval-one --preset "$preset" --coeff_json "$coeff" "${EXTRA[@]}"
    echo "    finished OK" >&2
  done
done

echo "[coeff-sweep] all ${total} runs finished. See preset output_root (default runs/eval) for metrics.json and video/." >&2
