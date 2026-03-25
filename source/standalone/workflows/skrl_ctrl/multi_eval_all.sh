#!/usr/bin/env bash
# Multi-section batch eval: source (3) + in-domain (100 sequential) + OOD (3) × checkpoints × policies.
#
# Manifest: one line per policy, comma-separated: agent_type,checkpoint_dir
#   agent_type: SAC | CTRLSAC | CTRLSAC-multi (case-insensitive; CTRL-SAC and CTRLSAC-MULTI accepted)
#   checkpoint_dir: directory containing best_agent.pt and agent_250000.pt
#
# In-domain coefficients: default OUT_DIR/coeffs/in_domain_100.json. If missing, generated via
#   $ISAACLAB_SH -p generate_multi_eval_coeffs.py --out-dir ... --master-seed $MASTER_SEED
# Override path with IN_DOMAIN_COEFF_JSON=/path/to/in_domain_100.json if needed.
#
# Usage (inside Docker, from repo root or set ISAACLAB_ROOT):
#   cd /workspace/isaaclab
#   bash source/standalone/workflows/skrl_ctrl/multi_eval_all.sh MANIFEST.csv OUT_DIR [-- eval-batch args...]
#
# All Python (eval batches + CSV export) runs via: ISAACLAB_SH -p <script.py> ... (default: ISAACLAB_ROOT/isaaclab.sh)
#
# Env:
#   ISAACLAB_ROOT        default /workspace/isaaclab (must contain ./isaaclab.sh)
#   ISAACLAB_SH          default $ISAACLAB_ROOT/isaaclab.sh
#   MASTER_SEED          default 42 (eval-batch seed)
#   CKPTS                space-separated (default: best_agent.pt agent_250000.pt)
#   IN_DOMAIN_COEFF_JSON default $OUT_DIR/coeffs/in_domain_100.json (auto-generated if missing)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
ISAACLAB_SH="${ISAACLAB_SH:-$ISAACLAB_ROOT/isaaclab.sh}"
EVAL_BATCH_PY="${EVAL_BATCH_PY:-$SCRIPT_DIR/eval_batch.py}"
EXPORT_PY="${EXPORT_PY:-$SCRIPT_DIR/export_batch_bundle_csv.py}"
GEN_COEFF_PY="${GEN_COEFF_PY:-$SCRIPT_DIR/generate_multi_eval_coeffs.py}"
MASTER_SEED="${MASTER_SEED:-42}"
CKPTS="${CKPTS:-best_agent.pt agent_250000.pt}"

PRESET_SRC="$SCRIPT_DIR/configs/presets/eval_batch_multi_source3.json"
PRESET_IN="$SCRIPT_DIR/configs/presets/eval_batch_multi_indomain100.json"
PRESET_OOD="$SCRIPT_DIR/configs/presets/eval_batch_multi_ood3.json"

usage() {
  echo "Usage: $0 <manifest.csv> <out_dir> [-- extra args for eval-batch, e.g. --headless --enable_cameras]" >&2
  echo "Manifest lines: agent_type,checkpoint_dir (no commas in path)." >&2
  echo "IN_DOMAIN_COEFF_JSON defaults to OUT_DIR/coeffs/in_domain_100.json (auto-generated if missing)." >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

MANIFEST="$1"
OUT_DIR="$2"
shift 2

IN_DOMAIN_COEFF_JSON="${IN_DOMAIN_COEFF_JSON:-"$OUT_DIR/coeffs/in_domain_100.json"}"

EXTRA=( "$@" )
if [[ ${#EXTRA[@]} -eq 0 ]]; then
  EXTRA=( --headless --enable_cameras )
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

if [[ ! -f "$ISAACLAB_SH" ]]; then
  echo "isaaclab.sh not found: $ISAACLAB_SH" >&2
  exit 1
fi

if [[ ! -f "$IN_DOMAIN_COEFF_JSON" ]]; then
  coeff_dir="$(dirname "$IN_DOMAIN_COEFF_JSON")"
  echo "[multi-eval] Generating in-domain coeffs -> $IN_DOMAIN_COEFF_JSON" >&2
  mkdir -p "$coeff_dir"
  ( cd "$ISAACLAB_ROOT" && \
    "$ISAACLAB_SH" -p "$GEN_COEFF_PY" \
      --out-dir "$coeff_dir" \
      --master-seed "$MASTER_SEED" \
  )
fi

normalize_agent_type() {
  local raw a
  raw="$1"
  a="$(echo "$raw" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  case "$a" in
    SAC) echo SAC ;;
    CTRLSAC | CTRL-SAC) echo CTRLSAC ;;
    CTRLSAC-MULTI | CTRLSACMULTI) echo CTRLSAC-multi ;;
    *)
      echo "Unknown agent_type: $raw (use SAC, CTRLSAC, or CTRLSAC-multi)" >&2
      exit 1
      ;;
  esac
}

sanitize_label() {
  echo "$1" | tr '/ ' '__' | tr -cd '[:alnum:]_.-'
}

mkdir -p "$OUT_DIR/bundles" "$OUT_DIR/csv"

declare -a DETAIL_CSVS=()
declare -a SUMMARY_CSVS=()

run_section() {
  local section="$1"
  local preset="$2"
  local folder="$3"
  local ck="$4"
  local agent="$5"
  local label="$6"
  local coeff_json="${7:-}"

  local bundle="$OUT_DIR/bundles/${label}_$(sanitize_label "$ck")_${section}.pt"
  local detail="$OUT_DIR/csv/detail_${label}_$(sanitize_label "$ck")_${section}.csv"
  local summary="$OUT_DIR/csv/summary_${label}_$(sanitize_label "$ck")_${section}.csv"

  echo "[multi-eval] section=$section agent=$agent ckpt=$ck -> $bundle"

  local -a coeff_args=()
  if [[ -n "$coeff_json" ]]; then
    coeff_args=( --coeff_json "$coeff_json" )
  fi

  ( cd "$ISAACLAB_ROOT" && \
    "$ISAACLAB_SH" -p "$EVAL_BATCH_PY" \
      "${EXTRA[@]}" \
      --preset "$preset" \
      --folder "$folder" \
      --ckpt "$ck" \
      --out "$bundle" \
      --agent_type "$agent" \
      --seed "$MASTER_SEED" \
      "${coeff_args[@]}" \
  )

  ( cd "$ISAACLAB_ROOT" && \
    "$ISAACLAB_SH" -p "$EXPORT_PY" \
      "$bundle" \
      --out "$detail" \
      --summary-out "$summary" \
      --section "$section" \
      --agent-type "$agent" \
      --folder "$folder" \
      --ckpt "$ck" \
  )

  DETAIL_CSVS+=("$detail")
  SUMMARY_CSVS+=("$summary")
}

line_no=0
while IFS= read -r line || [[ -n "${line:-}" ]]; do
  line_no=$((line_no + 1))
  line="${line//$'\r'/}"
  [[ -z "${line//[[:space:]]/}" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue

  if ! IFS=',' read -r agent_raw folder_raw <<< "$line"; then
    echo "Bad line $line_no in manifest" >&2
    exit 1
  fi
  agent_raw="$(echo "$agent_raw" | xargs)"
  folder_raw="$(echo "$folder_raw" | xargs)"
  if [[ -z "$agent_raw" || -z "$folder_raw" ]]; then
    echo "Empty agent_type or folder at line $line_no" >&2
    exit 1
  fi

  agent="$(normalize_agent_type "$agent_raw")"
  folder="$folder_raw"

  if [[ ! -d "$folder" ]]; then
    echo "Not a directory (line $line_no): $folder" >&2
    exit 1
  fi

  read -r -a ckpt_arr <<< "$CKPTS"
  for ck in "${ckpt_arr[@]}"; do
    if [[ ! -f "$folder/$ck" ]]; then
      echo "Missing checkpoint $folder/$ck (line $line_no)" >&2
      exit 1
    fi
  done

  label="$(sanitize_label "$agent")_$(sanitize_label "$folder")"

  for ck in "${ckpt_arr[@]}"; do
    run_section "source" "$PRESET_SRC" "$folder" "$ck" "$agent" "$label" ""
    run_section "indomain" "$PRESET_IN" "$folder" "$ck" "$agent" "$label" "$IN_DOMAIN_COEFF_JSON"
    run_section "ood" "$PRESET_OOD" "$folder" "$ck" "$agent" "$label" ""
  done
done < "$MANIFEST"

if [[ ${#DETAIL_CSVS[@]} -gt 0 ]]; then
  MERGE="$OUT_DIR/all_rollouts.csv"
  head -1 "${DETAIL_CSVS[0]}" > "$MERGE"
  for f in "${DETAIL_CSVS[@]}"; do
    tail -n +2 "$f" >> "$MERGE"
  done
  echo "[multi-eval] Merged rollouts -> $MERGE"
fi

if [[ ${#SUMMARY_CSVS[@]} -gt 0 ]]; then
  SMERGE="$OUT_DIR/all_summaries.csv"
  head -1 "${SUMMARY_CSVS[0]}" > "$SMERGE"
  for f in "${SUMMARY_CSVS[@]}"; do
    tail -n +2 "$f" >> "$SMERGE"
  done
  echo "[multi-eval] Merged summaries -> $SMERGE"
fi

echo "[multi-eval] Done."
