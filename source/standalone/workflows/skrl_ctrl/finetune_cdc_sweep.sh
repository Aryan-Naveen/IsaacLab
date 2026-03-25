#!/usr/bin/env bash
# Sweep CDC online finetuning: every seed × SAC, CTRLSAC-multi, CTRLSAC.
#
# Trajectory coefficients for the Refine task:
#   COEFF_JSON defaults to configs/coeffs/finetune_deg3.json under skrl_ctrl (override with env).
#
# Checkpoints: under runs/torch/<Trajectory task>/<agent layout>/seed_<seed>/ (same layout as
# train_seed_sac.sh / train_seed_ctrlsac.sh). The newest matching path is not used; we require
# .../checkpoints/${CKPT_NAME} somewhere under seed_<seed>.
#
# Env:
#   COEFF_JSON   — trajectory coeff JSON (default: configs/coeffs/finetune_deg3.json under skrl_ctrl).
#                  Finetune logs/checkpoints go under runs/finetune-online/CDC_finetuning/<stem>/
#                  where <stem> is the coeff file basename without .json. W&B run names include
#                  the same <stem> (e.g. SAC_finetune_finetune_deg3_seed42).
#   ENV_VERSION  — task suffix (default: legtrain)
#   SEEDS        — space-separated (default: 42 43 44 45 46)
#   CKPT_NAME    — checkpoint filename (default: agent_250000.pt)
#   ISAACLAB_SH  — launcher (default: <repo>/isaaclab.sh, i.e. ./isaaclab.sh from repo root)
#   FINETUNE_CDC_NO_HEADLESS — set to 1 to omit default --headless (local GUI / debugging)
#   FINETUNE_SMOKE_POLICY_PHASED — set to 0 to omit policy_phased_learning_rate for CTRLSAC-multi jobs only
#   POLICY_PHASED_LR — default 1e-6; both phases use this unless PHASE1/PHASE2 set (CTRLSAC-multi only)
#   POLICY_PHASED_LR_PHASE1 / POLICY_PHASED_LR_PHASE2 — optional per-phase overrides
#
# Trailing args pass through after --headless and --preset (e.g. --enable_cameras if you enable video in the preset).
#
# Example:
#   ./finetune_cdc_sweep.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-$REPO_ROOT/isaaclab.sh}"
HEADLESS_ARGS=(--headless)
if [[ "${FINETUNE_CDC_NO_HEADLESS:-}" == "1" ]]; then
  HEADLESS_ARGS=()
fi
ENV_VERSION="${ENV_VERSION:-legtrain}"
SEEDS="${SEEDS:-42 43 44 45 46}"
CKPT_NAME="${CKPT_NAME:-agent_250000.pt}"
POLICY_PHASED_LR_DEFAULT="${POLICY_PHASED_LR:-1e-6}"

BASE_PRESET="${SCRIPT_DIR}/configs/presets/finetune_online_cdc_finetuning.json"
TASK="Isaac-Quadcopter-${ENV_VERSION}-Trajectory-Direct-v0"

DEFAULT_COEFF_JSON="${SCRIPT_DIR}/configs/coeffs/finetune_deg3.json"
COEFF_JSON="${COEFF_JSON:-$DEFAULT_COEFF_JSON}"
if [[ ! -f "$COEFF_JSON" ]]; then
  echo "[finetune_cdc_sweep] COEFF_JSON not found: $COEFF_JSON" >&2
  exit 1
fi
# Output layout: .../CDC_finetuning/<coeff_basename_stem>/<agent>/seed_<seed>/
COEFF_OUT_TAG="$(basename "${COEFF_JSON%.json}")"
if [[ ! -f "$BASE_PRESET" ]]; then
  echo "[finetune_cdc_sweep] Missing base preset: $BASE_PRESET" >&2
  exit 1
fi
if [[ ! -x "$ISAACLAB_SH" && ! -f "$ISAACLAB_SH" ]]; then
  echo "[finetune_cdc_sweep] ISAACLAB_SH not found: $ISAACLAB_SH" >&2
  exit 1
fi

merge_preset_to() {
  local out_path="$1"
  local agent_type="$2"
  local checkpoint_dir="$3"
  local seed="$4"
  local wandb_name="$5"
  local output_root="$6"
  local merge_phased=()
  if [[ "$agent_type" == "CTRLSAC-multi" ]] && [[ "${FINETUNE_SMOKE_POLICY_PHASED:-1}" != "0" ]]; then
    merge_phased=(
      --policy-phased-learning-rate
      --policy-phased-lr-phase1 "${POLICY_PHASED_LR_PHASE1:-$POLICY_PHASED_LR_DEFAULT}"
      --policy-phased-lr-phase2 "${POLICY_PHASED_LR_PHASE2:-$POLICY_PHASED_LR_DEFAULT}"
    )
  fi
  "$ISAACLAB_SH" -p "$SCRIPT_DIR/merge_finetune_cdc_preset.py" \
    --base "$BASE_PRESET" \
    --out "$out_path" \
    --agent-type "$agent_type" \
    --checkpoint-dir "$checkpoint_dir" \
    --checkpoint-name "$CKPT_NAME" \
    --seed "$seed" \
    --wandb-name "$wandb_name" \
    --output-root "$output_root" \
    --coeff-json "$COEFF_JSON" \
    "${merge_phased[@]}"
}

torch_subdir_for_agent() {
  case "$1" in
    SAC) echo "SAC" ;;
    CTRLSAC) echo "CTRL-SAC-False" ;;
    CTRLSAC-multi) echo "CTRL-SAC-True" ;;
    *) echo "[finetune_cdc_sweep] unknown agent_type: $1" >&2; exit 1 ;;
  esac
}

wandb_slug_for_agent() {
  case "$1" in
    SAC) echo "SAC" ;;
    CTRLSAC) echo "CTRLSAC" ;;
    CTRLSAC-multi) echo "CTRLSAC_multi" ;;
    *) echo "[finetune_cdc_sweep] unknown agent_type: $1" >&2; exit 1 ;;
  esac
}

read -r -a SEED_ARR <<< "$SEEDS"
AGENTS=(SAC CTRLSAC-multi CTRLSAC)
total=$((${#SEED_ARR[@]} * ${#AGENTS[@]}))
echo "[finetune_cdc_sweep] ${total} jobs (${#SEED_ARR[@]} seeds × ${#AGENTS[@]} agents), ENV_VERSION=${ENV_VERSION}, COEFF_OUT_TAG=${COEFF_OUT_TAG}" >&2

done=0
for seed in "${SEED_ARR[@]}"; do
  for agent in "${AGENTS[@]}"; do
    ((++done)) || true
    sub="$(torch_subdir_for_agent "$agent")"
    seed_dir="${REPO_ROOT}/runs/torch/${TASK}/${sub}/seed_${seed}"
    found="$(find "$seed_dir" -type f -path "*/checkpoints/*" -name "$CKPT_NAME" 2>/dev/null | head -n1 || true)"
    if [[ -z "$found" ]]; then
      echo "[finetune_cdc_sweep] (${done}/${total}) No ${CKPT_NAME} under ${seed_dir}" >&2
      exit 1
    fi
    ckpt_dir="$(dirname "$found")"
    slug="$(wandb_slug_for_agent "$agent")"
    wandb_name="${slug}_finetune_${COEFF_OUT_TAG}_seed${seed}"
    out_root="${REPO_ROOT}/runs/finetune-online/CDC_finetuning/${COEFF_OUT_TAG}/${slug}/seed_${seed}"
    tmp="$(mktemp "${TMPDIR:-/tmp}/finetune_cdc_sweep_XXXXXX.json")"
    merge_preset_to "$tmp" "$agent" "$ckpt_dir" "$seed" "$wandb_name" "$out_root"
    echo "[finetune_cdc_sweep] (${done}/${total}) ${agent} seed=${seed} ckpt_dir=${ckpt_dir}" >&2
    "$ISAACLAB_SH" -p "$SCRIPT_DIR/finetune_online.py" "${HEADLESS_ARGS[@]}" --preset "$tmp" "$@"
    rm -f "$tmp"
    echo "    finished OK" >&2
  done
done

echo "[finetune_cdc_sweep] all ${total} runs finished." >&2
