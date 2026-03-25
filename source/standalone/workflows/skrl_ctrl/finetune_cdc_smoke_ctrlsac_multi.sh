#!/usr/bin/env bash
# Single CTRLSAC-multi online finetune for a quick W&B curve check (CDC_finetuning project).
#
# Defaults match finetune_online_cdc_finetuning.json + multitask phased policy LR: phase1 and phase2
# use the same LR (default 1e-6 via POLICY_PHASED_LR). Override per-phase only if needed.
#
# Env:
#   COEFF_JSON   — trajectory coeff JSON (default: configs/coeffs/finetune_deg3.json under skrl_ctrl)
#   ENV_VERSION  — default legtrain
#   SEED         — default 42
#   CKPT_NAME    — default agent_250000.pt
#   ISAACLAB_SH  — default <repo>/isaaclab.sh (./isaaclab.sh from repo root)
#   WANDB_NAME   — default CDC_smoke_ctrlsac_multi_seed<SEED>
#   FINETUNE_CDC_NO_HEADLESS — set to 1 to omit default --headless (local GUI / debugging)
#   FINETUNE_SMOKE_POLICY_PHASED — set to 0 to disable policy_phased_learning_rate in merged preset
#   POLICY_PHASED_LR — default 1e-6; used for both phase1 and phase2 when phases not set separately
#   POLICY_PHASED_LR_PHASE1 / POLICY_PHASED_LR_PHASE2 — optional overrides (default: POLICY_PHASED_LR)
#
# Trailing args pass through to the Isaac Python entrypoint after --headless and --preset.
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
SEED="${SEED:-42}"
CKPT_NAME="${CKPT_NAME:-agent_250000.pt}"

BASE_PRESET="${SCRIPT_DIR}/configs/presets/finetune_online_cdc_finetuning.json"
TASK="Isaac-Quadcopter-${ENV_VERSION}-Trajectory-Direct-v0"
AGENT="CTRLSAC-multi"
SUB="CTRL-SAC-True"

POLICY_PHASED_LR_DEFAULT="${POLICY_PHASED_LR:-1e-6}"

DEFAULT_COEFF_JSON="${SCRIPT_DIR}/configs/coeffs/finetune_deg3.json"
COEFF_JSON="${COEFF_JSON:-$DEFAULT_COEFF_JSON}"
if [[ ! -f "$COEFF_JSON" ]]; then
  echo "[finetune_cdc_smoke] coeff JSON not found: $COEFF_JSON" >&2
  exit 1
fi

seed_dir="${REPO_ROOT}/runs/torch/${TASK}/${SUB}/seed_${SEED}"
found="$(find "$seed_dir" -type f -path "*/checkpoints/*" -name "$CKPT_NAME" 2>/dev/null | head -n1 || true)"
if [[ -z "$found" ]]; then
  echo "[finetune_cdc_smoke] No ${CKPT_NAME} under ${seed_dir}" >&2
  exit 1
fi
ckpt_dir="$(dirname "$found")"
WANDB_NAME="${WANDB_NAME:-CDC_smoke_ctrlsac_multi_seed${SEED}}"
out_root="${REPO_ROOT}/runs/finetune-online/CDC_finetuning/smoke_CTRLSAC_multi/seed_${SEED}"

tmp="$(mktemp "${TMPDIR:-/tmp}/finetune_cdc_smoke_XXXXXX.json")"
MERGE_PHASED=()
if [[ "${FINETUNE_SMOKE_POLICY_PHASED:-1}" != "0" ]]; then
  MERGE_PHASED=(
    --policy-phased-learning-rate
    --policy-phased-lr-phase1 "${POLICY_PHASED_LR_PHASE1:-$POLICY_PHASED_LR_DEFAULT}"
    --policy-phased-lr-phase2 "${POLICY_PHASED_LR_PHASE2:-$POLICY_PHASED_LR_DEFAULT}"
  )
fi
"$ISAACLAB_SH" -p "$SCRIPT_DIR/merge_finetune_cdc_preset.py" \
  --base "$BASE_PRESET" \
  --out "$tmp" \
  --agent-type "$AGENT" \
  --checkpoint-dir "$ckpt_dir" \
  --checkpoint-name "$CKPT_NAME" \
  --seed "$SEED" \
  --wandb-name "$WANDB_NAME" \
  --output-root "$out_root" \
  --coeff-json "$COEFF_JSON" \
  "${MERGE_PHASED[@]}"

echo "[finetune_cdc_smoke] ${AGENT} seed=${SEED} wandb_name=${WANDB_NAME} ckpt_dir=${ckpt_dir}" >&2
"$ISAACLAB_SH" -p "$SCRIPT_DIR/finetune_online.py" "${HEADLESS_ARGS[@]}" --preset "$tmp" "$@"
rm -f "$tmp"
echo "[finetune_cdc_smoke] done." >&2
