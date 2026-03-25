#!/usr/bin/env bash

set -e  # exit on error

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for i in {0..10}; do
  echo "Starting runs for base${i}.json"

  make start_eval_one EVAL_PRESET=eval_CTRL_multi_zero_shot.json \
    ARGS="--coeff_json /workspace/isaaclab/source/standalone/workflows/skrl_ctrl/configs/coeffs/sweep/base${i}.json" \
    > "$LOG_DIR/base${i}_ctrl_multi_zero_shot.log" 2>&1
  echo "Finished CTRL multi zero shot for base${i}"

  make start_eval_one EVAL_PRESET=eval_SAC.json \
    ARGS="--coeff_json /workspace/isaaclab/source/standalone/workflows/skrl_ctrl/configs/coeffs/sweep/base${i}.json" \
    > "$LOG_DIR/base${i}_sac.log" 2>&1
  echo "Finished SAC for base${i}"

  # make start_eval_one EVAL_PRESET=eval_CTRL.json \
  #   ARGS="--coeff_json /workspace/isaaclab/source/standalone/workflows/skrl_ctrl/configs/coeffs/sweep/base${i}.json" \
  #   > "$LOG_DIR/base${i}_ctrl.log" 2>&1
  # echo "Finished CTRL for base${i}"

done

echo "All runs complete."