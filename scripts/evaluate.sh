#!/bin/bash

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <checkpoint_folder> <agent_type>"
    exit 1
fi

# Get the arguments
CHECKPOINT_FOLDER="$1"
AGENT_TYPE="$2"

# Maximum number of parallel processes
MAX_PARALLEL=5

# Array to store checkpoint files
CHECKPOINTS=($(find "$CHECKPOINT_FOLDER" -type f ! -name "best_agent.pt"))
TOTAL_CHECKPOINTS=${#CHECKPOINTS[@]}

# Shared counter for progress
PROGRESS_FILE="/tmp/progress_$$.txt"
echo 0 > "$PROGRESS_FILE"

# Function to run the command for a single checkpoint
run_command() {
    local CHECKPOINT="$1"
    local FILENAME=$(basename "$CHECKPOINT")

    # Run the isaaclab.sh command silently with dynamic agent type
    ./isaaclab.sh -p source/standalone/workflows/skrl_ctrl/eval.py \
        --agent_type "$AGENT_TYPE" \
        --experiment legeval-predef \
        --ckpt "$FILENAME" \
        --folder "$CHECKPOINT_FOLDER" \
        --headless > eval.log 2>&1

    # Update progress
    (
        flock -x 200
        PROGRESS=$(cat "$PROGRESS_FILE")
        PROGRESS=$((PROGRESS + 1))
        echo "$PROGRESS" > "$PROGRESS_FILE"
        PERCENT=$((PROGRESS * 100 / TOTAL_CHECKPOINTS))
        echo "Progress: $PERCENT% completed ($PROGRESS of $TOTAL_CHECKPOINTS)"
    ) 200>"$PROGRESS_FILE.lock"
}

# Export variables and function for parallel execution
export -f run_command
export CHECKPOINT_FOLDER
export AGENT_TYPE
export TOTAL_CHECKPOINTS
export PROGRESS_FILE

# Run commands in parallel
printf "%s\n" "${CHECKPOINTS[@]}" | \
    xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'run_command "$@"' _ {}

# Clean up
rm -f "$PROGRESS_FILE" "$PROGRESS_FILE.lock"
