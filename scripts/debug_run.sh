#!/usr/bin/env bash
# Quick helper to start the training script inside the isaac-lab-base container with debugpy listening.
# Usage: ./scripts/debug_run.sh [port] [--no-wait]
# Default port: 5678. By default the script waits for the debugger to attach.

set -euo pipefail

# Resolve repo root (one level up from scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yaml"

PORT=${1:-5678}
WAIT_FLAG="--wait-for-client"
if [ "${2-}" = "--no-wait" ]; then
  WAIT_FLAG=""
fi

echo "Starting debug run (port=${PORT}) in container 'isaac-lab-base'..."

# Verify compose file exists
if [ ! -f "${COMPOSE_FILE}" ]; then
  echo "Error: docker compose file not found at ${COMPOSE_FILE}" >&2
  exit 1
fi

# Check the container is running
if ! docker compose -f "${COMPOSE_FILE}" ps --services --filter "status=running" | grep -q "isaac-lab-base"; then
  echo "Container 'isaac-lab-base' is not running. Start it with:" >&2
  echo "  docker compose -f ${COMPOSE_FILE} up -d isaac-lab-base" >&2
  exit 1
fi

docker compose -f "${COMPOSE_FILE}" exec isaac-lab-base bash -lc \
  "/workspace/isaaclab/isaaclab.sh -p -m debugpy --listen 0.0.0.0:${PORT} ${WAIT_FLAG} source/standalone/workflows/skrl_ctrl/train_sac.py --headless --enable_cameras"

echo "Debug run launched. Attach your debugger to localhost:${PORT}."
