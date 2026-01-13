#!/usr/bin/env bash
# Helper to kill debug / training python processes inside the isaac-lab-base container.
# Usage: ./scripts/debug_kill.sh

set -euo pipefail

# Resolve repo root (one level up from scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yaml"

echo "Killing training/debug python processes in container 'isaac-lab-base'..."

if [ ! -f "${COMPOSE_FILE}" ]; then
	echo "Error: docker compose file not found at ${COMPOSE_FILE}" >&2
	exit 1
fi

if ! docker compose -f "${COMPOSE_FILE}" ps --services --filter "status=running" | grep -q "isaac-lab-base"; then
	echo "Container 'isaac-lab-base' is not running. Nothing to kill." >&2
	exit 0
fi

# Try to kill the specific training script first, fallback to python processes matching debugpy
docker compose -f "${COMPOSE_FILE}" exec isaac-lab-base bash -lc "pkill -f 'train_sac.py' || true"
docker compose -f "${COMPOSE_FILE}" exec isaac-lab-base bash -lc "pkill -f 'debugpy' || true"

echo "Done."
