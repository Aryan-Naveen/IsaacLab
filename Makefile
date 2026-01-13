.PHONY: build up start debug-run debug-kill stop rebuild logs shell restart

# Path to docker compose file in the repo
COMPOSE_FILE := docker/docker-compose.yaml
SERVICE := isaac-lab-base
TRAIN_SCRIPT := source/standalone/workflows/skrl_ctrl/train_sac.py
DEBUG_PORT ?= 5678

# Build the base image
build:
	@echo "Building ${SERVICE} image..."
	docker compose -f ${COMPOSE_FILE} build ${SERVICE}

# Start the container (build first)
up: build
	@echo "Starting ${SERVICE} container..."
	docker compose -f ${COMPOSE_FILE} up -d ${SERVICE}

# Start the training script (normal, non-debug)
start:
	@echo "Starting training (non-debug) in container '${SERVICE}'..."
	@docker compose -f ${COMPOSE_FILE} ps --services --filter "status=running" | grep -q "${SERVICE}" || (echo "Container not running. Run 'make up' first." >&2; exit 1)
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc "/workspace/isaaclab/isaaclab.sh -p ${TRAIN_SCRIPT} --headless --enable_cameras"

# Start training under debugpy (wraps the helper script)
debug-run:
	@echo "Starting debug run (port=${DEBUG_PORT})..."
	@./scripts/debug_run.sh ${DEBUG_PORT}

# Kill debug/training processes inside the container
debug-kill:
	@echo "Killing debug/training processes in '${SERVICE}'..."
	@./scripts/debug_kill.sh

# Stop and remove the container
stop:
	@echo "Stopping ${SERVICE}..."
	docker compose -f ${COMPOSE_FILE} down

# Rebuild image and restart container
rebuild: stop build up

# Show container logs (follow)
logs:
	@echo "Tailing logs for ${SERVICE} (CTRL-C to stop)..."
	docker compose -f ${COMPOSE_FILE} logs -f ${SERVICE}

# Open a shell in the container
shell:
	@echo "Opening shell into ${SERVICE}..."
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash

# Kill then restart the training (useful for quick iteration)
restart: debug-kill start
