.PHONY: build up start start_sac start_ctrlsac start_eval debug-run debug-run-ctrlsac debug-kill stop rebuild logs shell restart help

# Docker Compose (Isaac Lab base image — needs profile "base" because the service is profile-gated)
COMPOSE_FILE := docker/docker-compose.yaml
SERVICE := isaac-lab-base
export COMPOSE_PROFILES ?= base

# In-container Isaac Lab launcher and skrl_ctrl scripts (bind-mount: ./source -> /workspace/isaaclab/source)
ISAACLAB_SH := /workspace/isaaclab/isaaclab.sh
SKRL_DIR := source/standalone/workflows/skrl_ctrl
SAC_TRAIN_SCRIPT := $(SKRL_DIR)/train_sac.py
CTRLSAC_TRAIN_SCRIPT := $(SKRL_DIR)/train.py
EVAL_SCRIPT := $(SKRL_DIR)/eval.py

DEBUG_PORT ?= 5678

# Extra CLI args after AppLauncher flags, e.g.:
#   make start_ctrlsac ARGS='--env_version legtrain-finetune --ckpt /workspace/isaaclab/runs/torch/.../agent_300000.pt'
#   make start_ctrlsac ARGS='--multitask --experiment_json /workspace/isaaclab/source/standalone/workflows/skrl_ctrl/configs/default_experiment.json'
ARGS ?=

# Eval (eval.py requires --folder). Example:
#   make start_eval EVAL_ARGS='--experiment legeval-predef --folder /workspace/isaaclab/runs/torch/.../checkpoints --ckpt agent_300000.pt --agent_type CTRLSAC'
EVAL_ARGS ?=

## help: Show this help
help:
	@grep -E '^## ' Makefile | sed 's/## //'

## build: Build the isaac-lab-base image
build:
	@echo "Building ${SERVICE} image..."
	docker compose -f ${COMPOSE_FILE} build ${SERVICE}

## up: Build (if needed) and start the container in the background
up: build
	@echo "Starting ${SERVICE} container (COMPOSE_PROFILES=$(COMPOSE_PROFILES))..."
	docker compose -f ${COMPOSE_FILE} up -d ${SERVICE}

# Shared guard: container must be running
_check_container:
	@docker compose -f ${COMPOSE_FILE} ps --services --filter "status=running" | grep -q "${SERVICE}" || (echo "Container not running. Run 'make up' first." >&2; exit 1)

## start / start_ctrlsac: Run CTRLSAC training (train.py shim → cli train-ctrlsac)
start: start_ctrlsac

start_ctrlsac: _check_container
	@echo "Starting CTRLSAC training in '${SERVICE}'..."
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(CTRLSAC_TRAIN_SCRIPT) --headless --enable_cameras $(ARGS)'

## start_sac: Run SAC baseline training (train_sac.py shim → cli train-sac)
start_sac: _check_container
	@echo "Starting SAC training in '${SERVICE}'..."
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(SAC_TRAIN_SCRIPT) --headless --enable_cameras $(ARGS)'

## start_eval: Run eval.py (set EVAL_ARGS with --folder, --ckpt, etc.)
start_eval: _check_container
	@echo "Starting eval in '${SERVICE}'..."
	@test -n "$(strip $(EVAL_ARGS))" || (echo "Set EVAL_ARGS, e.g. EVAL_ARGS='--experiment legeval-predef --folder /path/to/run --ckpt agent.pt --agent_type CTRLSAC'" >&2; exit 1)
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(EVAL_SCRIPT) --headless --enable_cameras $(EVAL_ARGS)'

## debug-run: Start train_sac.py under debugpy (see scripts/debug_run.sh)
debug-run:
	@echo "Starting debug run (port=${DEBUG_PORT})..."
	@./scripts/debug_run.sh ${DEBUG_PORT}

## debug-run-ctrlsac: Like debug-run but for CTRLSAC (train.py)
debug-run-ctrlsac:
	@echo "Starting CTRLSAC debug run (port=${DEBUG_PORT})..."
	@docker compose -f ${COMPOSE_FILE} ps --services --filter "status=running" | grep -q "${SERVICE}" || (echo "Container not running. Run 'make up' first." >&2; exit 1)
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc \
		'cd /workspace/isaaclab && $(ISAACLAB_SH) -p -m debugpy --listen 0.0.0.0:$(DEBUG_PORT) --wait-for-client $(CTRLSAC_TRAIN_SCRIPT) --headless --enable_cameras $(ARGS)'

## debug-kill: Kill debug/training Python processes inside the container
debug-kill:
	@echo "Killing debug/training processes in '${SERVICE}'..."
	@./scripts/debug_kill.sh

## stop: Stop and remove the container
stop:
	@echo "Stopping ${SERVICE}..."
	docker compose -f ${COMPOSE_FILE} down

## rebuild: Rebuild image and restart container
rebuild: stop build up

## logs: Follow container logs
logs:
	@echo "Tailing logs for ${SERVICE} (CTRL-C to stop)..."
	docker compose -f ${COMPOSE_FILE} logs -f ${SERVICE}

## shell: Interactive bash in the container
shell: _check_container
	@echo "Opening shell into ${SERVICE} (repo at /workspace/isaaclab)..."
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash

## restart: Kill stuck training then start CTRLSAC again
restart: debug-kill start_ctrlsac
