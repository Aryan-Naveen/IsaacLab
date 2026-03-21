.PHONY: build up start start_sac start_ctrlsac start_eval start_eval_one start_eval_batch start_finetune debug-run debug-run-ctrlsac debug-kill stop rebuild logs shell restart help

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
EVAL_BATCH_SCRIPT := $(SKRL_DIR)/eval_batch.py
EVAL_ONE_SCRIPT := $(SKRL_DIR)/eval_one.py
FINETUNE_SCRIPT := $(SKRL_DIR)/finetune.py

DEBUG_PORT ?= 5678

# Extra CLI args after AppLauncher flags, e.g.:
#   make start_ctrlsac ARGS='--env_version legtrain-finetune --ckpt /workspace/isaaclab/runs/torch/.../agent_300000.pt'
#   make start_ctrlsac ARGS='--multitask --experiment_json /workspace/isaaclab/source/standalone/workflows/skrl_ctrl/configs/default_experiment.json'
ARGS ?=

# Eval (eval.py requires --folder). Example:
#   make start_eval EVAL_ARGS='--experiment legeval-predef --folder /workspace/isaaclab/runs/torch/.../checkpoints --ckpt agent_300000.pt --agent_type CTRLSAC'
EVAL_ARGS ?=

# Preset-driven eval / finetune (JSON or YAML if PyYAML installed).
#   Bare filename -> /workspace/isaaclab/$(SKRL_DIR)/configs/presets/<name>
#   Repo-relative (contains /, not absolute) -> /workspace/isaaclab/<path>
#   Absolute path -> unchanged
# Examples:
#   make start_eval_one EVAL_PRESET=eval_one_example.json
#   make start_eval_batch EVAL_PRESET=batch_basis3_B128.json
#   make start_finetune EVAL_PRESET=finetune_ctrlsac_multi_deg3.json
# Optional overrides: ARGS='--seed 1 --folder /path --ckpt agent.pt'
EVAL_PRESET ?=

# In-container path to default preset folder (used when EVAL_PRESET has no /)
SKRL_PRESET_DIR := /workspace/isaaclab/$(SKRL_DIR)/configs/presets
# Pure Make (no $(shell case … esac)): Make treats ")" in shell "*)" as end of $(shell …).
EVAL_PRESET_RESOLVED = $(if $(strip $(EVAL_PRESET)),$(if $(filter /%,$(EVAL_PRESET)),$(EVAL_PRESET),$(if $(findstring /,$(EVAL_PRESET)),/workspace/isaaclab/$(EVAL_PRESET),$(SKRL_PRESET_DIR)/$(EVAL_PRESET))),)

# --- eval-batch quick mode (used when EVAL_BATCH_ARGS and EVAL_PRESET are empty) ---
#   make start_eval_batch EVAL_BATCH_AGENT=sac EVAL_BATCH_VIDEO=1
#   make start_eval_batch EVAL_BATCH_FOLLOW=1 EVAL_BATCH_HQ_RENDER=1
#   make start_eval_batch EVAL_BATCH_AGENT=ctrlsac
# Agent: sac | SAC -> SAC + EVAL_BATCH_FOLDER_SAC; anything else -> CTRLSAC-multi + EVAL_BATCH_FOLDER_CTRL
EVAL_BATCH_AGENT ?= ctrlsac
EVAL_BATCH_VIDEO ?= 0
EVAL_BATCH_FOLLOW ?= 0
# 1 = --hq_viewport_render (higher RTX quality headless kit; slower)
EVAL_BATCH_HQ_RENDER ?= 0
EVAL_BATCH_B ?= 1
EVAL_BATCH_CKPT_CTRL ?= best_agent.pt
EVAL_BATCH_CKPT_SAC ?= agent_290000.pt
EVAL_BATCH_TASK ?= Isaac-Quadcopter-Refine-Trajectory-Direct-v0
EVAL_BATCH_COEFF_JSON ?= /workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/quadcopter/trajectory_cfgs/custom_traj.json
EVAL_BATCH_FOLDER_CTRL ?= /workspace/isaaclab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/CTRL-SAC-True/ckpts/checkpoints
EVAL_BATCH_FOLDER_SAC ?= /workspace/isaaclab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/SAC/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0-SAC/checkpoints
EVAL_BATCH_OUT_CTRL ?= /workspace/isaaclab/runs/eval_batch_bundle_ctrlsac.pt
EVAL_BATCH_OUT_SAC ?= /workspace/isaaclab/runs/eval_batch_bundle_sac.pt
# Optional: single output path for both agents (overrides OUT_CTRL / OUT_SAC when set)
EVAL_BATCH_OUT ?=

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

## start_eval: Legacy eval.py (set EVAL_ARGS with --folder, --ckpt, etc.)
start_eval: _check_container
	@echo "Starting eval in '${SERVICE}'..."
	@test -n "$(strip $(EVAL_ARGS))" || (echo "Set EVAL_ARGS, e.g. EVAL_ARGS='--experiment legeval-predef --folder /path/to/run --ckpt agent.pt --agent_type CTRLSAC'" >&2; exit 1)
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(EVAL_SCRIPT) --headless --enable_cameras $(EVAL_ARGS)'

## start_eval_one: Single trajectory + follower video + metrics.json (preset JSON; edit checkpoint paths in preset file)
start_eval_one: _check_container
	@echo "Starting eval-one in '${SERVICE}'..."
	@test -n "$(strip $(EVAL_PRESET))" || (echo "Set EVAL_PRESET, e.g. EVAL_PRESET=eval_one_example.json" >&2; exit 1)
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(EVAL_ONE_SCRIPT) --headless --enable_cameras --preset "$(EVAL_PRESET_RESOLVED)" $(ARGS)'

## start_finetune: Few-shot offline rounds (preset JSON; see configs/presets/finetune_ctrlsac_multi_deg3.json)
start_finetune: _check_container
	@echo "Starting finetune in '${SERVICE}'..."
	@test -n "$(strip $(EVAL_PRESET))" || (echo "Set EVAL_PRESET, e.g. EVAL_PRESET=finetune_ctrlsac_multi_deg3.json" >&2; exit 1)
	docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(FINETUNE_SCRIPT) --headless --enable_cameras --preset "$(EVAL_PRESET_RESOLVED)" $(ARGS)'

## start_eval_batch: eval_batch.py — preset (EVAL_PRESET), raw args (EVAL_BATCH_ARGS), or legacy quick mode
start_eval_batch: _check_container
	@echo "Starting eval-batch in '${SERVICE}'..."
	@if [ -n "$(strip $(EVAL_BATCH_ARGS))" ]; then \
	  docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(EVAL_BATCH_SCRIPT) --headless --enable_cameras $(EVAL_BATCH_ARGS)'; \
	elif [ -n "$(strip $(EVAL_PRESET))" ]; then \
	  docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc 'cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(EVAL_BATCH_SCRIPT) --headless --enable_cameras --preset "$(EVAL_PRESET_RESOLVED)" $(ARGS)'; \
	else \
	  case "$(EVAL_BATCH_AGENT)" in sac|SAC) _T=SAC; _F='$(EVAL_BATCH_FOLDER_SAC)'; _O='$(EVAL_BATCH_OUT_SAC)'; _C='$(EVAL_BATCH_CKPT_SAC)' ;; \
	    *) _T=CTRLSAC-multi; _F='$(EVAL_BATCH_FOLDER_CTRL)'; _O='$(EVAL_BATCH_OUT_CTRL)'; _C='$(EVAL_BATCH_CKPT_CTRL)' ;; esac; \
	  if [ -n "$(strip $(EVAL_BATCH_OUT))" ]; then _O='$(EVAL_BATCH_OUT)'; fi; \
	  _V=''; \
	  if [ "$(EVAL_BATCH_FOLLOW)" = "1" ]; then _V=' --record_follow_video'; \
	  elif [ "$(EVAL_BATCH_VIDEO)" = "1" ]; then _V=' --record_topdown_video'; fi; \
	  _HQ=''; [ "$(EVAL_BATCH_HQ_RENDER)" = "1" ] && _HQ=' --hq_viewport_render'; \
	  docker compose -f ${COMPOSE_FILE} exec ${SERVICE} bash -lc "cd /workspace/isaaclab && $(ISAACLAB_SH) -p $(EVAL_BATCH_SCRIPT) --headless --enable_cameras \
	    --task '$(EVAL_BATCH_TASK)' --coeff_json '$(EVAL_BATCH_COEFF_JSON)' \
	    --folder '$$_F' --ckpt '$$_C' --out '$$_O' --B $(EVAL_BATCH_B) --agent_type $$_T$$_V$$_HQ"; \
	fi

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
