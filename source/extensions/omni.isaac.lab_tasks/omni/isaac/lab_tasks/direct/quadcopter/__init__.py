# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quacopter environment."""

import gymnasium as gym

from . import agents
from .quadcopter_env import *

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-legtrain-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryLegendreTrainingEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-legtrain-random-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryTrainingRandomTaskEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-legeval-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryLegendreEvalEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-legeval-predef-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryPreDefEvalEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-legtrain-finetune-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryDownStreamFinetuneEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-Refine-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryRefineEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-legood-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryLegendreOODEnvCfg,
    },
)

gym.register(
    id="Isaac-Quadcopter-OOD-Trajectory-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterTrajectoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterTrajectoryOODEnvCfg,
    },
)
