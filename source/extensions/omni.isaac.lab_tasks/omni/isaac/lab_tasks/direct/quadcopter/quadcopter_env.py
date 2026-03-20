# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatible exports for direct quadcopter tasks (implementation split under this package)."""

from __future__ import annotations

from .configs import (
    QuadcopterEnvCfg,
    QuadcopterTrajectoryDownStreamFinetuneEnvCfg,
    QuadcopterTrajectoryLegendreEvalEnvCfg,
    QuadcopterTrajectoryLegendreOODEnvCfg,
    QuadcopterTrajectoryLegendreTrainingEnvCfg,
    QuadcopterTrajectoryLinearEnvCfg,
    QuadcopterTrajectoryOODEnvCfg,
    QuadcopterTrajectoryPreDefEvalEnvCfg,
    QuadcopterTrajectoryRefineEnvCfg,
    QuadcopterTrajectoryTrainingRandomTaskEnvCfg,
)
from .simple_env import QuadcopterEnv
from .trajectory_env import QuadcopterTrajectoryEnv
from .trajectory_generator import PolynomialTrajectoryGenerator
from .ui_window import QuadcopterEnvWindow

__all__ = [
    "QuadcopterEnv",
    "QuadcopterEnvCfg",
    "QuadcopterEnvWindow",
    "QuadcopterTrajectoryDownStreamFinetuneEnvCfg",
    "QuadcopterTrajectoryEnv",
    "QuadcopterTrajectoryLegendreEvalEnvCfg",
    "QuadcopterTrajectoryLegendreOODEnvCfg",
    "QuadcopterTrajectoryLegendreTrainingEnvCfg",
    "QuadcopterTrajectoryLinearEnvCfg",
    "QuadcopterTrajectoryOODEnvCfg",
    "QuadcopterTrajectoryPreDefEvalEnvCfg",
    "QuadcopterTrajectoryRefineEnvCfg",
    "QuadcopterTrajectoryTrainingRandomTaskEnvCfg",
    "PolynomialTrajectoryGenerator",
]
