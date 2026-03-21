# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration classes for direct quadcopter tasks."""

from __future__ import annotations

import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip

from .ui_window import QuadcopterEnvWindow


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    num_observations = 12
    num_states = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


@configclass
class QuadcopterTrajectoryLinearEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    window = 10
    num_observations = 13 + window * 6
    num_states = 0
    debug_vis = True
    noise = False
    include_coeffecients = True
    thresh_div = 0.3
    thresh_stable = 1.5
    mode = 0

    curriculum = False
    profile = [10]
    total_iterations = 3e5
    buffer_history = 100
    # When True and eval_mode() is used: crash/timeout does not reset; wrenches zeroed (e.g. eval video).
    freeze_on_done_in_eval = False
    # When > 0 and eval_mode(): place robot on XY disk around trajectory start (see trajectory_env); uses offset_r=0 on generator.
    initial_pose_xy_radius_max_m: float = 0.0
    # Allow initial_pose_xy when not eval (e.g. offline data collection without eval_mode); default off for training safety.
    initial_pose_xy_when_not_eval: bool = False

    # Termination (``died``): altitude + attitude + spin (see ``_get_dones`` in trajectory_env).
    crash_z_min_m: float = 0.1
    crash_z_max_m: float = 2.0
    # Minimum dot(body +Z, world +Z). Below this => crashed (large roll/pitch / inverted).
    crash_body_up_z_dot_min: float = 0.0
    # Body-frame angular velocity norm (rad/s); above => crashed (unrecoverable tumble).
    crash_max_ang_vel_rad_s: float = 28.0

    ui_window_class_type = QuadcopterEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    pos_reward_scale = 25
    vel_reward_scale = 2
    av_rew_scale = 5
    thrust_rew_scale = 10
    torques_rew_scale = 1
    survival_rew_scale = 5
    predefined_task_coeff = None

    if include_coeffecients:
        num_observations += 7

    def __post_init__(self):
        self.viewer.eye = [0.0, 4.0, 7.5]
        self.viewer.lookat = [1.0, 0.0, 0.0]
        self.viewer.up = [0.0, 0.0, 0.0]
        # Higher than default (1280, 720) for clearer RecordVideo / rgb_array captures.
        self.viewer.resolution = (1920, 1080)


@configclass
class QuadcopterTrajectoryLegendreTrainingEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 5
    curriculum = True
    profile = [3]


@configclass
class QuadcopterTrajectoryTrainingRandomTaskEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 5
    curriculum = True
    noise = True
    profile = [3, 3, 6, -3, -3]
    buffer_history = 1


@configclass
class QuadcopterTrajectoryLegendreEvalEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 7
    freeze_on_done_in_eval = True


@configclass
class QuadcopterTrajectoryPreDefEvalEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 7
    np.random.seed(42)
    predefined_task_coeff = [[0, 0, 0, 0, 0, 1, 0]]
    freeze_on_done_in_eval = True


@configclass
class QuadcopterTrajectoryDownStreamFinetuneEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 7
    predefined_task_coeff = [[0, 0, 0, 0, 0, 1, 0]]
    total_iterations = int(5e4)


@configclass
class QuadcopterTrajectoryRefineEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    """Single-task refinement / few-shot: override ``predefined_task_coeff`` at runtime via ``gym.make(..., cfg=...)``."""

    mode = 7
    curriculum = False
    profile = [10]
    predefined_task_coeff = [[0, 0, 0, 0, 0, 1, 0]]


@configclass
class QuadcopterTrajectoryLegendreOODEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 6


@configclass
class QuadcopterTrajectoryOODEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 4

    def __post_init__(self):
        self.viewer.eye = [-0.0, 4.0, 7.5]
        self.viewer.lookat = [1.0, 4.0, 0.0]
        self.viewer.up = [-1.0, 0.0, 0.0]
        self.viewer.resolution = (1920, 1080)
