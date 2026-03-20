# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-tracking quadcopter direct RL environment."""

from __future__ import annotations

import torch

import gymnasium as gym
import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import subtract_frame_transforms

from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

from .configs import QuadcopterTrajectoryLinearEnvCfg
from .trajectory_generator import PolynomialTrajectoryGenerator


class QuadcopterTrajectoryEnv(DirectRLEnv):
    cfg: QuadcopterTrajectoryLinearEnvCfg

    def __init__(self, cfg: QuadcopterTrajectoryLinearEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._generator = PolynomialTrajectoryGenerator(
            self.device,
            self.num_envs,
            max_traj_dur=self.cfg.episode_length_s + 0.25,
            freq=1 / self.step_dt,
            mode=cfg.mode,
            predef_coeff=cfg.predefined_task_coeff,
        )
        self.lvl = self.cfg.profile[0]
        self._desired_trajectory_w = torch.zeros(self.num_envs, self._generator.N, 3, device=self.device)
        self._active_trajectory_command = torch.zeros(self.num_envs, 7, device=self.device)
        self._desired_trajectory_vel_w = torch.zeros(self.num_envs, self._generator.N, 3, device=self.device)

        self.episode_max_len = int(self.cfg.episode_length_s / (self.step_dt))

        self.episode_timesteps = torch.zeros(self.num_envs, device=self.device).type(torch.int64)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_rew",
                "vel_rew",
                "av_rew",
                "thrust_rew",
                "torques_rew",
                "survival_rew",
            ]
        }

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        self.single_observation_space = gym.spaces.Dict()

        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

        self.radiusdt = 0.05
        self.radius = 0.0

        self.eval = False

    def eval_mode(self):
        self.eval = True
        self._generator.activate_eval_mode()
        self.results = {}

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        window_wp = []
        window_vel = []
        for i in range(self.num_envs):
            desired_trajectory_window_b, _ = subtract_frame_transforms(
                self._robot.data.root_state_w[i, :3].repeat(self.cfg.window, 1),
                self._robot.data.root_state_w[i, 3:7].repeat(self.cfg.window, 1),
                self._desired_trajectory_w[i, self.episode_timesteps[i] : self.episode_timesteps[i] + self.cfg.window],
            )
            desired_trajectory_vel_window_w = self._desired_trajectory_vel_w[
                i, self.episode_timesteps[i] : self.episode_timesteps[i] + self.cfg.window, :
            ]
            window_wp.append(desired_trajectory_window_b)
            window_vel.append(desired_trajectory_vel_window_w)

            if self.eval and self._generator.curr_experiment_tracker[i].item() in self.results:
                self.results[self._generator.curr_experiment_tracker[i].item()]["pose"][self.episode_timesteps[i]] = (
                    self._robot.data.root_state_w[i, :3]
                )

        window_wp = torch.stack(window_wp).view(self.num_envs, -1)
        window_vel = torch.stack(window_vel).view(self.num_envs, -1)

        quat = self._robot.data.root_state_w[:, 3:7]
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
                quat,
                window_wp,
                window_vel,
            ],
            dim=-1,
        )
        if self.cfg.include_coeffecients:
            obs = torch.cat(
                [
                    obs,
                    self._active_trajectory_command,
                ],
                dim=-1,
            )

        self.episode_timesteps += 1
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        T = self.cfg.thresh_div

        Ft = (self._actions[:, 0] + 1.0) / 2.0
        torques = self._actions[:, 1:]

        distance_to_trajectory = torch.linalg.norm(
            self._desired_trajectory_w[torch.arange(self.num_envs), self.episode_timesteps] - self._robot.data.root_pos_w,
            dim=1,
        )
        pos_rew = T - distance_to_trajectory

        distance_to_desired_vel = torch.linalg.norm(
            self._desired_trajectory_vel_w[torch.arange(self.num_envs), self.episode_timesteps]
            - self._robot.data.root_lin_vel_w,
            dim=1,
        )
        vel_rew = T - distance_to_desired_vel

        hover_T = 1 / self.cfg.thrust_to_weight
        thrust_rew = 0.5 - torch.absolute(hover_T - Ft)
        torques_rew = 1 - torch.sum(torch.absolute(torques), axis=1)

        av_rew = torch.sum(self.cfg.thresh_stable - (torch.absolute(self._robot.data.root_ang_vel_w)), dim=1)

        survive_rew = 1 - self._get_dones()[0].long()

        rewards = {
            "pos_rew": pos_rew * self.cfg.pos_reward_scale * self.step_dt,
            "vel_rew": vel_rew * self.cfg.vel_reward_scale * self.step_dt,
            "av_rew": av_rew * self.cfg.av_rew_scale * self.step_dt,
            "thrust_rew": thrust_rew * self.cfg.thrust_rew_scale * self.step_dt,
            "torques_rew": torques_rew * self.cfg.torques_rew_scale * self.step_dt,
            "survival_rew": survive_rew * self.cfg.survival_rew_scale * self.step_dt,
        }
        reward = torch.exp(torch.sum(torch.stack(list(rewards.values())), dim=0))
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if self.eval:
            for env in env_ids:
                tid = self._generator.curr_experiment_tracker[env].item()
                if tid in self.results.keys() and "MSE" not in self.results[tid]:
                    self.results[tid]["crashed"] = torch.count_nonzero(self.reset_terminated[env]).item()
                    self.results[tid]["time_alive"] = self.episode_timesteps[env].item()
                    self.results[tid]["trajectory_legendre"] = self._generator.legendre[tid]
                    self.results[tid]["trajectory_monomial"] = self._generator.coefficients[tid]
                    self.results[tid]["total_reward_wo_survival"] = torch.sum(
                        torch.tensor([self._episode_sums[key][env] for key in self._episode_sums.keys() if key != "survival_rew"])
                    )
                    self.results[tid]["total_reward"] = torch.sum(
                        torch.tensor([self._episode_sums[key][env] for key in self._episode_sums.keys()])
                    )

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["curriculum_lvl"] = torch.tensor([self.lvl], device=self.device)

        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.num_envs > 1:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.episode_timesteps[env_ids] = 0
        self._actions[env_ids] = 0.0
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        if self.cfg.curriculum:
            self.update_task_difficulty()

        self._desired_trajectory_w[env_ids], self._desired_trajectory_vel_w[env_ids], self._active_trajectory_command[env_ids] = (
            self._generator.generate_trajectory(default_root_state[:, :3], len(env_ids), env_ids, offset_r=self.radius, lvl=self.lvl)
        )

        if self.eval:
            for env in env_ids:
                tid = self._generator.curr_experiment_tracker[env].item()
                if tid not in self.results:
                    self.results[tid] = {
                        "trajectory": self._desired_trajectory_w[env].clone(),
                        "pose": torch.zeros_like(self._desired_trajectory_w[env]),
                    }

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_traj_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                marker_cfg.prim_path = "/Visuals/Command/goal_traj_visualizer"
                self.goal_traj_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "immediate_wpt_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/immediate_wpt_visualizer"
                self.immediate_wpt_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_traj_visualizer.set_visibility(True)
            self.immediate_wpt_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_traj_visualizer"):
                self.goal_traj_visualizer.set_visibility(False)
            if hasattr(self, "immediate_wpt_visualizer"):
                self.immediate_wpt_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_traj_visualizer.visualize(self._desired_trajectory_w.view(-1, 3))
        for i in range(self.episode_timesteps.size().numel()):
            self.immediate_wpt_visualizer.visualize(self._desired_trajectory_w[i, self.episode_timesteps[i]].view(-1, 3))

    def update_task_difficulty(self):
        self.radius = self.radiusdt * (int(self._sim_step_counter / 1e5) + 1)
        self.lvl = self.cfg.profile[int((self._sim_step_counter / (2 * self.cfg.total_iterations)) * len(self.cfg.profile))]
