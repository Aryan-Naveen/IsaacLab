# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

import gymnasium as gym
import numpy as np
##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    num_observations = 12
    num_states = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


@configclass
class QuadcopterTrajectoryLinearEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    num_actions = 4
    window = 10
    num_observations = 13 + window*6
    num_states = 0
    debug_vis = True
    thresh_div = 0.3
    thresh_stable = 1.5
    mode = 0
    


    ui_window_class_type = QuadcopterEnvWindow

    # simulation
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    pos_reward_scale = 10
    vel_reward_scale = 2
    av_rew_scale = 5
    thrust_rew_scale = 10
    torques_rew_scale = 1
    survival_rew_scale = 5
    def __post_init__(self):
        """Post initialization."""
        # Set viewer settings
        self.viewer.eye = [0.0, 4.0, 7.5]  # Positioned directly above
        self.viewer.lookat = [1.0, 0.0, 0.0]  # Looking at the center of the environment
        self.viewer.up = [0.0, 0.0, 0.0]  # Up direction is now along the y-axis


@configclass
class QuadcopterTrajectoryMultiEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 3

@configclass
class QuadcopterTrajectoryDiagonalEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 1
    def __post_init__(self):
        """Post initialization."""
        # Set viewer settings
        self.viewer.eye = [-0.0, 4.0, 7.5]  # Positioned directly above
        self.viewer.lookat = [1.0, 4.0, 0.0]  # Looking at the center of the environment
        self.viewer.up = [-1.0, 0.0, 0.0]  # Up direction is now along the y-axis

@configclass
class QuadcopterTrajectoryQuadraticEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 2
    def __post_init__(self):
        """Post initialization."""
        # Set viewer settings
        self.viewer.eye = [-0.0, 4.0, 7.5]  # Positioned directly above
        self.viewer.lookat = [1.0, 4.0, 0.0]  # Looking at the center of the environment
        self.viewer.up = [-1.0, 0.0, 0.0]  # Up direction is now along the y-axis

    
@configclass
class QuadcopterTrajectoryOODEnvCfg(QuadcopterTrajectoryLinearEnvCfg):
    mode = 4
    episode_length_s = 4.0
    def __post_init__(self):
        """Post initialization."""
        # Set viewer settings
        self.viewer.eye = [-0.0, 4.0, 7.5]  # Positioned directly above
        self.viewer.lookat = [1.0, 4.0, 0.0]  # Looking at the center of the environment
        self.viewer.up = [-1.0, 0.0, 0.0]  # Up direction is now along the y-axis
        

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
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

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


class PolynomialTrajectoryGenerator:
    def __init__(self, device, max_traj_dur= 10, freq=100, vn=0.5, mode=0):
        self.N = int(max_traj_dur*freq)
        self.vn = vn
        self.H = max_traj_dur
        self.device = device
        
        if mode == 0:
            self.coefficients = torch.tensor([[0, 0, 0, 0, 0, 0]])
        elif mode == 1:
            self.coefficients = torch.tensor([[0, 1, 0, 0, 0, 0]])
        elif mode == 2:            
            self.coefficients = torch.tensor([[0, 0, 1, 0, 0, 0]])
        elif mode == 3:
            self.coefficients = torch.tensor([[0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0]])
        elif mode == 4:
            self.coefficients = torch.tensor([[0, 1, 1, 0, 0, 0]])
            
        
        self.coefficients = self.coefficients.to(self.device)
    
    def generate_trajectory(self, rpose, num_environments, offset_r=0.05):
        pos0 = torch.rand(num_environments, 1,3, device=self.device)*offset_r + rpose.unsqueeze(1)       
        traj_ = torch.zeros(num_environments, self.N, 3, device =self.device)
        traj_[:, :, 0] = torch.linspace(0, self.H * self.vn, self.N, device=self.device)        

        # traj = torch.repeat_interleave(pos0, self.N, axis=1) + traj_        
        
        # velocity = torch.zeros(traj.shape, device=self.device)
        # velocity[:, :, 1] = self.vn
        # return traj, velocity
        
        random_indices = torch.randint(0, self.coefficients.shape[0], (num_environments,), device=self.device)
        selected_coeffs = self.coefficients[random_indices]  # Shape: (num_environments, num_coeffs)

        # Generate x values
        x = traj_[:, :, 0]  # Shape: (num_environments, self.N)

        # Polynomial computation for y-axis (traj_[:, :, 1])
        powers = torch.arange(selected_coeffs.shape[1], device=self.device)  # Shape: (num_coeffs)
        x_powers = x.unsqueeze(2).pow(powers)  # Shape: (num_environments, self.N, num_coeffs)
        traj_[:, :, 1] = torch.sum(selected_coeffs.unsqueeze(1) * x_powers, dim=2)  # Shape: (num_environments, self.N)

        # Compute the derivative (velocity for y-axis)
        deriv_powers = powers[:-1]  # Remove the constant term
        deriv_coeffs = selected_coeffs[:, 1:] * powers[1:]  # Derivative coefficients


        vx = torch.ones_like(x) * self.vn  # Constant velocity for x
        vy = torch.sum(deriv_coeffs.unsqueeze(1) * x.unsqueeze(2).pow(deriv_powers), dim=2)  # Derivative for y
        vz = torch.zeros_like(vx)  # Derivative for z (can be distinct)

        velocities = torch.stack([vx, vy, vz], dim=2)  # Shape: (num_environments, self.N, 3)
        
        traj = torch.repeat_interleave(pos0, self.N, axis=1) + traj_
        
        
        

        return traj, velocities        

class QuadcopterTrajectoryEnv(DirectRLEnv):
    cfg: QuadcopterTrajectoryEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._generator = PolynomialTrajectoryGenerator(self.device, max_traj_dur = self.cfg.episode_length_s + 2.0, freq=1/self.step_dt, mode=cfg.mode)
        self._desired_trajectory_w = torch.zeros(self.num_envs, self._generator.N, 3, device=self.device)
        self._desired_trajectory_vel_w = torch.zeros(self.num_envs, self._generator.N, 3, device=self.device)

        self.positions = []


        self.episode_timesteps = torch.zeros(self.num_envs, device=self.device).type(torch.int64)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_rew",
                "vel_rew",
                "av_rew",
                "thrust_rew",
                "torques_rew",
                "survival_rew"
            ]
        }
        
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        
        

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)


        self.radiusdt = 0.05
        self.radius = 0.0
        
        self.eval = False

    def eval_mode(self):
        self.eval = True

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        # Assuming agent_position is a list or tuple [x, y, z]
        
        agent_position = self._robot.data.root_state_w[0, :3]
        
        
        if self.eval:
            self.positions.append(self._robot.data.root_state_w[0, :3].cpu().numpy())
        # self.cfg.viewer.eye = [agent_position[0], agent_position[1], 5.0]
        # self.cfg.viewer.lookat = agent_position
        # self.cfg.viewer.up = [-1.0, 0.0, 0.0]  # Keep the up direction the same


    def _get_observations(self) -> dict:
        window_wp = []
        window_vel = []
        for i in range(self.num_envs):
            desired_trajectory_window_b, _ = subtract_frame_transforms(
                self._robot.data.root_state_w[i, :3].repeat(self.cfg.window, 1), self._robot.data.root_state_w[i, 3:7].repeat(self.cfg.window, 1), self._desired_trajectory_w[i, self.episode_timesteps[i]: self.episode_timesteps[i] + self.cfg.window]
            )
            desired_trajectory_vel_window_w = self._desired_trajectory_vel_w[i, self.episode_timesteps[i]: self.episode_timesteps[i] + self.cfg.window, :]
            window_wp.append(desired_trajectory_window_b)
            window_vel.append(desired_trajectory_vel_window_w)
        
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
                window_vel
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        self.episode_timesteps += 1
        return observations

    

    def _get_rewards(self) -> torch.Tensor:                
        
        T = self.cfg.thresh_div
        
        Ft = (self._actions[:, 0] + 1.0) / 2.0
        torques =  self._actions[:, 1:]
        
        distance_to_trajectory = torch.linalg.norm(self._desired_trajectory_w[torch.arange(self.num_envs), self.episode_timesteps] - self._robot.data.root_pos_w, dim=1)
        pos_rew = T - distance_to_trajectory
        
        distance_to_desired_vel = torch.linalg.norm(self._desired_trajectory_vel_w[torch.arange(self.num_envs), self.episode_timesteps] - self._robot.data.root_lin_vel_w, dim=1)
        vel_rew = T - distance_to_desired_vel      
        
        hover_T = 1/self.cfg.thrust_to_weight
        thrust_rew = 0.5 - torch.absolute(hover_T - Ft)
        torques_rew = 1 - torch.sum(torch.absolute(torques), axis=1)
        
        av_rew = torch.sum(self.cfg.thresh_stable - (torch.absolute(self._robot.data.root_ang_vel_w)), dim=1)

        survive_rew = 1-self._get_dones()[0].long()


        rewards = {
            "pos_rew": pos_rew * self.cfg.pos_reward_scale * self.step_dt,
            "vel_rew": vel_rew * self.cfg.vel_reward_scale * self.step_dt,
            "av_rew": av_rew * self.cfg.av_rew_scale * self.step_dt,
            "thrust_rew": thrust_rew * self.cfg.thrust_rew_scale * self.step_dt,
            "torques_rew": torques_rew * self.cfg.torques_rew_scale * self.step_dt,
            "survival_rew": survive_rew * self.cfg.survival_rew_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
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

        # Logging
        final_distance_to_trajectory = torch.linalg.norm(
            self._desired_trajectory_w[env_ids, self.episode_timesteps[env_ids]] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # print(extras["Episode_Termination/died"])
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_trajectory"] = final_distance_to_trajectory.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))


        self.episode_timesteps[env_ids] = 0
        self._actions[env_ids] = 0.0
        # Sample new commands
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self.update_task_difficulty(int(self._sim_step_counter/1e5) + 1)
        
        if self.eval: self.radius = 0.0
        self._desired_trajectory_w[env_ids], self._desired_trajectory_vel_w[env_ids] = self._generator.generate_trajectory(default_root_state[:, :3], len(env_ids), offset_r = self.radius)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_traj_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_traj_visualizer"
                self.goal_traj_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "immediate_wpt_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/immediate_wpt_visualizer"
                self.immediate_wpt_visualizer = VisualizationMarkers(marker_cfg)                
            # set their visibility to true
            self.goal_traj_visualizer.set_visibility(True)
            self.immediate_wpt_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_traj_visualizer"):
                self.goal_traj_visualizer.set_visibility(False)
            if hasattr(self, "immediate_wpt_visualizer"):
                self.immediate_wpt_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_traj_visualizer.visualize(self._desired_trajectory_w.view(-1, 3))
        for i in range(self.episode_timesteps.size().numel()):
            self.immediate_wpt_visualizer.visualize(self._desired_trajectory_w[i, self.episode_timesteps[i]].view(-1, 3))

    def update_task_difficulty(self, level):
            self.radius = self.radiusdt * level