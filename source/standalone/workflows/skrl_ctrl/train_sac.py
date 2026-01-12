import torch
import torch.nn as nn
import torch.nn.functional as F


# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omni.isaac.lab.utils.dict import print_dict
from networks.actor import DiagGaussianActor, DiagGaussianActorPolicy
import os
import gymnasium as gym
# import gym
import numpy as np


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
# define hidden dimension
actor_hidden_dim = 512
actor_hidden_depth = 3



# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


GRAVITY = 9.81


class MellingerControllerNN(nn.Module):
    def __init__(self, mass=0.034, thrust_to_weight = 1.9, robot_weight=0.333, moment_scale = 0.01):
        super().__init__()

        self.mass = mass
        self.tw = thrust_to_weight
        self.W = robot_weight
        self.moment_scale = moment_scale

        # Position gains
        self.kp_xy = nn.Parameter(torch.tensor(0.0))
        self.kd_xy = nn.Parameter(torch.tensor(0.0))
        self.ki_xy = nn.Parameter(torch.tensor(0.0))
        self.kp_z  = nn.Parameter(torch.tensor(0.0))
        self.kd_z  = nn.Parameter(torch.tensor(0.0))
        self.ki_z  = nn.Parameter(torch.tensor(0.0))

        # Attitude gains
        self.kR_xy = nn.Parameter(torch.tensor(0.0))
        self.kw_xy = nn.Parameter(torch.tensor(0.0))
        self.kR_z  = nn.Parameter(torch.tensor(0.0))
        self.kw_z  = nn.Parameter(torch.tensor(0.0))

        # Integrators
        self.register_buffer("i_p", torch.zeros(3))
        self.register_buffer("i_R", torch.zeros(3))

    def reset(self):
        self.i_p.zero_()
        self.i_R.zero_()

    def forward(self, obs, dt=0.01):
        """
        obs: (B,17)
        returns: (B,4) actions in [-1, 1] compatible with Isaac quadrotor
        """

        B = obs.shape[0]

        # ---------------- unpack ----------------
        ep    = obs[:, 0:3]     # position error
        ev    = obs[:, 3:6]     # velocity error
        yaw   = obs[:, 6]
        quat  = obs[:, 7:11]   # (x,y,z,w)
        omega = obs[:, 11:14]  # body rates

        # ---------------- POSITION PID ----------------
        # integrator must be detached from graph
        self.i_p = (self.i_p + ep.mean(0) * dt).detach()

        a_des = torch.zeros_like(ep)

        a_des[:, 0] = self.kp_xy * ep[:, 0] + self.kd_xy * ev[:, 0] + self.ki_xy * self.i_p[0]
        a_des[:, 1] = self.kp_xy * ep[:, 1] + self.kd_xy * ev[:, 1] + self.ki_xy * self.i_p[1]
        a_des[:, 2] = (
            self.kp_z * ep[:, 2]
            + self.kd_z * ev[:, 2]
            + self.ki_z * self.i_p[2]
            + GRAVITY
        )

        # physical force
        F = self.mass * a_des

        # desired body z-axis
        zB_des = F / (torch.norm(F, dim=1, keepdim=True) + 1e-6)

        # ---------------- BODY Z AXIS ----------------
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        zB = torch.stack(
            [
                2 * (x * z + y * w),
                2 * (y * z - x * w),
                1 - 2 * (x * x + y * y),
            ],
            dim=1,
        )

        # ---------------- ATTITUDE ERROR ----------------
        eR = torch.cross(zB, zB_des, dim=1)

        self.i_R = (self.i_R + eR.mean(0) * dt).detach()

        # ---------------- MOMENTS ----------------
        M = torch.zeros_like(zB)

        M[:, 0] = -self.kR_xy * eR[:, 0] - self.kw_xy * omega[:, 0]
        M[:, 1] = -self.kR_xy * eR[:, 1] - self.kw_xy * omega[:, 1]
        M[:, 2] = -self.kR_z  * eR[:, 2] - self.kw_z  * omega[:, 2]

        # ---------------- ISAAC ACTUATOR MAPPING ----------------
        # thrust along body z
        T = (F * zB).sum(1)

        # normalized thrust: hover = 0
        a_T = 2 * T / (self.tw * self.W) - 1

        # normalized moments
        a_M = M / self.moment_scale

        # Isaac already clips moments — only squash thrust
        a_T = torch.tanh(a_T)
        a_M = torch.clamp(a_M, -1.0, 1.0)

        return torch.cat([a_T.unsqueeze(1), a_M], dim=1)

# define models (stochastic and deterministic models) using mixins
class StochasticMellingerActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = MellingerControllerNN()
        self.mlp_preprocess = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 15),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.mlp_preprocess(inputs["states"])
        return self.net(x), self.log_std_parameter, {}



class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the Isaac Lab environment
cli_args = ["--video", "--enable_cameras"]  # enable video recording and camera rendering
# load and wrap the Isaac Gym environment
task_version = "legtrain"
finetune = task_version == 'legtrain-finetune'
task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"
num_envs = 32 if not finetune else 1
env = load_isaaclab_env(task_name=task_name, num_envs=num_envs, cli_args=cli_args)

video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task_version}/", "videos", "train", "SAC"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": 400,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device



# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=int(1e5), num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticMellingerActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["random_timesteps"] = 12e3 if not finetune else 0
cfg["learning_starts"] = 12e3 if not finetune else 25e3
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True if not finetune else False
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0 if not finetune else 0.06
# logging to wandb and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 10000 if not finetune else 1000
cfg["experiment"]["directory"] = f"runs/torch/{task_name}/SAC"
cfg["experiment"]["experiment_name"] = f"{task_name}-SAC"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {
    "project": "Isaac-Lab-Quadcopter-SAC",
    "name": f"{task_name}-SAC",
    "config": {
        "task": task_name,
        "num_envs": num_envs,
        "batch_size": 256,
    }
}


agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
timesteps = int(3e5) if not finetune else int(5e4)
if finetune:
    agent.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/SAC/25-02-01_23-09-09-780539_SAC/checkpoints/agent_300000.pt")
    memory.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/SAC/memories/25-02-11_04-00-39-059963_memory_0x7fb974303e50.pt")

cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
memory.save(cfg["experiment"]["directory"], format = 'pt')

# start training
trainer.train()