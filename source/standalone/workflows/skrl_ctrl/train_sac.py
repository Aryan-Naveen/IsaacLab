import torch
import torch.nn as nn

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
import os
import gymnasium as gym
# import gym
import numpy as np


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


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

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions,   1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the Isaac Lab environment
cli_args = ["--video"]
# load and wrap the Isaac Gym environment
env = load_isaaclab_env(task_name="Isaac-Quadcopter-Trajectory-Direct-v0", num_envs=64, cli_args=cli_args)

video_kwargs = {
    "video_folder": os.path.join("runs/torch/Quadcopter", "videos", "sac_train"),
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
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 1024
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["random_timesteps"] = 25e3
cfg["learning_starts"] = 25e3
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/QuadCopter"

agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": int(1e6), "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)



# start training
trainer.train()