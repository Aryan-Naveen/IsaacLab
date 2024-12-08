import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer, StepTrainer
from skrl.utils import set_seed

from sac.actor import DiagGaussianActor
from sac.critic import Critic
from sac.feature import Phi, Mu, Theta

from ctrlsac_agent import CTRLSACAgent
from omni.isaac.lab.utils.dict import print_dict
import os
import gymnasium as gym
# import gym
import numpy as np

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


cli_args = ["--video"]
# load and wrap the Isaac Gym environment
task_version = "Linear"
task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"
env = load_isaaclab_env(task_name = task_name, num_envs=64, cli_args=cli_args)

video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task_version}/", "videos", "train"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": 400,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device


# instantiate a memory as experience replay
memory_size=int(1e5)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

# define hidden dimension
actor_hidden_dim = 256
actor_hidden_depth = 2

# define feature dimension 
feature_dim = 2048
feature_hidden_dim = 1024

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = DiagGaussianActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

models["critic_1"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

models["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

models["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   device = device)

models["target_critic_2"] = Critic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                device = device)



# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 1024
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["weight_decay"] = 0
cfg["random_timesteps"] = 25e3
cfg["learning_starts"] = 25e3
cfg["grad_norm_clip"] = 1.0
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 10000
cfg['use_feature_target'] = False
cfg['extra_feature_steps'] = 3
cfg['target_update_period'] = 2
cfg['eval'] = False
cfg['phi_and_nabla_mu_lr'] = 0.003
cfg['alpha'] = 0.1
cfg['auto_entropy_tuning']=True
cfg['num_noises']=1000
cfg['critic_elu_layer_regularizer_lambda'] = 0
cfg['DARL_noise_a'] = 0.3
cfg['DARL_noise_b'] = 0.1
cfg['sigma_scale_factor'] = 0.449



cfg["experiment"]["directory"] = f"runs/torch/{task_name}/CTRL-SAC/{cfg['extra_feature_steps']}-{cfg['feature_learning_rate']}-{feature_hidden_dim}-{feature_dim}-{memory_size}-small"


agent = DIFFSRSACAgent(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

cfg_trainer = {"timesteps": int(1e6), "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# train the agent(s)
trainer.train()


