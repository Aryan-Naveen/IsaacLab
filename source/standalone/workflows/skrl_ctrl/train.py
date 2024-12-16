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

from sac.actor import DiagGaussianActor, StochasticActor, DiagGaussianActorPolicy
from sac.critic import Critic, TestCritic
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
task_version = "Multi"
multitask = True
task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"
env = load_isaaclab_env(task_name = task_name, num_envs=32, cli_args=cli_args)

video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task_version}/{str(multitask)}", "videos", "train"),
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
actor_hidden_dim = 512
actor_hidden_depth = 3

# define feature dimension 
feature_dim = 512
feature_hidden_dim = 1024

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticActor(observation_space = env.observation_space,
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


models["phi"] = Phi(observation_space = env.observation_space, 
				    action_space = env.action_space, 
				    feature_dim = feature_dim, 
				    hidden_dim = feature_hidden_dim,
                    device = device
                )

models["frozen_phi"] = Phi(observation_space = env.observation_space, 
    				       action_space = env.action_space, 
	    			       feature_dim = feature_dim, 
		    	           hidden_dim = feature_hidden_dim,
                           device = device
                        )

models["theta"] = Theta(
    		        observation_space = env.observation_space,
		            action_space = env.action_space, 
		            feature_dim = feature_dim, 
                    device = device
                )

models["mu"] = Mu(
                observation_space = env.observation_space, 
                action_space = env.action_space, 
                feature_dim = feature_dim, 
                hidden_dim = feature_hidden_dim,
                device = device
            )

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["weight_decay"] = 0
cfg["feature_learning_rate"] = 1e-4
cfg["random_timesteps"] = 25e3
cfg["learning_starts"] = 25e3
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-5
cfg["initial_entropy_value"] = 1.0
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 10000
cfg['use_feature_target'] = True
cfg['extra_feature_steps'] = 0
cfg['extra_critic_steps'] = 2
cfg['target_update_period'] = 1
cfg['eval'] = False

cfg["experiment"]["directory"] = f"runs/torch/{task_name}/CTRL-SAC/{feature_hidden_dim}-{feature_dim}-{memory_size}"
cfg['alpha'] = 1e-3

agent = CTRLSACAgent(
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


