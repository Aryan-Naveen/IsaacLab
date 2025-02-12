import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from utils.utils import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer, StepTrainer
from skrl.utils import set_seed

from networks.actor import DiagGaussianActor, StochasticActor, DiagGaussianActorPolicy
from networks.critic import Critic, SACCritic
from networks.feature import Phi, Mu, Theta

from agents.ctrlsac_agent import CTRLSACAgent
from omni.isaac.lab.utils.dict import print_dict
import os
import gymnasium as gym
# import gym
import numpy as np
import argparse
from omni.isaac.lab.app import AppLauncher

# seed for reproducibility


parser = argparse.ArgumentParser(description="Run the eval script with customizable parameters.")
# Add arguments
parser.add_argument("--env_version", type=str, default="legtrain-active-bo")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse arguments
args, _ = parser.parse_known_args()

cli_args = ["--video"]
# load and wrap the Isaac Gym environment
task_version = args.env_version
finetune = task_version == 'legtrain-finetune'
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"

num_envs = 32 if not finetune else 1
env = load_isaaclab_env(task_name = task_name, num_envs=num_envs, cli_args=cli_args)

# using multitask parameterization
multitask = True

video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task_version}/", "videos", "train", "CTRLSAC"),
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

# define task feature dimensions
cdim = 512

# state dimensions
task_state_dim = 67
drone_state_dim = 13



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
                            task_state_dim = task_state_dim,
                            cdim = cdim,
                            multitask = multitask,
                            device = device)

models["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            task_state_dim = task_state_dim,
                            cdim = cdim,
                            multitask = multitask,
                            device = device)

models["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   task_state_dim = task_state_dim,
                                   cdim = cdim,
                                   multitask = multitask,
                                   device = device)

models["target_critic_2"] = Critic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                task_state_dim = task_state_dim,
                                cdim = cdim,
                                multitask = multitask,
                                device = device)


models["phi"] = Phi(observation_space = env.observation_space, 
				    action_space = env.action_space, 
				    feature_dim = feature_dim, 
				    hidden_dim = feature_hidden_dim,
                    drone_state_dim = drone_state_dim,
                    multitask = multitask,
                    device = device
                )

models["frozen_phi"] = Phi(observation_space = env.observation_space, 
    				       action_space = env.action_space, 
	    			       feature_dim = feature_dim, 
		    	           hidden_dim = feature_hidden_dim,
                           drone_state_dim = drone_state_dim,
                           multitask = multitask,
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
                drone_state_dim = drone_state_dim,
                multitask = multitask,
                device = device
            )

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4 if not finetune else 3e-4
cfg["critic_learning_rate"] = 1e-4 if not finetune else 3e-4
cfg["weight_decay"] = 0
cfg["feature_learning_rate"] = 1e-4 if not finetune else 1e-5
cfg["random_timesteps"] = 12e3 if not finetune else 0
cfg["learning_starts"] = 12e3 if not finetune else 25e3
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True if not finetune else False
cfg["entropy_learning_rate"] = 1e-5
cfg["initial_entropy_value"] = 1.0 if not finetune else 0.06
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 10000 if not finetune else 1000
cfg['use_feature_target'] = True
cfg['extra_feature_steps'] = 0
cfg['extra_critic_steps'] = 2
cfg['target_update_period'] = 1
cfg['eval'] = False

cfg["experiment"]["directory"] = f"runs/torch/{task_name}/CTRL-SAC-{multitask}/"
cfg['alpha'] = 1e-3

cfg['memory'] = None if not finetune else None

agent = CTRLSACAgent(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

timesteps = int(3e5) if not finetune else int(5e4)
if finetune:
    agent.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/CTRL-SAC/25-02-01_23-07-59-960690_CTRLSACAgent/checkpoints/agent_300000.pt")

cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
memory.save(cfg["experiment"]["directory"], format = 'pt')
# train the agent(s)
trainer.train()


