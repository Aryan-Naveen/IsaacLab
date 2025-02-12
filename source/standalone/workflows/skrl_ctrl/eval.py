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

from networks.actor import DiagGaussianActor, StochasticActor
from networks.critic import Critic, SACCritic
from networks.feature import Phi, Mu, Theta

from agents.ctrlsac_agent import CTRLSACAgent
from omni.isaac.lab.utils.dict import print_dict
import os
import gymnasium as gym
import numpy as np

import argparse
from omni.isaac.lab.app import AppLauncher
from pathlib import Path


experiments = {
    "legeval": [515, False, 100],
    "legeval-predef": [515, False, 1],
    "legood": [500, True, 1],
    "OOD": [500, True, 1],
    "legtrain": [3000, True, 1]
}


parser = argparse.ArgumentParser(description="Run the eval script with customizable parameters.")
# Add arguments
parser.add_argument("--experiment", type=str, default="legeval-predef", choices=experiments.keys(), help="Specify the task name (default: OOD).")
parser.add_argument("--agent_type", type=str, default="CTRLSAC-multi", choices=["CTRLSAC-multi", "SAC", "CTRLSAC"], help="Specify the agent type (default: CTRLSAC).")
parser.add_argument("--ckpt", type=str, default="-1", help="Specify the checkpoint name (default: best_agent).")
parser.add_argument("--folder", type=str)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse arguments
args, _ = parser.parse_known_args()
task = args.experiment
agent_type = args.agent_type
ckpt = args.ckpt
folder = args.folder
# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


cli_args = ["--video"]



experiment_length = experiments[task][0]
record_video = experiments[task][1] 
output_dir = f"runs/experiments/{task}/{agent_type}/{folder.split('/')[-4]}"
num_envs = experiments[task][2] 
video_kwargs = {
    "video_folder": os.path.join(output_dir, "videos"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": experiment_length,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = load_isaaclab_env(task_name=f"Isaac-Quadcopter-{task}-Trajectory-Direct-v0", num_envs=num_envs, cli_args=cli_args)

if record_video: env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=experiment_length, num_envs=env.num_envs, device=device)

# define hidden dimension
actor_hidden_dim = 256
actor_hidden_depth = 3

# define feature dimension 
feature_dim = 512
feature_hidden_dim = 1024

# define task feature dimensions
cdim = 512

# state dimensions
task_state_dim = 67
drone_state_dim = 13
multitask = agent_type == 'CTRLSAC-multi'
if agent_type == 'CTRLSAC-multi':
    agent_type = 'CTRLSAC'

agentclasses = {
    "CTRLSAC": CTRLSACAgent,
    "SAC": SAC
}



# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["SAC"] = {}
models["SAC"]["policy"] = StochasticActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

models["SAC"]["critic_1"] = SACCritic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

models["SAC"]["critic_2"] = SACCritic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

models["SAC"]["target_critic_1"] = SACCritic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   device = device)

models["SAC"]["target_critic_2"] = SACCritic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                device = device)

models["CTRLSAC"] = {}
models["CTRLSAC"]["policy"] = StochasticActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

models["CTRLSAC"]["critic_1"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            task_state_dim = task_state_dim,
                            cdim = cdim,
                            multitask = multitask,
                            device = device)

models["CTRLSAC"]["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            task_state_dim = task_state_dim,
                            cdim = cdim,
                            multitask = multitask,
                            device = device)

models["CTRLSAC"]["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   task_state_dim = task_state_dim,
                                   cdim = cdim,
                                   multitask = multitask,
                                   device = device)

models["CTRLSAC"]["target_critic_2"] = Critic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                task_state_dim = task_state_dim,
                                cdim = cdim,
                                multitask = multitask,
                                device = device)


models["CTRLSAC"]["phi"] = Phi(observation_space = env.observation_space, 
				    action_space = env.action_space, 
				    feature_dim = feature_dim, 
				    hidden_dim = feature_hidden_dim,
                    drone_state_dim = drone_state_dim,
                    multitask = multitask,
                    device = device
                )

models["CTRLSAC"]["frozen_phi"] = Phi(observation_space = env.observation_space, 
    				       action_space = env.action_space, 
	    			       feature_dim = feature_dim, 
		    	           hidden_dim = feature_hidden_dim,
                           drone_state_dim = drone_state_dim,
                           multitask = multitask,
                           device = device
                        )

models["CTRLSAC"]["theta"] = Theta(
    		        observation_space = env.observation_space,
		            action_space = env.action_space, 
		            feature_dim = feature_dim, 
                    device = device
                )

models["CTRLSAC"]["mu"] = Mu(
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
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["weight_decay"] = 0
cfg["feature_learning_rate"] = 5e-5
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1e6
cfg["experiment"]["write_interval"] = 0
cfg["experiment"]["checkpoint_interval"] = 0
cfg['use_feature_target'] = False
cfg['extra_feature_steps'] = 1
cfg['target_update_period'] = 1
cfg['eval'] = True
cfg['alpha'] = None




AgentClass = agentclasses[agent_type]

agent = AgentClass(
    models=models[agent_type],
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device
)



agent.load(f"{folder}/{ckpt}")
cfg_trainer = {"timesteps": experiment_length, "headless": True}
env.eval_mode()
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.eval()
os.makedirs(output_dir, exist_ok=True)
torch.save(env.results, f"{output_dir}/{ckpt[:-3]}.pth")

