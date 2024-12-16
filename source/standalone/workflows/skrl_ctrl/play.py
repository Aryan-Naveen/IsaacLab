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

from networks.actor import DiagGaussianActor, StochasticActor
from networks.critic import Critic
from networks.feature import Phi, Mu, Theta

from agents.ctrlsac_agent import CTRLSACAgent
from omni.isaac.lab.utils.dict import print_dict
import os
import gymnasium as gym
# import gym
import numpy as np

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


cli_args = ["--video"]
# load and wrap the Isaac Gym environment
task = "OOD"
multitask = False
env = load_isaaclab_env(task_name=f"Isaac-Quadcopter-{task}-Trajectory-Direct-v0", num_envs=1, cli_args=cli_args)


video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task}/{str(multitask)}", "videos", "eval"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": 1000,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device


# instantiate a memory as experience replay
experiment_length = int(300)
memory = RandomMemory(memory_size=experiment_length, num_envs=env.num_envs, device=device)

# define hidden dimension
actor_hidden_dim = 512
actor_hidden_depth = 2

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
                            device = device, 
                            multitask = multitask
                            )

models["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device, 
                            multitask = multitask
                            )

models["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   device = device, 
                                   multitask = multitask
                                   )

models["target_critic_2"] = Critic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                device = device, 
                                multitask = multitask
                                )


models["phi"] = Phi(observation_space = env.observation_space, 
				    action_space = env.action_space, 
				    feature_dim = feature_dim, 
				    hidden_dim = feature_hidden_dim,
                    device = device,
                    multitask = multitask
                )

models["frozen_phi"] = Phi(observation_space = env.observation_space, 
    				       action_space = env.action_space, 
	    			       feature_dim = feature_dim, 
		    	           hidden_dim = feature_hidden_dim,
                           device = device,
                           multitask = multitask
                        )

models["theta"] = Theta(
    		        observation_space = env.observation_space,
		            action_space = env.action_space, 
		            feature_dim = feature_dim, 
                    device = device,
                    multitask = multitask
                )

models["mu"] = Mu(
                observation_space = env.observation_space, 
                action_space = env.action_space, 
                feature_dim = feature_dim, 
                hidden_dim = feature_hidden_dim,
                device = device,
                multitask = multitask
            )

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
cfg["feature_learning_rate"] = 5e-5
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 25e3
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000000000000000
cfg["experiment"]["checkpoint_interval"] = 10000000000000
# cfg["experiment"]["directory"] = "runs/torch/QuadCopter-CTRL"
cfg['use_feature_target'] = False
cfg['extra_feature_steps'] = 1
cfg['target_update_period'] = 1
cfg['eval'] = True
cfg['alpha'] = None


agent = CTRLSACAgent(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

## Linear CTRLSAC: "/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-Linear-Trajectory-Direct-v0/CTRL-SAC/1024-512-100000/24-12-10_19-59-27-142587_CTRLSACAgent/checkpoints/best_agent.pt"
## Multi CTRLSAC: "/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-Multi-Trajectory-Direct-v0/CTRL-SAC/1024-512-100000/24-12-11_18-47-55-198869_CTRLSACAgent/checkpoints/best_agent.pt"
agent.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-Multi-Trajectory-Direct-v0/CTRL-SAC/1024-512-100000/24-12-15_19-57-01-146307_CTRLSACAgent/checkpoints/best_agent.pt")
env.eval_mode()
cfg_trainer = {"timesteps": experiment_length, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.eval()

distance = agent.memory.tensors_view['states'][:, 13:16]

folder = f"/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/{task}/"
# print(np.savetxt(f"{folder}ctrl_agent_positions.txt", np.array(env.positions)))
# print(np.savetxt(f"{folder}trajectory.txt", env._desired_trajectory_w[0, :].cpu().numpy()))

print(np.abs(distance.cpu()).mean(axis=0))