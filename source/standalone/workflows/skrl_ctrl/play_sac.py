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
from networks.critic import Critic, TestCritic
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
task = "Linear"
env = load_isaaclab_env(task_name=f"Isaac-Quadcopter-{task}-Trajectory-Direct-v0", num_envs=1, cli_args=cli_args)


video_kwargs = {
    "video_folder": os.path.join(f"runs/torch/{task}", "videos", "eval"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": 1000,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device

experiment_length = int(300)
# instantiate a memory as experience replay
memory = RandomMemory(memory_size=experiment_length, num_envs=env.num_envs, device=device)

# define hidden dimension
actor_hidden_dim = 256
actor_hidden_depth = 2

# define feature dimension 
feature_dim = 512
feature_hidden_dim = 256

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

models["critic_1"] = TestCritic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

models["critic_2"] = TestCritic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

models["target_critic_1"] = TestCritic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   device = device)

models["target_critic_2"] = TestCritic(observation_space = env.observation_space,
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
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100000000
cfg["experiment"]["checkpoint_interval"] = 80000000000000
# cfg["experiment"]["directory"] = "runs/torch/QuadCopter-CTRL"
cfg['use_feature_target'] = False
cfg['extra_feature_steps'] = 0
cfg['target_update_period'] = 1



agent = SAC(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

## Linear SAC: "/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-Linear-Trajectory-Direct-v0/SAC/24-12-11_14-02-12-408306_SAC/checkpoints/best_agent.pt"
## Multi SAC: "/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-Multi-Trajectory-Direct-v0/SAC/24-12-11_16-28-19-830592_SAC/checkpoints/best_agent.pt"
agent.load("/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-Linear-Trajectory-Direct-v0/SAC/24-12-11_14-02-12-408306_SAC/checkpoints/best_agent.pt")

cfg_trainer = {"timesteps": experiment_length, "headless": True}
env.eval_mode()
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.eval()

distance = agent.memory.tensors_view['states'][:, 13:16]

folder = f"/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/{task}/"
print(np.savetxt(f"{folder}sac_agent_positions.txt", np.array(env.positions)))
print(np.savetxt(f"{folder}trajectory.txt", env._desired_trajectory_w[0, :].cpu().numpy()))
print(np.abs(distance.cpu()).mean(axis=0))