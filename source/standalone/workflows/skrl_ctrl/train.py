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
from sac.critic import Critic, TestCritic
from sac.feature import Phi, Mu, Theta

from ctrlsac_agent import CTRLSACAgent
from omni.isaac.lab.utils.dict import print_dict
import os
import gymnasium as gym
import numpy as np
import optuna

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


cli_args = ["--video"]
# load and wrap the Isaac Gym environment
env = load_isaaclab_env(task_name="Isaac-Quadcopter-Trajectory-Direct-v0", num_envs=1, cli_args=cli_args)

video_kwargs = {
    "video_folder": os.path.join("runs/torch/Quadcopter-Trajectory", "videos", "train"),
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
memory = RandomMemory(memory_size=int(1e6), num_envs=env.num_envs, device=device)

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
models["policy"] = DiagGaussianActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

models["critic_1"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            hidden_dim = feature_hidden_dim, 
                            device = device)

models["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            hidden_dim = feature_hidden_dim, 
                            device = device)

models["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   hidden_dim = feature_hidden_dim, 
                                   device = device)

models["target_critic_2"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   hidden_dim = feature_hidden_dim, 
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
BASELINE_CFG = SAC_DEFAULT_CONFIG.copy()
BASELINE_CFG["gradient_steps"] = 1
BASELINE_CFG["batch_size"] = 256
BASELINE_CFG["discount_factor"] = 0.99
BASELINE_CFG["polyak"] = 0.005
BASELINE_CFG["actor_learning_rate"] = 1e-4
BASELINE_CFG["critic_learning_rate"] = 1e-4
BASELINE_CFG["weight_decay"] = 0
BASELINE_CFG["feature_learning_rate"] = 1e-4
BASELINE_CFG["random_timesteps"] = 800
BASELINE_CFG["learning_starts"] = 800
BASELINE_CFG["grad_norm_clip"] = 0.0
BASELINE_CFG["learn_entropy"] = True
BASELINE_CFG["entropy_learning_rate"] = 1e-4
BASELINE_CFG["initial_entropy_value"] = 1.0
# logging to TensorBoard and write checkpoints (in timesteps)
BASELINE_CFG["experiment"]["write_interval"] = 800
BASELINE_CFG["experiment"]["checkpoint_interval"] = 8000
BASELINE_CFG["experiment"]["directory"] = "runs/torch/Quadcopter-Trajectory"
BASELINE_CFG['use_feature_target'] = True
BASELINE_CFG['extra_feature_steps'] = 2
BASELINE_CFG['target_update_period'] = 1

def objective(trial: Trial):
    # Suggest hyperparameters to optimize
    feature_learning_rate = trial.suggest_loguniform("feature_learning_rate", 1e-6, 1e-3)
    critic_learning_rate = trial.suggest_loguniform("critic_learning_rate", 1e-6, 1e-3)
    actor_learning_rate = trial.suggest_loguniform("actor_learning_rate", 1e-6, 1e-3)
    extra_feature_steps = trial.suggest_int("extra_feature_steps", 1, 5)
    
    # Update the agent's configuration with the suggested hyperparameters
    cfg = BASELINE_CFG.copy()
    cfg["feature_learning_rate"] = feature_learning_rate
    cfg["critic_learning_rate"] = critic_learning_rate
    cfg["actor_learning_rate"] = actor_learning_rate
    cfg["extra_feature_steps"] = extra_feature_steps

    # Initialize the agent and trainer
    agent = CTRLSACAgent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    cfg_trainer = {"timesteps": int(1e5), "headless": True}
    trainer = StepTrainer(env=env, agents=agent, cfg=cfg_trainer)

    # Training loop with intermediate pruning
    cumulative_reward = 0
    for timestep in range(cfg_trainer["timesteps"]):
        _, rewards, _, _, _ = trainer.train(timestep=timestep)
        cumulative_reward += rewards.mean().item()

        # Report the intermediate results to Optuna
        trial.report(cumulative_reward / (timestep + 1), step=timestep)

        # Check if the trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return the final performance metric (e.g., average cumulative reward)
    return cumulative_reward / cfg_trainer["timesteps"]

study = optuna.create_study(direction="maximize", pruner=MedianPruner())
study.optimize(objective, n_trials=50, timeout=4500)  # 50 trials, 1-hour timeout

# Print the best hyperparameters
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

# Save study for later analysis
study_name = "sac_hyperparam_optimization"
study_storage = f"sqlite:///{study_name}.db"
optuna.study.create_study(study_name=study_name, storage=study_storage, direction="maximize")
