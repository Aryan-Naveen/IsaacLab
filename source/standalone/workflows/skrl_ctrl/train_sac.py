import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from typing import Any, Mapping, Tuple, Union, cast

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from omni.isaac.lab.utils.dict import print_dict
from networks.actor import DiagGaussianActor, StochasticActor, DiagGaussianActorPolicy
import gymnasium as gym
# import gym
import numpy as np

from networks.controller import PIDController
from memories.trajectorymemory import TrajectoryRandomMemory
# from agents.sac_agent import SACMod as SAC


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
# define hidden dimension
actor_hidden_dim = 512
actor_hidden_depth = 3

# load and wrap the Isaac Lab environment
cli_args = ["--video", "--enable_cameras"]  # enable video recording and camera rendering
# load and wrap the Isaac Gym environment
task_version = "legtrain"
finetune = task_version == 'legtrain-finetune'
task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"
num_envs = 32 if not finetune else 1
env = load_isaaclab_env(task_name=task_name, num_envs=num_envs, cli_args=cli_args)
traj_len = 16

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)

class MLP(nn.Module):
	def __init__(self,
								input_dim,
								hidden_dim,
								output_dim,
								hidden_depth,
								output_mod=None):
		super().__init__()
		self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
											output_mod)
		self.apply(weight_init)

	def forward(self, x):
		return self.trunk(x)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, hidden_activation=nn.ELU(inplace=True), output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), hidden_activation] # inplace=True
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), hidden_activation] # inplace=True
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk



# class StochasticActor(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False,
#                  clip_log_std=True, min_log_std=-5, max_log_std=2):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

#         # Main Actor Network
#         self.net = nn.Sequential(
#             nn.Linear(self.num_observations, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.num_actions),
#             nn.Tanh()
#         )
        
#         # Learnable Log Std Parameter
#         self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -2.0), requires_grad=False)

#     def compute(self, inputs, role=""):
#         """
#         Handles both dictionary inputs (offline/buffers) and tensor inputs (online/inference).
#         """
#         # 1. Standardize Input
#         # If input is a dict (Offline/Trainer), extract "states".
#         # If input is a Tensor (Online/Env), use it directly.
#         if isinstance(inputs, dict):
#             x = inputs["states"]
#         elif isinstance(inputs, torch.Tensor):
#             x = inputs
#         else:
#             raise ValueError(f"Unsupported input type: {type(inputs)}")

#         # 2. Ensure Device Compatibility
#         # This prevents errors if external scripts pass CPU tensors to a GPU model
#         if x.device != self.device:
#             x = x.to(self.device)

#         # 3. Forward Pass
#         # Returns mean (mu), log_std, and an empty dict for extra outputs (like values)
#         return self.net(x), self.log_std_parameter, {}



    
# define models (stochastic and deterministic models) using mixins
class StochasticMellingerActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Ensure mixin attributes exist (skrl version may use different names)
        self._clip_log_std = clip_log_std
        self._log_std_min = min_log_std
        self._log_std_max = max_log_std
        self._clip_actions = clip_actions
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        self._clip_actions_min = torch.tensor(low, device=device, dtype=torch.float32) if low is not None else torch.tensor(-1.0, device=device, dtype=torch.float32)
        self._clip_actions_max = torch.tensor(high, device=device, dtype=torch.float32) if high is not None else torch.tensor(1.0, device=device, dtype=torch.float32)

        self.integral_horizon = traj_len  # Use buffer mode with specified horizon

        self.net = PIDController(num_envs, device, integral_horizon=traj_len)
        # log_std in task-encoder space (9 dims). Use 0.0 so scale=1 and reparameterization
        # gradient is non-zero (with -4 or -5, scale~0.01 and grad can underflow to 0).
        self.log_std_parameter = nn.Parameter(torch.full((9,), 0.0, device=device))

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # map from states to mean encoding (9-dim) and log_std over encoding
        mean_encoding, log_std_enc, outputs = self.compute(inputs, role)

        # clamp log standard deviations (in encoding space)
        if self._clip_log_std:
            log_std_enc = torch.clamp(log_std_enc, self._log_std_min, self._log_std_max)

        self._log_std = log_std_enc
        self._num_samples = mean_encoding.shape[0]

        # distribution over task-encoder output (9-dim)
        scale = log_std_enc.exp()
        self._distribution = Normal(mean_encoding, scale)

        # sample encoding using reparameterization trick
        encoding = self._distribution.rsample()

        # deterministic action from PID given sampled encoding
        states = inputs["states"] if isinstance(inputs, dict) else inputs
        states = cast(torch.Tensor, states)
        if states.device != self.device:
            states = states.to(self.device)
        actions = self.net(states, encoding=encoding)

        # clip actions
        # MIGHT BE CAUSE OF 0 GRADIENT
        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        # log prob of the sampled encoding: use explicit formula so gradient to mean_encoding is
        # guaranteed (d/d(loc) log N(x;loc,scale) = (loc-x)/scale^2). PyTorch Normal.log_prob
        # can fail to backprop to loc in some versions when loc is the output of another module.
        log_prob = (
            -0.5 * ((encoding - mean_encoding) / scale).pow(2)
            - log_std_enc
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # mean action = PID with mean encoding (no noise)
        with torch.no_grad():
            mean_actions = self.net(states, encoding=mean_encoding)
        outputs["mean_actions"] = mean_actions
        outputs["mean_encoding"] = mean_encoding  # for auxiliary loss so task_encoder gets gradient
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return the log standard deviation of the model

        :return: Log standard deviation of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> log_std = model.get_log_std()
            >>> print(log_std.shape)
            torch.Size([4096, 8])
        """
        return self._log_std.repeat(self._num_samples, 1)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Normal
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Normal(loc: torch.Size([4096, 8]), scale: torch.Size([4096, 8]))
        """
        return self._distribution


    def compute(self, inputs, role):
        states = inputs["states"] if isinstance(inputs, dict) else inputs
        if states.device != self.device:
            states = states.to(self.device)
        # task_states: 90 dims at 10:-1 (obs len 101) or 10:-6 (obs len 106)
        obs_dim = states.shape[-1]
        if obs_dim == 101:
            task_states = states[:, 10:-1]
        else:
            task_states = states[:, 10:-6]
        mean_encoding = self.net.task_encoder(task_states)  # (B, 9)
        # broadcast log_std to batch for consistent interface
        log_std = self.log_std_parameter.unsqueeze(0).expand(
            mean_encoding.shape[0], -1
        )
        return mean_encoding, log_std, {}

    def pre_process_offline(self, inputs):
        """
        Calculates accumulated integral errors without the overhead of full control physics.
        Returns the final observation concatenated with the calculated integral states.
        """
        return self.net.preprocess_obs_offline(inputs)

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
        states = inputs["states"]
        actions = inputs["taken_actions"]

        cat_inputs = torch.cat([states, actions], dim=1)
        
        # Pass through MLP
        q_values = self.net(cat_inputs)

        return q_values, {}


class DSLPIDObservationPreprocessor(nn.Module):
    def __init__(self, device: str | torch.device, controller):
        """
        Args:
            device: The device (cpu/cuda) - passed automatically if in kwargs or handled manually
            controller: Your initialized DSLPIDControlTraining instance
        """
        super().__init__()
        self.device = device
        self.controller = controller

    def forward(self, obs: torch.Tensor, train: bool = False) -> torch.Tensor:
        """
        SKRL calls this method.
        obs shape: (Batch_Size, Observation_Dim)
        """
        # SKRL usually passes 2D tensors (Batch, Dim).
        # Your pre_process_offline expects 3D tensors (Batch, Time, Dim).
        # We unsqueeze to add a time dimension of 1.
        if obs.dim() == 2:
            obs_expanded = obs.unsqueeze(1)
        else:
            obs_expanded = obs

        # Call your custom method
        # This returns (Batch, State_Dim + 6)
        return self.controller.pre_process_offline(obs_expanded)

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
# memory = TrajectoryRandomMemory(memory_size=int(7e5), num_envs=env.num_envs, traj_len=traj_len, device=device)
memory = RandomMemory(memory_size=int(1e5), num_envs=env.num_envs, device=device)


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
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Wire environment scaling parameters into the policy's differentiable controller
# so it can convert its physical outputs into the env's normalized action space.
try:
    unwrapped = getattr(env, "unwrapped", env)
    thrust_to_weight = getattr(getattr(unwrapped, "cfg", None), "thrust_to_weight", None)
    robot_weight = getattr(unwrapped, "_robot_weight", None)
    moment_scale = getattr(getattr(unwrapped, "cfg", None), "moment_scale", None)
    if thrust_to_weight is not None and robot_weight is not None and moment_scale is not None:
        try:
            models["policy"].net.set_env_params(thrust_to_weight=1.9, robot_weight=0.0282, moment_scale=0.01)
            print(f"[INFO] Set env params on policy.net: thrust_to_weight={thrust_to_weight}, robot_weight={robot_weight}, moment_scale={moment_scale}")
            # Quick sanity check: estimate the normalized action that corresponds to gravity/hover
            try:
                gravity_z = float(models["policy"].net.gravity[2].to(models["policy"].net.MIXER_MATRIX.device))
                denom = float(models["policy"].net.thrust_to_weight * models["policy"].net.robot_weight)
                if denom != 0:
                    hover_action0 = 2.0 * (gravity_z / denom) - 1.0
                    print(f"[INFO] Estimated hover action0 (before clamp) = {hover_action0:.4f} (should be in [-1,1] for nominal values)")
                else:
                    print("[WARN] denom is zero when computing hover action; skipping hover estimate")
            except Exception as e:
                print("[WARN] Failed to compute hover-action sanity check:", e)
        except Exception as e:
            print("[WARN] Failed to set env params on policy.net:", e)
    else:
        print("[WARN] Env scaling params not found; policy will return physical outputs unless handled elsewhere.")
except Exception as e:
    print("[WARN] Error while wiring env params to policy.net:", e)

bs = 256
# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
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
# logging to TensorBoard and write checkpoints (in timesteps)
# logging to wandb and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000  # task_encoder stats are logged at this interval too
cfg["experiment"]["checkpoint_interval"] = 10000 if not finetune else 1000
cfg["experiment"]["directory"] = f"runs/torch/{task_name}/SAC"
cfg["experiment"]["experiment_name"] = f"{task_name}-SAC"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {
    "project": "CDC",
    "name": f"SAC-baseline",
    "config": {
        "task": task_name,
        "num_envs": num_envs,
        "batch_size": bs,
    }
}


agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
timesteps = int(5e5) if not finetune else int(25e4)

cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
memory.save(cfg["experiment"]["directory"], format = 'pt')

# start training
trainer.train()
