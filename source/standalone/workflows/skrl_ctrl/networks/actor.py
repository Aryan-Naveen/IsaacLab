"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""


import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from typing import Any, Mapping, Tuple, Union

from torch.distributions import Normal

from skrl.models.torch import Model, GaussianMixin


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk

class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
        mu = tr(mu)
    return mu

class DiagGaussianActor(GaussianMixin, Model):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, observation_space, action_space, hidden_dim, hidden_depth,
                log_std_bounds, device):
    Model.__init__(self, observation_space, action_space, device)
    GaussianMixin.__init__(self, min_log_std=log_std_bounds[0], max_log_std=log_std_bounds[1], reduction="mean")

    self.log_std_bounds = log_std_bounds
    # self.trunk = mlp(observation_space.shape[0], hidden_dim, 2 * action_space.shape[0],
    #                         hidden_depth)
    self.trunk = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2 * self.num_actions))
    self.apply(weight_init)
    
    
  def compute(self, inputs, role):
    mu, log_std = self.trunk(inputs["states"]).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                  1)
    
    
    return mu, log_std, {}
  

  def act(self,
          inputs: Mapping[str, Union[torch.Tensor, Any]],
          role: str = "", explore = True) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
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
      # map from states/observations to mean actions and log standard deviations
      mean_actions, log_std, outputs = self.compute(inputs, role)


      self._log_std = log_std
      self._num_samples = mean_actions.shape[0]

      # distribution
      self._distribution = SquashedNormal(mean_actions, log_std.exp())

      # sample using the reparameterization trick
      actions = self._distribution.rsample()
      
      # log of the probability density function
      log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions))
      if self._reduction is not None:
          log_prob = self._reduction(log_prob, dim=-1)
      if log_prob.dim() != actions.dim():
          log_prob = log_prob.unsqueeze(-1)

      outputs["mean_actions"] = self._distribution.mean
      return actions, log_prob, outputs
    
    
# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space,hidden_dim, hidden_depth,
                log_std_bounds, device, clip_actions=False,
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
      





class TanhTransform2(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform2)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal2(pyd.transformed_distribution.TransformedDistribution):
  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform2()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
        mu = tr(mu)
    return mu


class DiagGaussianActor2(nn.Module):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                log_std_bounds):
    super().__init__()

    self.log_std_bounds = log_std_bounds
    self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim,
                            hidden_depth)

    self.outputs = dict()
    self.apply(weight_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                  1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    dist = SquashedNormal2(mu, std)
    return dist


class DiagGaussianActorPolicy(GaussianMixin, Model):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, observation_space, action_space, hidden_dim, hidden_depth,
                log_std_bounds, device):
    Model.__init__(self, observation_space, action_space, device)
    GaussianMixin.__init__(self, min_log_std=log_std_bounds[0], max_log_std=log_std_bounds[1], reduction="mean")

    # self.log_std_bounds = log_std_bounds
    # # self.trunk = mlp(observation_space.shape[0], hidden_dim, 2 * action_space.shape[0],
    # #                         hidden_depth)
    # self.trunk = nn.Sequential(nn.Linear(self.num_observations, 1024),
    #                              nn.ReLU(),
    #                              nn.Linear(1024, 512),
    #                              nn.ReLU(),
    #                              nn.Linear(512, 2 * self.num_actions))
    # self.apply(weight_init)
    
    self.actor = DiagGaussianActor2(obs_dim = observation_space.shape[0],
                                    action_dim = action_space.shape[0],
                                    hidden_dim = hidden_dim,
                                    hidden_depth = hidden_depth,
                                    log_std_bounds = log_std_bounds)
    
    
  def compute(self, inputs, role):
    # mu, log_std = self.trunk(inputs["states"]).chunk(2, dim=-1)

    # # constrain log_std inside [log_std_min, log_std_max]
    # log_std = torch.tanh(log_std)
    # log_std_min, log_std_max = self.log_std_bounds
    # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
    #                                                               1)
    
    dist = self.actor(inputs['states'])
    # sample using the reparameterization trick
    actions = dist.rsample()
    
    # log of the probability density function
    log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
    
    return actions, log_prob, {}
