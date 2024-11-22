from torch import nn
import torch.nn.functional as F
from skrl.models.torch import Model, DeterministicMixin
import torch


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, feature_dim, hidden_dim, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.feature_dim = feature_dim

        self.qnet = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim, 1))

    def compute(self, inputs, role):
        q1 = self.qnet(inputs['z_phi'])
        return q1, {}


class TestCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}
