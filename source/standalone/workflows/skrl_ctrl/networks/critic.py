from torch import nn
import torch.nn.functional as F
from skrl.models.torch import Model, DeterministicMixin
import torch


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, feature_dim, device, multitask=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.feature_dim = feature_dim
        
        self.net = nn.Sequential(nn.Linear(feature_dim, 1024),
                                 nn.ELU(),
                                 nn.Linear(1024, 1))

        if multitask:
            self.cnet = nn.Sequential(nn.Linear(taskstate_dim, 512),
                                 nn.ELU(),
                                 nn.Linear(512, cdims),
                                 nn.Sigmoid())
            
                                

    def compute_multitask(self, inputs, role):
        task_feature = self.cnet(inputs['z_phi'])

    def compute(self, inputs, role):
        q = self.net(inputs['z_phi'])
        return q, {}


class TestCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, feature_dim, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}
