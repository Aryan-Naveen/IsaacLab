from torch import nn
import torch.nn.functional as F
from skrl.models.torch import Model, DeterministicMixin
import torch


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, feature_dim, task_state_dim, cdim, device, multitask=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.feature_dim = feature_dim
        self.multitask = multitask
        

        if multitask:
            self.net = nn.Sequential(nn.Linear(2*feature_dim, 1024),
                            nn.ELU(),
                            nn.Linear(1024, 1))

            self.w = nn.Sequential(nn.Linear(task_state_dim, 512),
                                 nn.ELU(),
                                 nn.Linear(512, feature_dim),
                                 nn.Sigmoid())

        else:
            self.net = nn.Sequential(nn.Linear(feature_dim, 1024),
                            nn.ELU(),
                            nn.Linear(1024, 1))
            
                                

    def compute_multitask(self, inputs):
        task_weight = self.w(inputs['task_states'])
        # q = torch.einsum("ij,ij->i", inputs['z_phi'], task_weight).unsqueeze(1)
        q = self.net(torch.cat([inputs['z_phi'], task_weight], dim= 1))
        return q
    
    
    
    def compute_standard(self, inputs):
        return self.net(inputs['z_phi'])

    def compute(self, inputs, role):
        q = self.compute_multitask(inputs) if self.multitask else self.compute_standard(inputs)
        return q, {}

        

class SACCritic(DeterministicMixin, Model):
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
