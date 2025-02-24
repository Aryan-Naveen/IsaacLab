import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F

from skrl.models.torch import Model, DeterministicMixin

class Phi(DeterministicMixin, Model):
    """
    phi: s, a -> z_phi in R^d
    """
    def __init__(
            self, 
            observation_space, 
            action_space, 
            feature_dim, 
            hidden_dim, 
            drone_state_dim,
            device,
            multitask = False
        ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.multitask = multitask
        state_dim = observation_space.shape[0] if not self.multitask else drone_state_dim
        action_dim = action_space.shape[0]
        
        self.feature_dim = feature_dim
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)


    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["actions"]], axis=-1) if not self.multitask else torch.cat([inputs["drone_states"], inputs["actions"]], axis=-1)
        z = F.elu(self.l1(x)) 
        z = F.elu(self.l2(z)) 
        z_phi = self.layer_norm(self.l3(z))
        return  z_phi, {}

class Mu(DeterministicMixin, Model):
    """
    mu': s' -> z_mu in R^d
    """
    def __init__(
        self, 
        observation_space,
        action_space,
        feature_dim,
        hidden_dim,
        drone_state_dim,
        device,
        multitask = False
        ):

        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)


        self.multitask = multitask
        state_dim = observation_space.shape[0] if not self.multitask else drone_state_dim
        self.feature_dim = feature_dim

        self.l1 = nn.Linear(state_dim , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)


    def compute(self, inputs, role):
        z = F.elu(self.l1(inputs['states'])) if not self.multitask else F.elu(self.l1(inputs['drone_states']))
        z = F.elu(self.l2(z)) 
        z_mu = F.tanh(self.l3(z)) 

        return  z_mu, {}


class Theta(DeterministicMixin, Model):
    """
    Linear theta 
    <phi(s, a), theta> = r 
    """
    def __init__(
        self, 
        observation_space,
        action_space,
        feature_dim,
        device,
        multitask = False
        ):

        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)


        self.l = nn.Linear(feature_dim, 1)

    def compute(self, inputs, role):
        r = self.l(inputs["feature"])
        return r, {}
