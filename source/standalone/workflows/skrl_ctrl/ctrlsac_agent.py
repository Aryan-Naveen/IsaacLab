from typing import Union, Tuple, Dict, Any, Optional

import gym, gymnasium
import copy

import torch
from torch import nn
import torch.nn.functional as F

from skrl.memories.torch import Memory
from skrl.models.torch import Model

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl import config, logger

class CTRLSACAgent(SAC):
    """
    SAC with VAE learned latent features
    """
    def __init__(
                self,
                models: Dict[str, Model],
                memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                device: Optional[Union[str, torch.device]] = None,
                cfg: Optional[dict] = None
                ) -> None:
        
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg
        )
        self.phi = self.models.get("phi", None)
        self.frozen_phi = self.models.get("frozen_phi", None)
        self.mu = self.models.get("mu", None)
        self.theta = self.models.get("theta", None)
        self.use_feature_target = cfg['use_feature_target']
        self.extra_feature_steps = cfg['extra_feature_steps']

        if self.use_feature_target:
            self.phi_target = copy.deepcopy(self.phi)
            self.frozen_phi_target = copy.deepcopy(self.frozen_phi)


        self.feature_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.theta.parameters()),
            weight_decay=cfg['weight_decay'], lr=cfg['feature_learning_rate'])

        self.checkpoint_modules["phi"] = self.phi
        self.checkpoint_modules["frozen_phi"] = self.frozen_phi
        self.checkpoint_modules["mu"] = self.mu
        self.checkpoint_modules["theta"] = self.theta
        
        self.target_update_period = cfg['target_update_period']

    def feature_step(self, sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones):
        z_phi, _, _ = self.phi({"states": sampled_states, "actions": sampled_actions}, role = "feature_phi")
        z_mu_next, _, _ = self.mu({"states": sampled_next_states}, role = "feature_mu")

        labels = torch.eye(sampled_states.shape[0]).to(self.device)

        # we take NCE gamma = 1 here, the paper uses 0.2
        contrastive = (z_phi[:, None, :] * z_mu_next[None, :, :]).sum(-1) 
        model_loss = nn.CrossEntropyLoss()
        model_loss = model_loss(contrastive, labels)
        

		# prob_loss = torch.mm(z_phi, z_mu_next.t()).mean(dim=1)
		# prob_loss = (z_phi * z_mu_next).sum(-1).clamp(min=1e-4)
		# prob_loss = prob_loss.log().square().mean()  
  
        r, _, _ = self.theta({"feature": z_phi}, role = "feature_theta")
        r_loss = 0.5 * F.mse_loss(r, sampled_rewards).mean()
        loss = model_loss + r_loss

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            'r_loss': r_loss.item(),
        }


    def update_feature_target(self):
        self.phi_target.update_parameters(self.phi, polyak=self._polyak)

    @property
    def entropy_coefficient(self):
        return self.log_entropy_coefficient.exp()

    def critic_step(self, sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones):
        """
        Critic update step
        """			
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")
        
            if self.use_feature_target:
                z_phi, _, _ = self.frozen_phi_target({"states": sampled_states, "actions": sampled_actions}, role="feature")
                z_phi_next, _, _ = self.frozen_phi_target({"states": sampled_next_states, "actions": next_actions}, role = "next_feature")
            else:
                z_phi, _, _ = self.frozen_phi({"states": sampled_states, "actions": sampled_actions}, role = "feature")
                z_phi_next, _, _ = self.frozen_phi({"states": sampled_next_states, "actions": next_actions}, role = "next_feature")

            next_q1, _, _ = self.target_critic_1({"z_phi": z_phi_next}, role="target_critic")
            next_q2, _, _ = self.target_critic_2({"z_phi": z_phi_next}, role="target_critic")
            next_q = torch.min(next_q1, next_q2) - self.entropy_coefficient * next_log_prob
            target_q = sampled_rewards + sampled_dones.logical_not() * self._discount_factor * next_q 
            
        q1, _, _ = self.critic_1({"z_phi": z_phi}, role="critic")
        q2, _, _ = self.critic_2({"z_phi": z_phi}, role="critic")
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = (q1_loss + q2_loss)/2

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(), 
            'q2_loss': q2_loss.item(),
            'q1': q1,
            'q2': q2,
            'q_loss': q_loss,
            'target_q': target_q
            }

    def actor_step(self, sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones):
        """
        Actor update step
        """	
        actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
        z_phi, _, _ = self.frozen_phi({"states": sampled_states, "actions": actions}, role="feature")
        q1, _, _ = self.critic_1({"z_phi": z_phi}, role="critic")
        q2, _, _ = self.critic_2({"z_phi": z_phi}, role="critic")
        q = torch.min(q1, q2)

        actor_loss = ((self.entropy_coefficient) * log_prob - q).mean()
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # entropy learning
        if self._learn_entropy:
            # compute entropy loss
            self.entropy_optimizer.zero_grad()
            entropy_loss = (self.entropy_coefficient * (-log_prob - self._target_entropy).detach()).mean()

            # optimization step (entropy)
            entropy_loss.backward()
            self.entropy_optimizer.step()
        
        

        
        return {
            'actor_loss': actor_loss,
            'entropy_loss': entropy_loss,
            'log_prob': log_prob.mean()
        }

    def update_target(self, steps):
        self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
        self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

    def update_learning_rate(self):
        if self._learning_rate_scheduler:
            self.policy_scheduler.step()
            self.critic_scheduler.step()

    def logging(self, policy_loss, critic_loss, feature_loss):
        self.track_data("Loss / Policy loss", policy_loss['actor_loss'].item())
        self.track_data("Loss / Critic loss", critic_loss['q_loss'].item())
        self.track_data("Loss / Feature loss", feature_loss['total_loss'])

        self.track_data("Q-network / Q1 (max)", torch.max(critic_loss['q1']).item())
        self.track_data("Q-network / Q1 (min)", torch.min(critic_loss['q1']).item())
        self.track_data("Q-network / Q1 (mean)", torch.mean(critic_loss['q1']).item())

        self.track_data("Q-network / Q2 (max)", torch.max(critic_loss['q2']).item())
        self.track_data("Q-network / Q2 (min)", torch.min(critic_loss['q2']).item())
        self.track_data("Q-network / Q2 (mean)", torch.mean(critic_loss['q2']).item())

        self.track_data("Target / Target (max)", torch.max(critic_loss['target_q']).item())
        self.track_data("Target / Target (min)", torch.min(critic_loss['target_q']).item())
        self.track_data("Target / Target (mean)", torch.mean(critic_loss['target_q']).item())



        if self._learn_entropy:
            self.track_data("Loss / Entropy loss", policy_loss['entropy_loss'].item())
            self.track_data("Log Prob", policy_loss['log_prob'].item())
            self.track_data("Coefficient / Entropy coefficient", self.entropy_coefficient.item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
            self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        #
        # if self.memory is not None:
        #     # reward shaping
        #     if self._rewards_shaper is not None:
        #         rewards = self._rewards_shaper(rewards, timestep, timesteps)
        #
        #     # storage transition in memory
        #     self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
        #                             terminated=terminated, truncated=truncated)
        #     for memory in self.secondary_memories:
        #         memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
        #                            terminated=terminated, truncated=truncated)
        self.cur_data = (states, actions, rewards, next_states, terminated, truncated)


    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        for _ in range(self.extra_feature_steps+1):
            # sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.cur_data
            feature_loss = self.feature_step(sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones)

            if self.use_feature_target:
                self.update_feature_target()
            
        self.frozen_phi.load_state_dict(self.phi.state_dict().copy())
        if self.use_feature_target:
            self.frozen_phi_target.load_state_dict(self.phi_target.state_dict().copy())

        critic_loss = self.critic_step(sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones)
        actor_loss = self.actor_step(sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones)

        self.update_target(timestep)

        self.update_learning_rate()

        self.logging(actor_loss, critic_loss, feature_loss)
        
