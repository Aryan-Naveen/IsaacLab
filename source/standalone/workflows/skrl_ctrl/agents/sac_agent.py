from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch.sac import SAC
from skrl.memories.torch import Memory
from skrl.models.torch import Model



class SACMod(SAC):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: dict = {},
    ) -> None:
        """Soft Actor-Critic (SAC).

        https://arxiv.org/abs/1801.01290

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

    def _update(self, timestep: int, timesteps: int) -> None:

        for gradient_step in range(self._gradient_steps):

            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_terminated, sampled_truncated = \
            self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.no_grad():
                processed_states = self.policy.pre_process_offline(sampled_states)
                processed_next_states = self.policy.pre_process_offline(sampled_next_states)

            inputs = {"states": processed_states}
            next_inputs = {"states": processed_next_states}

            # ---------------------- Critic target ----------------------
            with torch.no_grad():
                next_actions, next_log_prob, _ = self.policy.act(next_inputs, role="policy")

                target_q1_values, _, _ = self.target_critic_1.act(
                    {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                )
                target_q2_values, _, _ = self.target_critic_2.act(
                    {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                )

                target_q_values = (
                    torch.min(target_q1_values, target_q2_values)
                    - self._entropy_coefficient * next_log_prob
                )

                target_values = (
                    sampled_rewards[:, -1]
                    + self._discount_factor
                    * sampled_terminated[:, -1].logical_not()
                    * target_q_values
                )

            # ---------------------- Critic loss ----------------------
            critic_1_values, _, _ = self.critic_1.act(
                {**inputs, "taken_actions": sampled_actions[:, -1]}, role="critic_1"
            )
            critic_2_values, _, _ = self.critic_2.act(
                {**inputs, "taken_actions": sampled_actions[:, -1]}, role="critic_2"
            )

            critic_loss = (
                F.mse_loss(critic_1_values, target_values) +
                F.mse_loss(critic_2_values, target_values)
            ) / 2

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self._grad_norm_clip
                )

            self.critic_optimizer.step()

            # ---------------------- Policy loss ----------------------
            actions, log_prob, _ = self.policy.act(inputs, role="policy")

            with torch.no_grad():
                critic_1_values, _, _ = self.critic_1.act(
                    {**inputs, "taken_actions": actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {**inputs, "taken_actions": actions}, role="critic_2"
                )

            policy_loss = (
                self._entropy_coefficient * log_prob
                - torch.min(critic_1_values, critic_2_values)
            ).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.policy_optimizer.step()

            # ---------------------- Entropy tuning ----------------------
            if self._learn_entropy:
                entropy_loss = -(
                    self.log_entropy_coefficient
                    * (log_prob + self._target_entropy).detach()
                ).mean()

                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_optimizer.step()

                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            # ---------------------- Target updates ----------------------
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # ---------------------- LR schedulers ----------------------
            # if self.policy_scheduler:
            #     self.policy_scheduler.step()
            # if self.critic_scheduler:
            #     self.critic_scheduler.step()
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # ---------------------- Logging ----------------------
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self._learning_rate_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])