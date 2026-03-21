"""SAC with optional offline AWR updates (Bellman critic + advantage-weighted BC)."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import gym
import gymnasium
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.sac import SAC
from skrl.memories.torch import Memory
from skrl.models.torch import Model


class OfflineAWRSAC(SAC):
    """Drop-in ``SAC`` subclass: online training unchanged; call ``configure_offline_finetune`` for AWR."""

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory, ...]]] = None,
        observation_space: Optional[Union[int, Tuple[int, ...], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int, ...], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        if not hasattr(self, "_tensors_names"):
            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]

    def configure_offline_finetune(
        self,
        *,
        critic_lr: float | None = None,
        actor_lr: float | None = None,
        awr_beta: float = 1.0,
        awr_weight_max: float = 20.0,
        awr_num_action_samples: int = 8,
    ) -> None:
        """Enable offline AWR: Bellman TD on critics, then advantage-weighted log-likelihood on the actor."""
        self.cfg["use_offline_awr_update"] = True
        self.cfg["awr_beta"] = float(awr_beta)
        self.cfg["awr_weight_max"] = float(awr_weight_max)
        self.cfg["awr_num_action_samples"] = int(awr_num_action_samples)
        self._learn_entropy = False

        if critic_lr is not None and getattr(self, "critic_optimizer", None) is not None:
            for g in self.critic_optimizer.param_groups:
                g["lr"] = float(critic_lr)
        if actor_lr is not None and getattr(self, "policy_optimizer", None) is not None:
            for g in self.policy_optimizer.param_groups:
                g["lr"] = float(actor_lr)

    def _metrics_logging_enabled(self) -> bool:
        if int(getattr(self, "write_interval", 0) or 0) > 0:
            return True
        if int(getattr(self, "_write_interval", 0) or 0) > 0:
            return True
        exp = self.cfg.get("experiment", {}) if isinstance(self.cfg, dict) else {}
        return int(exp.get("write_interval", 0) or 0) > 0

    def _offline_awr_update(self, timestep: int, timesteps: int) -> None:
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = (
            self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]
        )

        sampled_states = self._state_preprocessor(sampled_states, train=True)
        sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

        not_done = sampled_dones.logical_not()
        if not_done.dtype != sampled_rewards.dtype:
            not_done = not_done.to(dtype=sampled_rewards.dtype)

        with torch.no_grad():
            next_actions, _, _ = self.policy.act({"states": sampled_next_states}, role="policy")
            target_q1, _, _ = self.target_critic_1.act(
                {"states": sampled_next_states, "taken_actions": next_actions},
                role="target_critic_1",
            )
            target_q2, _, _ = self.target_critic_2.act(
                {"states": sampled_next_states, "taken_actions": next_actions},
                role="target_critic_2",
            )
            target_q = torch.min(target_q1, target_q2)
            target_values = sampled_rewards + self._discount_factor * not_done * target_q

        q1, _, _ = self.critic_1.act(
            {"states": sampled_states, "taken_actions": sampled_actions},
            role="critic_1",
        )
        q2, _, _ = self.critic_2.act(
            {"states": sampled_states, "taken_actions": sampled_actions},
            role="critic_2",
        )
        critic_loss = 0.5 * (F.mse_loss(q1, target_values) + F.mse_loss(q2, target_values))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if config.torch.is_distributed:
            self.critic_1.reduce_parameters()
            self.critic_2.reduce_parameters()
        if self._grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                self._grad_norm_clip,
            )
        self.critic_optimizer.step()

        beta = float(self.cfg.get("awr_beta", 1.0))
        w_max = float(self.cfg.get("awr_weight_max", 20.0))
        k_samples = max(1, int(self.cfg.get("awr_num_action_samples", 8)))

        with torch.no_grad():
            q1_d, _, _ = self.critic_1.act(
                {"states": sampled_states, "taken_actions": sampled_actions},
                role="critic_1",
            )
            q2_d, _, _ = self.critic_2.act(
                {"states": sampled_states, "taken_actions": sampled_actions},
                role="critic_2",
            )
            q_dataset = torch.min(q1_d, q2_d)

            v_terms = []
            for _ in range(k_samples):
                actions_pi, _, _ = self.policy.act({"states": sampled_states}, role="policy")
                q1_pi, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": actions_pi},
                    role="critic_1",
                )
                q2_pi, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": actions_pi},
                    role="critic_2",
                )
                v_terms.append(torch.min(q1_pi, q2_pi))
            v = torch.stack(v_terms, dim=0).mean(dim=0)

            adv = q_dataset - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weights = torch.exp(adv / beta).clamp(max=w_max)

        _, log_prob, _ = self.policy.act(
            {"states": sampled_states, "taken_actions": sampled_actions},
            role="policy",
        )
        policy_loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if config.torch.is_distributed:
            self.policy.reduce_parameters()
        if self._grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
        self.policy_optimizer.step()

        self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
        self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

        if self._learning_rate_scheduler:
            self.policy_scheduler.step()
            self.critic_scheduler.step()

        if self._metrics_logging_enabled():
            self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())
            self.track_data("Q-network / Q1 (mean)", q_dataset.mean().item())
            self.track_data("Advantage / mean", adv.mean().item())

    def _update(self, timestep: int, timesteps: int) -> None:
        if self.cfg.get("use_offline_awr_update"):
            for sub in range(self._gradient_steps):
                self._offline_awr_update(timestep + sub, timesteps)
            return
        super()._update(timestep, timesteps)
