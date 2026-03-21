"""Dataclasses for batch eval rollout artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


BUNDLE_VERSION = "2"


@dataclass
class RolloutRecord:
    """One completed episode / rollout."""

    positions: torch.Tensor  # (T, 3) world frame
    rmse: float
    total_reward: float  # sum of per-step rewards for the episode
    success: bool  # True iff no death termination (same as ``not crashed`` for batch eval)
    crashed: bool
    episode_length: int


@dataclass
class EvalBundleMeta:
    """Metadata stored with :class:`EvalRolloutBundle`."""

    bundle_version: str = BUNDLE_VERSION
    task_gym_id: str = ""
    checkpoint_path: str = ""
    B: int = 0
    success_radius_m: float = 0.3  # informational; batch success_rate is survival-only
    success_criterion: str = "combined_default"  # informational
    predefined_task_coeff: list[list[float]] | None = None
    init_noise_enabled: bool = False
    init_noise_config: dict[str, Any] | None = None
    policy_stochastic: bool = False
    policy_train_mode: bool = False
    seed: int = 0
    max_steps: int = 0


@dataclass
class EvalBundleMetrics:
    mean_total_reward: float = 0.0
    std_total_reward: float = 0.0
    success_rate: float = 0.0
    per_rollout_total_reward: list[float] = field(default_factory=list)
    per_rollout_success: list[bool] = field(default_factory=list)


@dataclass
class EvalRolloutBundle:
    meta: EvalBundleMeta
    metrics: EvalBundleMetrics
    reference: torch.Tensor
    rollouts: list[RolloutRecord]
    transitions: dict[str, torch.Tensor] | None = None
    transitions_path: str | None = None
    memory_path: str | None = None
    video_path: str | None = None
