from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .recording import record_transition
from .types import RolloutResult


def collect_until_all_envs_done_one_episode(
    env: Any,
    agent: Any,
    memory: Any,
    *,
    policy_act_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Any]],
    max_steps: int = 1_000_000,
) -> RolloutResult:
    """Roll out until each parallel env has completed at least one episode (terminated or truncated).

    ``policy_act_fn(obs_tensor) -> (actions, aux)`` should match your training pipeline (e.g. agent.act).

    Appends xy positions each step to :class:`RolloutResult` for plotting; fill references from
    ``env.unwrapped._desired_trajectory_w`` in your driver if needed.
    """
    num_envs = getattr(env, "num_envs", 1)
    device = getattr(env, "device", torch.device("cpu"))
    completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    positions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

    for _step in range(max_steps):
        actions, _ = policy_act_fn(obs)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.as_tensor(terminated, device=device)
        if truncated is None:
            truncated = torch.zeros(terminated.shape[0], dtype=torch.bool, device=device)
        elif not isinstance(truncated, torch.Tensor):
            truncated = torch.as_tensor(truncated, device=device)

        record_transition(
            memory,
            agent,
            states=obs,
            actions=actions,
            rewards=rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, device=device),
            next_states=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

        base = getattr(env, "unwrapped", env)
        robot = getattr(base, "_robot", None)
        if robot is not None:
            xy = robot.data.root_pos_w[:, :2].detach().cpu().numpy()
            for i in range(num_envs):
                positions[i].append(xy[i].copy())

        obs = next_obs
        ep_done = terminated.bool() | truncated.bool()
        completed = completed | ep_done
        if bool(completed.all().item()):
            break

    return RolloutResult(
        positions_xy_per_env=[np.stack(p, axis=0) if p else np.zeros((0, 2)) for p in positions],
        extras={"steps": _step + 1},
    )
