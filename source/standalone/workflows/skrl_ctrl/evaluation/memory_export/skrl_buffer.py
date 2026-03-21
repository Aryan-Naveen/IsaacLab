"""Bridge rollout tensors to skrl :class:`RandomMemory` and ``.pt`` export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from refinement.recording import record_transition


def append_transition_batch(
    memory: Any,
    agent: Any,
    *,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> None:
    """Record one vectorized transition (batch ``num_envs``)."""
    record_transition(
        memory,
        agent,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        terminated=terminated,
        truncated=truncated,
    )


def save_memory_pt(memory: Any, path: str | Path, *, format: str = "pt") -> None:
    """Persist skrl memory (same as training checkpoints)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    memory.save(str(path), format=format)


def fill_random_memory_from_transition_dict(
    mapping: dict[str, torch.Tensor],
    device: torch.device,
    agent: Any,
) -> Any:
    """Fill a :class:`RandomMemory` (``num_envs=1``) with stacked transitions."""
    from skrl.memories.torch import RandomMemory

    states = mapping["states"]
    n = states.shape[0]
    mem = RandomMemory(memory_size=n, num_envs=1, device=device)
    # Same layout as skrl SAC.init() (memory tensors must exist before add_samples).
    mem.create_tensor(name="states", size=agent.observation_space, dtype=torch.float32)
    mem.create_tensor(name="next_states", size=agent.observation_space, dtype=torch.float32)
    mem.create_tensor(name="actions", size=agent.action_space, dtype=torch.float32)
    mem.create_tensor(name="rewards", size=1, dtype=torch.float32)
    mem.create_tensor(name="terminated", size=1, dtype=torch.bool)
    agent._tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]
    for t in range(n):
        record_transition(
            mem,
            agent,
            states=states[t : t + 1].to(device),
            actions=mapping["actions"][t : t + 1].to(device),
            rewards=mapping["rewards"][t : t + 1].to(device),
            next_states=mapping["next_states"][t : t + 1].to(device),
            terminated=mapping["dones"][t : t + 1].to(device),
            truncated=None,
        )
    return mem
