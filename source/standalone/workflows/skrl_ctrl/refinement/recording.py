from __future__ import annotations

from typing import Any

import torch


def record_transition(
    memory: Any,
    agent: Any,
    *,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor | None = None,
) -> None:
    """Append one vectorized transition to ``memory`` using field names expected by ``agent``.

    Matches skrl torch SAC memory layout (see Toni-SM/skrl SAC.init, e.g. commit 3bd530cf):
    tensors ``states``, ``actions``, ``rewards``, ``next_states``, ``terminated`` (bool).
    ``agent._tensors_names`` is set in ``SAC.init()`` or by offline helpers such as
    ``fill_random_memory_from_transition_dict``.
    """
    names = getattr(agent, "_tensors_names", None)
    if truncated is not None:
        term_bool = terminated.bool() | truncated.bool()
    else:
        term_bool = terminated.bool() if terminated.dtype == torch.bool else terminated != 0

    if names is None:
        names = ("states", "actions", "rewards", "next_states", "terminated")

    mapping = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "terminated": term_bool.to(dtype=torch.bool),
    }
    if hasattr(memory, "add_samples"):
        try:
            memory.add_samples([mapping[n] for n in names if n in mapping], names=names)
        except TypeError:
            memory.add_samples(**{n: mapping[n] for n in names if n in mapping})
    else:
        raise NotImplementedError("Adapt record_transition to your RandomMemory API (add_samples / record).")
