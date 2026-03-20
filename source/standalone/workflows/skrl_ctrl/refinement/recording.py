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

    Aligns with skrl SAC-style memories: ``states``, ``actions``, ``rewards``, ``next_states``, and
    a done flag. If your skrl build uses ``dones`` only, pass ``terminated | truncated`` as a single
    done tensor via ``terminated=...`` and leave ``truncated`` as ``None``.

    The exact ``memory.add_samples`` / internal API may differ by skrl version; adjust this function
    after inspecting ``SequentialTrainer`` and ``agent._tensors_names`` in your install.
    """
    names = getattr(agent, "_tensors_names", None)
    if truncated is not None:
        dones = (terminated.bool() | truncated.bool()).to(dtype=torch.float32)
    else:
        dones = terminated.to(dtype=torch.float32)

    if names is None:
        names = ("states", "actions", "rewards", "next_states", "dones")

    mapping = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
    }
    if hasattr(memory, "add_samples"):
        try:
            memory.add_samples([mapping[n] for n in names if n in mapping], names=names)
        except TypeError:
            memory.add_samples(**{n: mapping[n] for n in names if n in mapping})
    else:
        raise NotImplementedError("Adapt record_transition to your RandomMemory API (add_samples / record).")
