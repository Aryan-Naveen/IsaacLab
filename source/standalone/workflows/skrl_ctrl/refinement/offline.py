from __future__ import annotations

from typing import Any


def offline_train_steps(agent: Any, num_steps: int, *, timestep_offset: int = 0) -> list[Any]:
    """Run ``agent._update`` for ``num_steps`` offline gradient steps.

    Configure the agent beforehand, e.g. ``random_timesteps=0``, ``learning_starts=0``, batch size, LR.
    Returns a list of per-step return values if ``_update`` returns anything (often ``None``).
    """
    logs = []
    for i in range(num_steps):
        out = agent._update(timestep=timestep_offset + i, timesteps=timestep_offset + num_steps)
        logs.append(out)
    return logs
