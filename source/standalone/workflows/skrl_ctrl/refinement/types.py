from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RolloutResult:
    """Data from a vectorized rollout (B parallel envs)."""

    positions_xy_per_env: list[np.ndarray] = field(default_factory=list)
    """Length B; each array is ``(T_i, 2)`` world-frame xy (or include z as third column if stored)."""

    reference_xy_per_env: list[np.ndarray] = field(default_factory=list)
    """Optional per-env reference polylines ``(N, 2)`` aligned with env timestep indexing."""

    extras: dict[str, Any] = field(default_factory=dict)
