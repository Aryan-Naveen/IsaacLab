from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_reference_vs_samples(
    reference_xy: np.ndarray,
    sample_trajs_xy: Sequence[np.ndarray],
    *,
    out_path: str | Path | None = None,
    title: str = "Reference vs samples",
) -> None:
    """Overlay reference polyline ``(N, 2)`` with sample trajectories (each ``(T, 2)``).

    Saves a PNG if ``out_path`` is set. Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_reference_vs_samples requires matplotlib") from e

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k-", linewidth=2, label="reference")
    for i, traj in enumerate(sample_trajs_xy):
        if traj.size == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.35, label=None if i else "samples")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
