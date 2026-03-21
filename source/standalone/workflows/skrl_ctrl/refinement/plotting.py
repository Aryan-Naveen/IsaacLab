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


def plot_eval_bundle_xy(
    bundle: "EvalRolloutBundle",
    *,
    ref_index: int = 0,
    out_path: str | Path | None = None,
    title: str = "Reference vs rollouts (XY)",
) -> None:
    """Overlay reference polyline with rollout XY paths from an :class:`EvalRolloutBundle`."""
    ref = bundle.reference[ref_index].numpy()[:, :2]
    trajs = [r.positions.numpy()[:, :2] for r in bundle.rollouts]
    plot_reference_vs_samples(ref, trajs, out_path=out_path, title=title)


def plot_eval_bundle_from_pt(
    bundle_path: str | Path,
    *,
    ref_index: int = 0,
    out_path: str | Path | None = None,
    title: str = "Reference vs rollouts (XY)",
) -> None:
    """Load ``EvalRolloutBundle`` from ``.pt`` and plot reference vs rollout XY paths."""
    from evaluation.bundle import load_bundle

    bundle = load_bundle(bundle_path)
    plot_eval_bundle_xy(bundle, ref_index=ref_index, out_path=out_path, title=title)


def plot_eval_bundle_xy_density(
    bundle_path: str | Path,
    *,
    ref_index: int = 0,
    out_path: str | Path | None = None,
    gridsize: int = 40,
    title: str = "XY density + reference",
) -> None:
    """Hexbin density of sampled XY positions with reference polyline overlay."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_eval_bundle_xy_density requires matplotlib") from e

    from evaluation.bundle import load_bundle

    bundle = load_bundle(bundle_path)
    ref = bundle.reference[ref_index].numpy()[:, :2]
    pts = np.concatenate([r.positions.numpy()[:, :2] for r in bundle.rollouts], axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(pts[:, 0], pts[:, 1], gridsize=gridsize, cmap="Blues", mincnt=1, alpha=0.65)
    ax.plot(ref[:, 0], ref[:, 1], "r-", linewidth=2, label="reference")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
