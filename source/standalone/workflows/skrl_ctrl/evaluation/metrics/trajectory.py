"""RMSE and success vs reference polyline (world frame)."""

from __future__ import annotations

import math

import torch

# Default composite success (``combined_default``): all must hold on aligned samples (3D norm).
COMBINED_SUCCESS_MAX_M = 1.0
COMBINED_SUCCESS_MEAN_M = 0.5
COMBINED_SUCCESS_TERMINAL_M = 0.5


def trajectory_rmse(
    positions: torch.Tensor,
    reference: torch.Tensor,
    *,
    xy_only: bool = False,
) -> float:
    """Root-mean-square error over time (aligned indices).

    ``positions`` and ``reference`` are ``(T,)`` or ``(T, 3)``; we use ``min(T_pos, T_ref)`` samples.
    """
    if positions.numel() == 0 or reference.numel() == 0:
        return float("nan")
    T = min(positions.shape[0], reference.shape[0])
    p = positions[:T]
    r = reference[:T]
    if p.dim() == 1:
        p = p.unsqueeze(-1)
    if r.dim() == 1:
        r = r.unsqueeze(-1)
    if xy_only:
        p = p[..., :2]
        r = r[..., :2]
    err = torch.linalg.norm(p - r, dim=-1)
    return math.sqrt(torch.mean(err**2).item())


def success_from_xy_error(
    positions: torch.Tensor,
    reference: torch.Tensor,
    *,
    crashed: bool,
    success_radius_m: float,
    criterion: str = "combined_default",
) -> bool:
    """``success = (not crashed) and`` a tracking predicate (see ``criterion``).

    Note: :func:`run_b_rollouts` uses survival-only success (``not crashed``) and does not call this.

    **Default composite** (``combined_default``; ignores ``success_radius_m``):

    - Over :math:`t \\in [0, \\min(T_{pos}, T_{ref}))`, 3D error :math:`e_t = \\|p_t - r_t\\|_2`.
    - Require :math:`\\max_t e_t < 1\\,\\mathrm{m}`, :math:`\\mathrm{mean}_t e_t < 0.5\\,\\mathrm{m}`,
      and terminal :math:`\\|p_{T-1}-r_{i}\\|_2 < 0.5\\,\\mathrm{m}` with
      :math:`i = \\min(T-1, N_{ref}-1)`.

    **Whole-horizon XY** (uses ``min(T_pos, T_ref)`` steps):

    - ``max_xy_below_radius``: :math:`\\max_t \\|e_{xy,t}\\| < \\text{radius}`.
    - ``mean_xy_below_radius``: :math:`\\mathrm{mean}_t \\|e_{xy,t}\\| < \\text{radius}`.

    **Endpoint**:

    - ``terminal_xy_below_radius`` / ``terminal_xyz_below_radius``: last step vs reference index
      :math:`i` (uses ``success_radius_m``).
    """
    if crashed:
        return False
    if positions.numel() == 0 or reference.numel() == 0:
        return False

    if criterion == "combined_default":
        T = min(int(positions.shape[0]), int(reference.shape[0]))
        p = positions[:T, :3]
        r = reference[:T, :3]
        err = torch.linalg.norm(p - r, dim=-1)
        if not (bool((err.max() < COMBINED_SUCCESS_MAX_M).item()) and bool((err.mean() < COMBINED_SUCCESS_MEAN_M).item())):
            return False
        ti = int(positions.shape[0] - 1)
        ti_r = min(ti, int(reference.shape[0] - 1))
        term = torch.linalg.norm(positions[ti, :3] - reference[ti_r, :3])
        return bool(term < COMBINED_SUCCESS_TERMINAL_M)

    if criterion in ("terminal_xy_below_radius", "terminal_xyz_below_radius"):
        ti = int(positions.shape[0] - 1)
        ti_r = min(ti, int(reference.shape[0] - 1))
        p = positions[ti]
        r = reference[ti_r]
        if criterion == "terminal_xy_below_radius":
            err = torch.linalg.norm(p[:2] - r[:2])
        else:
            err = torch.linalg.norm(p[:3] - r[:3])
        return bool(err < success_radius_m)

    T = min(positions.shape[0], reference.shape[0])
    p = positions[:T, :2]
    r = reference[:T, :2]
    err_xy = torch.linalg.norm(p - r, dim=-1)
    if criterion == "max_xy_below_radius":
        return bool((err_xy.max() < success_radius_m).item())
    if criterion == "mean_xy_below_radius":
        return bool((err_xy.mean() < success_radius_m).item())
    raise ValueError(f"Unknown success criterion: {criterion!r}")


def aggregate_rollout_metrics(
    rmses: list[float],
    successes: list[bool],
) -> tuple[float, float, float]:
    """Returns (mean_rmse, std_rmse, success_rate)."""
    import numpy as np

    arr = np.array([x for x in rmses if not math.isnan(x)], dtype=np.float64)
    mean_rmse = float(arr.mean()) if arr.size else 0.0
    std_rmse = float(arr.std(ddof=0)) if arr.size else 0.0
    sr = float(sum(successes) / len(successes)) if successes else 0.0
    return mean_rmse, std_rmse, sr


def aggregate_return_metrics(
    total_rewards: list[float],
    successes: list[bool],
) -> tuple[float, float, float]:
    """Returns (mean_total_reward, std_total_reward, success_rate)."""
    import numpy as np

    arr = np.array(total_rewards, dtype=np.float64)
    mean_tr = float(arr.mean()) if arr.size else 0.0
    std_tr = float(arr.std(ddof=0)) if arr.size else 0.0
    sr = float(sum(successes) / len(successes)) if successes else 0.0
    return mean_tr, std_tr, sr
