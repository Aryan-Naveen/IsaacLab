#!/usr/bin/env python3
"""Load eval run metrics and plot total reward by coefficient for each algorithm.

Layout expected (from ``eval_driver``)::

    <eval_root>/<agent>_<ckpt_stem>/coeff_<hash>/<timestamp>/metrics.json

``metrics.json`` currently stores scalar ``mean_total_reward`` / ``episode_total_reward``,
not per-step returns. This script plots those scalars across coefficients:

- **Sweep figure**: one line per algorithm (x = trajectory coefficient row, y = total reward).
- **Per-coefficient figure**: grid of bar charts comparing algorithms per coefficient (from metrics).

If a future eval writes ``episode_reward_cumulative`` (list of floats, one per step),
pass ``--per-step`` to plot that trajectory per (coeff, algorithm) instead.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_COEFF_DIR = re.compile(r"^coeff_([0-9a-f]+)$", re.IGNORECASE)


def _parse_algo_from_run_dir(name: str) -> str:
    """Human-readable label from folder like ``SAC_best_agent`` -> ``SAC``."""
    if name.endswith("_best_agent"):
        return name[: -len("_best_agent")]
    return name


def _discover_metrics(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not eval_root.is_dir():
        return rows
    for metrics_path in eval_root.rglob("metrics.json"):
        parent = metrics_path.parent.name
        coeff_parent = metrics_path.parent.parent
        run_dir = coeff_parent.parent
        m = _COEFF_DIR.match(coeff_parent.name)
        if not m:
            continue
        coeff_hash = m.group(1)
        algo_folder = run_dir.name
        timestamp = parent
        rows.append(
            {
                "path": metrics_path,
                "algo_folder": algo_folder,
                "algo": _parse_algo_from_run_dir(algo_folder),
                "coeff_hash": coeff_hash,
                "timestamp": timestamp,
            }
        )
    return rows


def _load_metrics(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _reward_from_metrics(data: dict[str, Any]) -> float:
    if "mean_total_reward" in data:
        return float(data["mean_total_reward"])
    if "episode_total_reward" in data:
        return float(data["episode_total_reward"])
    raise KeyError("metrics.json missing mean_total_reward / episode_total_reward")


def _coeff_row_from_metrics(data: dict[str, Any]) -> list[float] | None:
    """Legendre / basis row from eval_driver metrics; older runs may omit these."""
    row = data.get("coefficient_row")
    if isinstance(row, list) and row and all(isinstance(x, (int, float)) for x in row):
        return [float(x) for x in row]
    tc = data.get("trajectory_coefficients")
    if isinstance(tc, list) and tc and isinstance(tc[0], list):
        inner = tc[0]
        if inner and all(isinstance(x, (int, float)) for x in inner):
            return [float(x) for x in inner]
    return None


def _format_coeff_label(row: list[float] | None, coeff_hash: str, max_chars: int = 80) -> str:
    if row is None:
        return coeff_hash
    parts = [f"{x:g}" for x in row]
    s = "[" + ", ".join(parts) + "]"
    if len(s) > max_chars:
        s = s[: max_chars - 2] + "…]"
    return s


def _latest_per_algo_coeff(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep the lexicographically latest timestamp per (algo_folder, coeff_hash)."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["algo_folder"], r["coeff_hash"])
        prev = best.get(key)
        if prev is None or r["timestamp"] > prev["timestamp"]:
            best[key] = r
    return best


def _pivot(
    latest: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, dict[str, float]], list[str]]:
    """Coeff hashes sorted by trajectory values, algos sorted, reward[coeff][algo] = float, parallel labels."""
    coeff_set: set[str] = set()
    algo_set: set[str] = set()
    reward: dict[str, dict[str, float]] = defaultdict(dict)
    row_by_hash: dict[str, list[float] | None] = {}
    for (algo_folder, coeff_hash), meta in latest.items():
        data = _load_metrics(meta["path"])
        coeff_set.add(coeff_hash)
        if coeff_hash not in row_by_hash:
            row_by_hash[coeff_hash] = _coeff_row_from_metrics(data)
        algo = meta["algo"]
        algo_set.add(algo)
        reward[coeff_hash][algo] = _reward_from_metrics(data)

    def _coeff_sort_key(h: str) -> tuple:
        row = row_by_hash.get(h)
        if row is None:
            return (1, (), h)
        return (0, tuple(row), h)

    coeffs_sorted = sorted(coeff_set, key=_coeff_sort_key)
    algos_sorted = sorted(algo_set)
    labels = [_format_coeff_label(row_by_hash.get(h), h) for h in coeffs_sorted]
    return coeffs_sorted, algos_sorted, {c: dict(reward[c]) for c in coeffs_sorted}, labels


def plot_sweep(
    coeffs: list[str],
    coeff_labels: list[str],
    algos: list[str],
    reward: dict[str, dict[str, float]],
    out_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    longest = max((len(l) for l in coeff_labels), default=0)
    w = max(10.0, min(24.0, len(coeffs) * max(1.2, longest * 0.04)))
    fig, ax = plt.subplots(figsize=(w, 5))
    x = range(len(coeffs))
    for algo in algos:
        ys = [reward[c].get(algo, float("nan")) for c in coeffs]
        ax.plot(x, ys, marker="o", linewidth=1.5, label=algo)
    ax.set_xticks(list(x))
    ax.set_xticklabels(coeff_labels, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("trajectory coefficients (basis row)")
    ax.set_ylabel("total reward")
    ax.set_title("Total reward across coefficient sweep (latest run per algo × coeff)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_coeff_bars(
    coeffs: list[str],
    coeff_labels: list[str],
    algos: list[str],
    reward: dict[str, dict[str, float]],
    out_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(coeffs)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
    width = 0.8 / max(1, len(algos))
    x_base = np.arange(len(algos))
    for i, c in enumerate(coeffs):
        r, col = divmod(i, ncols)
        ax = axes[r][col]
        heights = [reward[c].get(a, float("nan")) for a in algos]
        ax.bar(x_base, heights, width=0.7, tick_label=algos)
        title = coeff_labels[i]
        if len(title) > 48:
            title = title[:45] + "…"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("total reward")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(coeffs), nrows * ncols):
        r, col = divmod(j, ncols)
        axes[r][col].set_visible(False)
    fig.suptitle("Total reward by algorithm (per coefficient)")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_step_trajectories(
    latest: dict[tuple[str, str], dict[str, Any]],
    key: str,
    out_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    # Group: coeff -> list of (algo, series)
    by_coeff: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    row_by_hash: dict[str, list[float] | None] = {}
    for (_algo_folder, coeff_hash), meta in latest.items():
        data = _load_metrics(meta["path"])
        if coeff_hash not in row_by_hash:
            row_by_hash[coeff_hash] = _coeff_row_from_metrics(data)
        if key not in data or not isinstance(data[key], list):
            continue
        series = [float(x) for x in data[key]]
        by_coeff[coeff_hash].append((meta["algo"], series))

    if not by_coeff:
        raise SystemExit(f"No metrics contained non-empty list field {key!r}")

    def _coeff_sort_key(h: str) -> tuple:
        row = row_by_hash.get(h)
        if row is None:
            return (1, (), h)
        return (0, tuple(row), h)

    coeffs = sorted(by_coeff.keys(), key=_coeff_sort_key)
    labels = [_format_coeff_label(row_by_hash.get(h), h) for h in coeffs]
    n = len(coeffs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for i, c in enumerate(coeffs):
        r, col = divmod(i, ncols)
        ax = axes[r][col]
        for algo, series in by_coeff[c]:
            ax.plot(range(len(series)), series, label=algo, linewidth=1.2)
        t = labels[i]
        if len(t) > 52:
            t = t[:49] + "…"
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("cumulative reward")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(len(coeffs), nrows * ncols):
        r, col = divmod(j, ncols)
        axes[r][col].set_visible(False)
    fig.suptitle(f"Per-episode cumulative reward ({key})")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path("runs/eval"),
        help="Root directory containing <agent>_*/coeff_*/<ts>/metrics.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: <eval-root>/plots, or ./eval_reward_plots if not writable)",
    )
    p.add_argument("--show", action="store_true", help="Show figures interactively")
    p.add_argument(
        "--per-step",
        metavar="JSON_KEY",
        default=None,
        help="If set, plot list field JSON_KEY as cumulative reward vs step (per coeff panel)",
    )
    args = p.parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        preferred = eval_root / "plots"
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            out_dir = preferred.resolve()
        except OSError:
            out_dir = (Path.cwd() / "eval_reward_plots").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

    rows = _discover_metrics(eval_root)
    if not rows:
        raise SystemExit(f"No metrics.json found under {eval_root}")

    latest = _latest_per_algo_coeff(rows)

    if args.per_step:
        plot_per_step_trajectories(
            latest,
            args.per_step,
            out_dir / "eval_reward_per_step.png",
            args.show,
        )
        return

    coeffs, algos, reward, coeff_labels = _pivot(latest)
    plot_sweep(
        coeffs,
        coeff_labels,
        algos,
        reward,
        out_dir / "eval_reward_sweep.png",
        args.show,
    )
    plot_per_coeff_bars(
        coeffs,
        coeff_labels,
        algos,
        reward,
        out_dir / "eval_reward_by_coeff.png",
        args.show,
    )
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
