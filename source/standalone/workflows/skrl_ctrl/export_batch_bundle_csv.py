#!/usr/bin/env python3
"""Export per-env rollout rewards from an ``EvalRolloutBundle`` (.pt) to CSV.

Assumes env ``i`` used coefficient row ``pool[i]`` (``sequential_pool_indices`` in eval_batch).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.bundle import load_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path, help="Path to EvalRolloutBundle .pt")
    ap.add_argument("--out", type=Path, required=True, help="Per-rollout CSV path")
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional single-row CSV with mean/std total reward",
    )
    ap.add_argument("--section", type=str, default="", help="Label column e.g. source | indomain | ood")
    ap.add_argument("--agent-type", type=str, default="", dest="agent_type")
    ap.add_argument("--folder", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    meta = bundle.meta
    pool = meta.predefined_task_coeff or []
    B = meta.B
    if len(bundle.rollouts) != B:
        raise ValueError(f"rollouts length {len(bundle.rollouts)} != meta.B {B}")
    if len(pool) < B:
        raise ValueError(f"coeff pool size {len(pool)} < B={B}")

    coeff_headers = [f"c{i}" for i in range(7)]
    fieldnames = [
        "section",
        "agent_type",
        "folder",
        "ckpt",
        "trial_idx",
        *coeff_headers,
        "total_reward",
        "crashed",
        "success",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, rec in enumerate(bundle.rollouts):
            row = pool[i]
            if len(row) < 7:
                row = list(row) + [0.0] * (7 - len(row))
            else:
                row = row[:7]
            w.writerow(
                {
                    "section": args.section,
                    "agent_type": args.agent_type,
                    "folder": args.folder,
                    "ckpt": args.ckpt,
                    "trial_idx": i,
                    **{f"c{j}": row[j] for j in range(7)},
                    "total_reward": rec.total_reward,
                    "crashed": int(rec.crashed),
                    "success": int(rec.success),
                }
            )

    if args.summary_out is not None:
        m = bundle.metrics
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "section",
                    "agent_type",
                    "folder",
                    "ckpt",
                    "B",
                    "mean_total_reward",
                    "std_total_reward",
                    "success_rate",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "section": args.section,
                    "agent_type": args.agent_type,
                    "folder": args.folder,
                    "ckpt": args.ckpt,
                    "B": B,
                    "mean_total_reward": m.mean_total_reward,
                    "std_total_reward": m.std_total_reward,
                    "success_rate": m.success_rate,
                }
            )


if __name__ == "__main__":
    main()
