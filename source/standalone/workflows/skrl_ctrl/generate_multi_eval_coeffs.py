#!/usr/bin/env python3
"""Generate ``in_domain_100.json``: Dirichlet convex combos of the three source vertices, shuffled.

Each row is ``[0, λ1, λ2, 0, 0, 0, 0]`` with ``(λ1, λ2, λ3) ~ Dirichlet(1,1,1)`` and
``λ1+λ2+λ3=1`` (same as λ1*p1 + λ2*p2 + λ3*p3 with one-hot p1, p2 and zero p3).

Run from repo (or skrl_ctrl) with numpy available::

    python source/standalone/workflows/skrl_ctrl/generate_multi_eval_coeffs.py --out-dir /tmp/me --master-seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for in_domain_100.json")
    ap.add_argument("--master-seed", type=int, required=True, help="RNG seed for sampling and shuffle")
    ap.add_argument("--n", type=int, default=100, help="Number of distinct coefficient rows")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.master_seed)

    rows: list[list[float]] = []
    for _ in range(args.n):
        l1, l2, l3 = rng.dirichlet((1.0, 1.0, 1.0))
        rows.append([0.0, float(l1), float(l2), 0.0, 0.0, 0.0, 0.0])

    rng.shuffle(rows)

    out_path = args.out_dir / "in_domain_100.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {out_path.resolve()} ({args.n} rows, master_seed={args.master_seed})")


if __name__ == "__main__":
    main()
