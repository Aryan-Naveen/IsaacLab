"""Merge base finetune-online CDC preset JSON with per-run fields.

Intended to be run via ``isaaclab.sh -p`` from bash (not bare ``python3``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Path to base preset JSON")
    p.add_argument("--out", required=True, help="Path to write merged preset JSON")
    p.add_argument("--agent-type", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--checkpoint-name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--wandb-name", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--coeff-json", required=True, help="Absolute path to trajectory coeff JSON")
    p.add_argument(
        "--policy-phased-learning-rate",
        action="store_true",
        help="Set policy_phased_learning_rate (CTRLSAC-multi: phase1 LR first half of learning window, then phase2).",
    )
    p.add_argument("--policy-phased-lr-phase1", type=float, default=1e-6)
    p.add_argument("--policy-phased-lr-phase2", type=float, default=1e-6)
    args = p.parse_args()

    base = Path(args.base)
    with open(base, encoding="utf-8") as f:
        d = json.load(f)
    d.update(
        {
            "agent_type": args.agent_type,
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_name": args.checkpoint_name,
            "seed": args.seed,
            "wandb_name": args.wandb_name,
            "output_root": args.output_root,
            "coeff_json": args.coeff_json,
        }
    )
    if args.policy_phased_learning_rate:
        d["policy_phased_learning_rate"] = True
        d["policy_phased_lr_phase1"] = args.policy_phased_lr_phase1
        d["policy_phased_lr_phase2"] = args.policy_phased_lr_phase2
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()
