"""Save / load :class:`EvalRolloutBundle` as ``.pt`` (torch pickle)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .schema import BUNDLE_VERSION, EvalBundleMeta, EvalBundleMetrics, EvalRolloutBundle, RolloutRecord


def _rollout_to_dict(r: RolloutRecord) -> dict[str, Any]:
    return {
        "positions": r.positions.detach().cpu(),
        "rmse": r.rmse,
        "total_reward": r.total_reward,
        "success": r.success,
        "crashed": r.crashed,
        "episode_length": r.episode_length,
    }


def _rollout_from_dict(d: dict[str, Any]) -> RolloutRecord:
    return RolloutRecord(
        positions=d["positions"],
        rmse=float(d.get("rmse", 0.0)),
        total_reward=float(d.get("total_reward", 0.0)),
        success=bool(d["success"]),
        crashed=bool(d["crashed"]),
        episode_length=int(d["episode_length"]),
    )


def bundle_to_state_dict(bundle: EvalRolloutBundle) -> dict[str, Any]:
    return {
        "bundle_version": bundle.meta.bundle_version,
        "meta": bundle.meta.__dict__,
        "metrics": bundle.metrics.__dict__,
        "reference": bundle.reference.detach().cpu(),
        "rollouts": [_rollout_to_dict(r) for r in bundle.rollouts],
        "transitions": {k: v.detach().cpu() for k, v in (bundle.transitions or {}).items()},
        "transitions_path": bundle.transitions_path,
        "memory_path": bundle.memory_path,
        "video_path": bundle.video_path,
    }


def bundle_from_state_dict(d: dict[str, Any]) -> EvalRolloutBundle:
    ver = d.get("bundle_version", BUNDLE_VERSION)
    meta = EvalBundleMeta(**d["meta"])
    md = d["metrics"]
    if ver == "1":
        metrics = EvalBundleMetrics(
            mean_total_reward=0.0,
            std_total_reward=0.0,
            success_rate=float(md.get("success_rate", 0.0)),
            per_rollout_total_reward=[],
            per_rollout_success=list(md.get("per_rollout_success", [])),
        )
    elif ver == BUNDLE_VERSION:
        metrics = EvalBundleMetrics(**md)
    else:
        raise ValueError(f"Unsupported bundle version {ver!r}, expected {BUNDLE_VERSION!r} or '1'")
    rollouts = [_rollout_from_dict(x) for x in d["rollouts"]]
    tr = d.get("transitions") or {}
    transitions = {k: v for k, v in tr.items()} if tr else None
    return EvalRolloutBundle(
        meta=meta,
        metrics=metrics,
        reference=d["reference"],
        rollouts=rollouts,
        transitions=transitions,
        transitions_path=d.get("transitions_path"),
        memory_path=d.get("memory_path"),
        video_path=d.get("video_path"),
    )


def save_bundle(bundle: EvalRolloutBundle, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle_to_state_dict(bundle), p)


def load_bundle(path: str | Path) -> EvalRolloutBundle:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict in {path}, got {type(d)}")
    return bundle_from_state_dict(d)
