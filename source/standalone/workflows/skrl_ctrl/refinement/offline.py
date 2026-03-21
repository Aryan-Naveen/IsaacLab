from __future__ import annotations

import os
import re
from typing import Any, Iterable

# skrl SAC / CTRLSACAgent — ``track_data`` name -> relative W&B path (under ``wandb_prefix``)
_LOSS_WANDB_KEYS: dict[str, str] = {
    "Loss / Policy loss": "offline/actor_loss",
    "Loss / Critic loss": "offline/critic_loss",
    "Loss / Feature loss": "offline/feature_loss",
    "Loss / Entropy loss": "offline/entropy_loss",
}


def _slug_loss_wandb(name: str) -> str:
    slug = name.removeprefix("Loss / ").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return f"offline/loss_{slug}" if slug else "offline/loss"


def _wandb_rel_key(name: str) -> str | None:
    """Map ``track_data`` metric name to a short W&B key (no ``wandb_prefix``)."""
    if not isinstance(name, str):
        return None
    if name.startswith("Loss /"):
        return _LOSS_WANDB_KEYS.get(name) or _slug_loss_wandb(name)
    if name.startswith("Q-network /"):
        rest = name.removeprefix("Q-network /").strip()
        slug = re.sub(r"[^a-z0-9]+", "_", rest.lower()).strip("_")
        return f"offline/q_{slug}" if slug else "offline/q_metric"
    if name.startswith("Advantage /"):
        rest = name.removeprefix("Advantage /").strip()
        slug = re.sub(r"[^a-z0-9]+", "_", rest.lower()).strip("_")
        return f"offline/adv_{slug}" if slug else "offline/adv_metric"
    if name.startswith("Coefficient /"):
        rest = name.removeprefix("Coefficient /").strip()
        slug = re.sub(r"[^a-z0-9]+", "_", rest.lower()).strip("_")
        return f"offline/coeff_{slug}" if slug else "offline/coeff_metric"
    return None


def _offline_step_iter(num_steps: int) -> Iterable[int]:
    if os.environ.get("FINETUNE_NO_TQDM", "").strip() in ("1", "true", "yes"):
        return range(num_steps)
    try:
        from tqdm import tqdm

        return tqdm(range(num_steps), desc="offline steps", unit="upd", leave=False, total=num_steps)
    except ImportError:
        return range(num_steps)


def offline_train_steps(
    agent: Any,
    num_steps: int,
    *,
    timestep_offset: int = 0,
    wandb_run: Any | None = None,
    wandb_step_start: int = 0,
    wandb_prefix: str = "finetune",
) -> list[Any]:
    """Run ``agent._update`` for ``num_steps`` offline gradient steps.

    When ``wandb_run`` is set, temporarily enables ``write_interval`` (cfg + attributes) and wraps
    ``track_data`` so policy/critic losses and (for AWR) Q / advantage scalars are logged to W&B
    **once per gradient update** at ``step=wandb_step_start + i``.

    For **AWR** with ``gradient_steps`` > 1, this temporarily sets ``gradient_steps`` to 1 and runs
    ``num_steps * previous_gradient_steps`` updates so each sub-step gets its own W&B point while
    preserving the total number of gradient updates.

    Configure the agent beforehand, e.g. ``random_timesteps=0``, ``learning_starts=0``.

    Progress bar: ``tqdm`` when installed; disable with env ``FINETUNE_NO_TQDM=1``.
    """
    logs: list[Any] = []
    if wandb_run is None:
        for i in _offline_step_iter(num_steps):
            out = agent._update(timestep=timestep_offset + i, timesteps=timestep_offset + num_steps)
            logs.append(out)
        return logs

    cfg = getattr(agent, "cfg", None)
    use_awr = isinstance(cfg, dict) and bool(cfg.get("use_offline_awr_update"))
    prev_gs = int(getattr(agent, "_gradient_steps", 1) or 1)
    if prev_gs < 1:
        prev_gs = 1
    prev_cfg_gs_present = isinstance(cfg, dict) and "gradient_steps" in cfg
    prev_cfg_gs = cfg["gradient_steps"] if prev_cfg_gs_present else None
    restore_gs = False
    if use_awr and prev_gs > 1:
        agent._gradient_steps = 1
        if isinstance(cfg, dict):
            cfg["gradient_steps"] = 1
        restore_gs = True
        num_steps_effective = num_steps * prev_gs
    else:
        num_steps_effective = num_steps

    exp = agent.cfg.setdefault("experiment", {})
    prev_wi = int(exp.get("write_interval", 0))
    exp["write_interval"] = 1
    prev_agent_wi: dict[str, Any] = {}
    for attr in ("write_interval", "_write_interval"):
        if hasattr(agent, attr):
            prev_agent_wi[attr] = getattr(agent, attr)
            setattr(agent, attr, 1)

    buf: dict[str, float] = {}
    orig_track = agent.track_data

    def _capture_track(name: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        rel = _wandb_rel_key(name)
        if rel is not None:
            try:
                buf[rel] = float(value.item()) if hasattr(value, "item") else float(value)
            except (TypeError, ValueError):
                pass
        try:
            return orig_track(name, value, *args, **kwargs)
        except TypeError:
            return orig_track(name, value)

    agent.track_data = _capture_track  # type: ignore[method-assign]

    try:
        for i in _offline_step_iter(num_steps_effective):
            buf.clear()
            out = agent._update(
                timestep=timestep_offset + i,
                timesteps=timestep_offset + num_steps_effective,
            )
            logs.append(out)
            if buf:
                row: dict[str, float | int] = {
                    f"{wandb_prefix}/{k}": v for k, v in buf.items()
                }
                wandb_run.log(row, step=wandb_step_start + i)
    finally:
        exp["write_interval"] = prev_wi
        for attr, val in prev_agent_wi.items():
            setattr(agent, attr, val)
        agent.track_data = orig_track  # type: ignore[method-assign]
        if restore_gs:
            agent._gradient_steps = prev_gs
            if isinstance(cfg, dict):
                if prev_cfg_gs_present:
                    cfg["gradient_steps"] = prev_cfg_gs
                else:
                    cfg.pop("gradient_steps", None)

    return logs
