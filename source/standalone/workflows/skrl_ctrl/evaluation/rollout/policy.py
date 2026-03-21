"""Policy actions for rollout (deterministic vs stochastic, train vs eval)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

# region agent log
_act_rollout_logged = False
_DBG_LOG_POLICY = Path(__file__).resolve().parents[6] / "runs" / "eval_batch_debug.ndjson"


def _agent_log_policy(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "timestamp": int(time.time() * 1000),
        "location": "evaluation/rollout/policy.py:act_for_rollout",
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": "post-fix",
    }
    try:
        _DBG_LOG_POLICY.parent.mkdir(parents=True, exist_ok=True)
        with open(_DBG_LOG_POLICY, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# endregion


def act_for_rollout(
    agent: Any,
    obs_policy: torch.Tensor,
    *,
    stochastic: bool,
    train_mode: bool,
) -> torch.Tensor:
    """Return actions ``(num_envs, action_dim)`` for vectorized observations.

    Uses ``agent.policy`` with optional ``explore=`` when supported; otherwise falls back to
    ``outputs['mean_actions']`` for deterministic behaviour.
    """
    policy = agent.policy
    if train_mode:
        policy.train()
    else:
        policy.eval()

    for name in ("critic_1", "critic_2", "target_critic_1", "target_critic_2", "phi", "frozen_phi", "mu", "theta"):
        m = getattr(agent, name, None)
        if m is not None:
            m.train(train_mode) if train_mode else m.eval()

    pre = getattr(agent, "_state_preprocessor", None)
    if callable(pre):
        states = pre(obs_policy, train=train_mode)
    else:
        states = obs_policy

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        try:
            out = policy.act({"states": states}, role="policy", explore=stochastic)
        except TypeError:
            out = policy.act({"states": states}, role="policy")

    if isinstance(out, tuple) and len(out) >= 3:
        actions, _logp, outputs = out[0], out[1], out[2]
    else:
        actions = out
        outputs = {}

    if not stochastic and isinstance(outputs, dict) and "mean_actions" in outputs:
        actions = outputs["mean_actions"]

    return actions
