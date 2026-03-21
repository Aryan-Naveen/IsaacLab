"""Vectorized rollout until each env completes one episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from evaluation.bundle.schema import EvalBundleMeta, EvalBundleMetrics, EvalRolloutBundle, RolloutRecord
from evaluation.memory_export.skrl_buffer import append_transition_batch, fill_random_memory_from_transition_dict, save_memory_pt
from evaluation.metrics.trajectory import aggregate_return_metrics, trajectory_rmse
from evaluation.rollout.policy import act_for_rollout


def find_trajectory_env(env: Any) -> Any:
    """Return the **innermost** env with trajectory markers.

    Outer Gymnasium/skrl wrappers may forward attributes to the Isaac env; taking the first match
    makes ``base.lvl`` / ``base.cfg`` resolve via the wrapper and triggers deprecation warnings.
    """
    cur: Any = env
    last_match: Any | None = None
    seen: set[int] = set()
    for _ in range(64):
        cid = id(cur)
        if cid in seen:
            break
        seen.add(cid)
        if hasattr(cur, "_desired_trajectory_w") and hasattr(cur, "_robot"):
            last_match = cur
        nxt = None
        if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
            nxt = cur.unwrapped
        elif hasattr(cur, "env"):
            nxt = cur.env
        elif hasattr(cur, "_env"):
            nxt = cur._env
        if nxt is None:
            break
        cur = nxt
    if last_match is None:
        raise RuntimeError("Could not find QuadcopterTrajectoryEnv in wrapper chain")
    return last_match


def _policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def run_b_rollouts(
    env: Any,
    agent: Any,
    *,
    task_gym_id: str,
    checkpoint_path: str,
    success_radius_m: float,
    success_criterion: str = "combined_default",
    max_steps: int | None = None,
    seed: int = 0,
    init_noise_enabled: bool = False,
    init_noise_config: dict[str, Any] | None = None,
    policy_stochastic: bool = False,
    policy_train_mode: bool = False,
    predefined_task_coeff: list[list[float]] | None = None,
    save_memory_path: str | None = None,
    save_transitions_path: str | None = None,
) -> EvalRolloutBundle:
    """Run until each parallel env finishes one episode; compute metrics and optional replay buffer.

    Per-rollout ``success`` and aggregate ``success_rate`` are **survival-only**: ``not crashed``,
    where ``crashed`` is whether the env reported ``terminated`` (death) before episode end.
    ``success_radius_m`` / ``success_criterion`` are only recorded in bundle metadata.
    """
    device = env.device
    num_envs = env.num_envs
    base = find_trajectory_env(env)

    if max_steps is None:
        max_steps = int(base.max_episode_length) + 64

    obs, _ = env.reset()
    prev_policy = _policy_obs(obs)

    refs = [base._desired_trajectory_w[i].clone() for i in range(num_envs)]
    positions: list[list[torch.Tensor]] = [[] for _ in range(num_envs)]
    for i in range(num_envs):
        positions[i].append(base._robot.data.root_pos_w[i].clone())

    completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ever_died = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)

    trans_lists: dict[str, list[torch.Tensor]] = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
    }

    step_count = 0
    while not completed.all() and step_count < max_steps:
        active = ~completed
        actions = act_for_rollout(
            agent,
            prev_policy,
            stochastic=policy_stochastic,
            train_mode=policy_train_mode,
        )
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        next_policy = _policy_obs(next_obs)

        # skrl/Isaac may return (B,) or (B,1); broadcasting (B,1) & (B,) breaks ever_died shape.
        terminated = terminated.reshape(-1)
        truncated = truncated.reshape(-1)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        ever_died = ever_died | (terminated.bool() & active)

        for i in range(num_envs):
            if not active[i]:
                continue
            episode_returns[i] = episode_returns[i] + rewards[i, 0]
            positions[i].append(base._robot.data.root_pos_w[i].clone())

            trans_lists["states"].append(prev_policy[i].detach().cpu())
            trans_lists["actions"].append(actions[i].detach().cpu())
            trans_lists["rewards"].append(rewards[i, 0].detach().cpu())
            trans_lists["next_states"].append(next_policy[i].detach().cpu())
            trans_lists["dones"].append((terminated[i] | truncated[i]).detach().cpu().to(dtype=torch.float32))

            if terminated[i] or truncated[i]:
                completed[i] = True

        prev_policy = next_policy
        step_count += 1

    if not completed.all():
        import warnings

        warnings.warn(
            f"run_b_rollouts: not all envs finished within max_steps={max_steps} "
            f"(completed={completed.tolist()})",
            RuntimeWarning,
            stacklevel=2,
        )

    rollouts: list[RolloutRecord] = []
    total_rewards: list[float] = []
    succs: list[bool] = []

    for i in range(num_envs):
        pos_i = torch.stack(positions[i], dim=0)
        ref_i = refs[i]
        rmse = trajectory_rmse(pos_i, ref_i, xy_only=False)
        crashed = bool(ever_died[i].item())
        succ = not crashed
        tr = float(episode_returns[i].item())
        rollouts.append(
            RolloutRecord(
                positions=pos_i.detach().cpu(),
                rmse=rmse,
                total_reward=tr,
                success=succ,
                crashed=crashed,
                episode_length=int(pos_i.shape[0]),
            )
        )
        total_rewards.append(tr)
        succs.append(succ)

    mean_total_reward, std_total_reward, success_rate = aggregate_return_metrics(total_rewards, succs)

    ref_tensor = torch.stack(refs, dim=0)

    transitions: dict[str, torch.Tensor] | None = None
    if trans_lists["states"]:
        transitions = {
            "states": torch.stack(trans_lists["states"], dim=0),
            "actions": torch.stack(trans_lists["actions"], dim=0),
            "rewards": torch.stack(trans_lists["rewards"], dim=0).unsqueeze(-1),
            "next_states": torch.stack(trans_lists["next_states"], dim=0),
            "dones": torch.stack(trans_lists["dones"], dim=0).unsqueeze(-1),
        }

    memory_path_out: str | None = None
    transitions_path_out: str | None = None

    if save_transitions_path and transitions is not None:
        p = Path(save_transitions_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(transitions, p)
        transitions_path_out = str(p)

    if save_memory_path and transitions is not None:
        mem = fill_random_memory_from_transition_dict(transitions, device, agent)
        p = Path(save_memory_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_memory_pt(mem, p)
        memory_path_out = str(p)

    meta = EvalBundleMeta(
        task_gym_id=task_gym_id,
        checkpoint_path=checkpoint_path,
        B=num_envs,
        success_radius_m=success_radius_m,
        success_criterion=success_criterion,
        predefined_task_coeff=predefined_task_coeff,
        init_noise_enabled=init_noise_enabled,
        init_noise_config=init_noise_config,
        policy_stochastic=policy_stochastic,
        policy_train_mode=policy_train_mode,
        seed=seed,
        max_steps=max_steps,
    )
    metrics = EvalBundleMetrics(
        mean_total_reward=mean_total_reward,
        std_total_reward=std_total_reward,
        success_rate=success_rate,
        per_rollout_total_reward=total_rewards,
        per_rollout_success=succs,
    )

    return EvalRolloutBundle(
        meta=meta,
        metrics=metrics,
        reference=ref_tensor.detach().cpu(),
        rollouts=rollouts,
        transitions=transitions,
        transitions_path=transitions_path_out,
        memory_path=memory_path_out,
        video_path=None,
    )
