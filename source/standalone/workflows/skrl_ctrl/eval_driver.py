"""Preset-driven eval-one, eval-batch, offline finetune, and online finetune (used by ``cli.py``)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from evaluation.bundle import save_bundle
from evaluation.memory_export.skrl_buffer import fill_random_memory_from_transition_dict
from evaluation.rendering.topdown import apply_follower_viewport, apply_topdown_viewport, wrap_record_video
from evaluation.rollout import find_trajectory_env, run_b_rollouts
from factory import (
    build_ctrlsac_agent,
    build_loaded_eval_agent,
    build_sac_agent,
    load_experiment_json,
    make_refine_env_cfg,
)
from presets import load_coeff_json, pool_indices
from refinement.offline import offline_train_steps
from utils.utils import load_isaaclab_env


def _resolve_against_preset(path_str: str, preset_file: Path | None) -> Path:
    p = Path(path_str).expanduser()
    if p.is_file():
        return p.resolve()
    if preset_file is not None:
        q = (preset_file.parent / p).resolve()
        if q.is_file():
            return q
    q2 = (Path.cwd() / p).resolve()
    if not q2.is_file():
        raise FileNotFoundError(f"Could not resolve path {path_str!r} (preset dir {preset_file})")
    return q2


def _coeff_short_hash(rows: list[list[float]]) -> str:
    raw = json.dumps(rows, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:8]


def _ckpt_stem(name: str) -> str:
    return Path(name).stem


def run_eval_one_from_preset(
    p: dict[str, Any],
    *,
    preset_file: Path | None = None,
    argv_remainder: list[str] | None = None,
) -> None:
    """Single env, follower RecordVideo by default, metrics.json + optional results."""
    _ = argv_remainder
    agent_type = str(p["agent_type"])
    folder = str(Path(p["checkpoint_dir"]).expanduser())
    ckpt = str(p["checkpoint_name"])
    task = str(p.get("gym_task_id", "Isaac-Quadcopter-Refine-Trajectory-Direct-v0"))
    seed = int(p.get("seed", 42))
    spawn_r = float(p.get("spawn_xy_radius_max_m", 0.0))
    video = bool(p.get("video", True))
    follow_h = float(p.get("video_follow_height", 2.0))
    out_root_raw = p.get("output_root", "runs/eval")
    output_root = Path(out_root_raw).expanduser()
    if not output_root.is_absolute() and preset_file is not None:
        output_root = (preset_file.parent / output_root).resolve()
    coeff_path_raw = p.get("coeff_json") or p.get("coeff_pool_json")
    if not coeff_path_raw:
        raise ValueError("eval_one preset requires coeff_json")
    coeff_path = _resolve_against_preset(str(coeff_path_raw), preset_file)
    coeff_rows = load_coeff_json(coeff_path)
    if len(coeff_rows) != 1:
        raise ValueError("eval_one expects exactly one coefficient row in coeff_json")
    freeze = bool(p.get("freeze_on_done_in_eval", False))
    hq = bool(p.get("hq_viewport_render", False))
    max_steps = int(p.get("max_steps", 0))

    set_seed(seed)
    ckpt_stem = _ckpt_stem(ckpt)
    chash = _coeff_short_hash(coeff_rows)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{agent_type}_{ckpt_stem}" / f"coeff_{chash}" / run_ts
    video_dir = run_dir / "video"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    def _refine_cfg():
        return make_refine_env_cfg(
            coeff_rows,
            num_envs=1,
            initial_pose_xy_radius_max_m=spawn_r,
            freeze_on_done_in_eval=freeze,
        )

    cli_args: list[str] = []
    if hq:
        cli_args.extend(["--experience", "isaaclab.python.headless.rendering.hq.kit"])
    if video:
        cli_args.extend(["--video", "--enable_cameras"])

    raw_env = load_isaaclab_env(
        task_name=task,
        num_envs=1,
        cli_args=cli_args,
        show_cfg=False,
        env_cfg_factory=_refine_cfg if "Refine" in task else None,
    )
    if video:
        apply_follower_viewport(
            raw_env,
            env_index=0,
            eye_offset=(0.0, 0.0, follow_h),
            lookat_offset=(0.0, 0.0, 0.0),
        )
        max_steps_v = max_steps if max_steps > 0 else 4096
        raw_env = wrap_record_video(
            raw_env,
            video_folder=str(video_dir),
            video_length=max_steps_v,
            step_trigger=lambda step: step == 0,
        )

    env = wrap_env(raw_env)
    device = env.device
    ckpt_path = Path(folder).expanduser() / ckpt
    agent, _mem = build_loaded_eval_agent(
        env,
        device,
        agent_type=agent_type,
        checkpoint_path=ckpt_path,
        memory_size=1024,
    )
    traj_env = find_trajectory_env(env)
    traj_env.eval_mode()
    max_episode_length = int(traj_env.max_episode_length)
    bundle = run_b_rollouts(
        env,
        agent,
        task_gym_id=task,
        checkpoint_path=str(ckpt_path),
        success_radius_m=float(p.get("success_radius", 0.3)),
        success_criterion=str(p.get("success_criterion", "combined_default")),
        max_steps=max_steps if max_steps > 0 else None,
        seed=seed,
        policy_stochastic=bool(p.get("policy_stochastic", False)),
        policy_train_mode=bool(p.get("policy_train_mode", False)),
        predefined_task_coeff=coeff_rows,
    )
    if video and hasattr(raw_env, "close_video_recorder"):
        raw_env.close_video_recorder()

    base = traj_env
    results_payload = getattr(base, "results", {})
    r0 = bundle.rollouts[0]
    run_abs = str(run_dir.resolve())
    metrics = {
        "mean_total_reward": bundle.metrics.mean_total_reward,
        "std_total_reward": bundle.metrics.std_total_reward,
        "crash_rate": 1.0 - bundle.metrics.success_rate,
        "success_rate": bundle.metrics.success_rate,
        "seed": seed,
        "spawn_xy_radius_max_m": spawn_r,
        "checkpoint": str(ckpt_path),
        "coeff_json": str(coeff_path.resolve()),
        # Explicit Legendre / basis coefficients (same layout as predefined_task_coeff; one row for eval_one).
        "trajectory_coefficients": coeff_rows,
        "coefficient_row": coeff_rows[0],
        "artifact_dir": run_abs,
        "max_episode_length": max_episode_length,
        "episode_length_steps": r0.episode_length,
        "episode_total_reward": r0.total_reward,
        "crashed": r0.crashed,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(results_payload, run_dir / "results.pth")
    print(
        f"[eval-one] rollout: {r0.episode_length} steps (env max_episode_length={max_episode_length}), "
        f"reward={r0.total_reward:.4f}, crashed={r0.crashed}",
        flush=True,
    )
    print(f"[eval-one] ARTIFACT_DIR={run_abs}", flush=True)
    print(f"[eval-one] metrics -> {run_dir / 'metrics.json'}", flush=True)
    if video:
        print(f"[eval-one] video_dir -> {video_dir}", flush=True)


def run_eval_batch_from_preset(
    p: dict[str, Any],
    *,
    preset_file: Path | None = None,
    argv_remainder: list[str] | None = None,
) -> None:
    _ = argv_remainder
    agent_type = str(p["agent_type"])
    folder = str(Path(p["checkpoint_dir"]).expanduser())
    ckpt = str(p["checkpoint_name"])
    task = str(p.get("gym_task_id", "Isaac-Quadcopter-Refine-Trajectory-Direct-v0"))
    B = int(p["B"])
    seed = int(p.get("seed", 42))
    spawn_r = float(p.get("spawn_xy_radius_max_m", 0.0))
    out_raw = p["output_bundle"]
    out_path = Path(out_raw).expanduser()
    if not out_path.is_absolute() and preset_file is not None:
        out_path = (preset_file.parent / out_path).resolve()
    # CLI --coeff_json merges as coeff_json; preset often sets coeff_pool_json — override must win.
    coeff_pool_raw = p.get("coeff_json") or p.get("coeff_pool_json")
    if not coeff_pool_raw:
        raise ValueError("eval_batch preset requires coeff_pool_json (or coeff_json as a list of rows)")
    coeff_pool_path = _resolve_against_preset(str(coeff_pool_raw), preset_file)
    pool = load_coeff_json(coeff_pool_path)
    pool_size = len(pool)
    if pool_size < 1:
        raise ValueError("Coefficient pool is empty")
    video = bool(p.get("video", False))
    topdown = bool(p.get("record_topdown_video", False))
    follow = bool(p.get("record_follow_video", False))
    hq = bool(p.get("hq_viewport_render", False))
    max_steps = int(p.get("max_steps", 0))
    follow_h = float(p.get("video_follow_height", 2.0))
    eye_z = float(p.get("video_eye_z", 10.0))

    set_seed(seed)
    sequential_pool = bool(p.get("sequential_pool_indices", False))
    if sequential_pool:
        if B > pool_size:
            raise ValueError(
                f"sequential_pool_indices requires B <= len(coeff pool); got B={B}, pool_size={pool_size}"
            )
        idx_np = None
    else:
        idx_np = pool_indices(B, pool_size, seed)

    def _refine_cfg():
        return make_refine_env_cfg(
            pool,
            num_envs=B,
            initial_pose_xy_radius_max_m=spawn_r,
        )

    cli_args: list[str] = []
    if hq:
        cli_args.extend(["--experience", "isaaclab.python.headless.rendering.hq.kit"])
    if video or topdown or follow:
        cli_args.extend(["--video", "--enable_cameras"])

    raw_env = load_isaaclab_env(
        task_name=task,
        num_envs=B,
        cli_args=cli_args,
        show_cfg=False,
        env_cfg_factory=_refine_cfg if "Refine" in task else None,
    )
    te = find_trajectory_env(raw_env)
    if sequential_pool:
        pool_idx_t = torch.arange(B, device=te.device, dtype=torch.long)
    else:
        assert idx_np is not None
        pool_idx_t = torch.as_tensor(idx_np, device=te.device, dtype=torch.long)
    te.set_per_env_pool_task_indices(pool_idx_t)

    video_path = None
    if topdown:
        apply_topdown_viewport(raw_env, env_index=0, eye_z=eye_z)
        vfolder = str(out_path.parent / "videos_batch")
        max_steps_v = max_steps if max_steps > 0 else 2048
        raw_env = wrap_record_video(raw_env, video_folder=vfolder, video_length=max_steps_v, step_trigger=lambda s: s == 0)
        video_path = vfolder
    elif follow:
        apply_follower_viewport(raw_env, env_index=0, eye_offset=(0.0, 0.0, follow_h), lookat_offset=(0.0, 0.0, 0.0))
        vfolder = str(out_path.parent / "videos_batch")
        max_steps_v = max_steps if max_steps > 0 else 2048
        raw_env = wrap_record_video(raw_env, video_folder=vfolder, video_length=max_steps_v, step_trigger=lambda s: s == 0)
        video_path = vfolder

    env = wrap_env(raw_env)
    device = env.device
    ckpt_path = Path(folder).expanduser() / ckpt
    agent, _mem = build_loaded_eval_agent(env, device, agent_type=agent_type, checkpoint_path=ckpt_path, memory_size=1024)
    find_trajectory_env(env).eval_mode()

    bundle = run_b_rollouts(
        env,
        agent,
        task_gym_id=task,
        checkpoint_path=str(ckpt_path),
        success_radius_m=float(p.get("success_radius", 0.3)),
        success_criterion=str(p.get("success_criterion", "combined_default")),
        max_steps=max_steps if max_steps > 0 else None,
        seed=seed,
        policy_stochastic=bool(p.get("policy_stochastic", False)),
        policy_train_mode=bool(p.get("policy_train_mode", False)),
        predefined_task_coeff=pool,
        save_memory_path=p.get("save_memory") or None,
        save_transitions_path=p.get("save_transitions") or None,
    )
    if (topdown or follow) and hasattr(raw_env, "close_video_recorder"):
        raw_env.close_video_recorder()
    if video_path:
        bundle.video_path = video_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, str(out_path))
    summary = {
        "mean_total_reward": bundle.metrics.mean_total_reward,
        "std_total_reward": bundle.metrics.std_total_reward,
        "crash_rate": 1.0 - bundle.metrics.success_rate,
        "success_rate": bundle.metrics.success_rate,
        "B": B,
        "seed": seed,
        "pool_size": pool_size,
        "sequential_pool_indices": sequential_pool,
        "checkpoint": str(ckpt_path),
        "bundle": str(out_path.resolve()),
    }
    metrics_path = out_path.with_name(out_path.name + ".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[eval-batch] mean_total_reward={bundle.metrics.mean_total_reward:.4f} "
        f"crash_rate={summary['crash_rate']:.4f} -> {out_path}"
    )


def run_finetune_from_preset(
    p: dict[str, Any],
    *,
    preset_file: Path | None = None,
    argv_remainder: list[str] | None = None,
) -> None:
    _ = argv_remainder
    agent_type = str(p["agent_type"])
    folder = str(Path(p["checkpoint_dir"]).expanduser())
    ckpt = str(p["checkpoint_name"])
    task = str(p.get("gym_task_id", "Isaac-Quadcopter-Refine-Trajectory-Direct-v0"))
    B = int(p["B"])
    seed = int(p.get("seed", 42))
    spawn_r = float(p.get("spawn_xy_radius_max_m", 1.0))
    num_rounds = int(p.get("num_rounds", 10))
    offline_steps = int(p.get("offline_steps_per_round", 1000))
    out_root_raw = p.get("output_root", "runs/finetune")
    output_root = Path(out_root_raw).expanduser()
    if not output_root.is_absolute() and preset_file is not None:
        output_root = (preset_file.parent / output_root).resolve()
    coeff_raw = p.get("coeff_json")
    if not coeff_raw:
        raise ValueError("finetune preset requires coeff_json (single-task rows)")
    coeff_path = _resolve_against_preset(str(coeff_raw), preset_file)
    coeff_rows = load_coeff_json(coeff_path)
    memory_fill_size = int(p.get("memory_size", 500_000))
    hq = bool(p.get("hq_viewport_render", False))
    max_steps = int(p.get("max_steps", 0))
    batch_size = int(p.get("finetune_batch_size", 256))
    use_wandb = bool(p.get("wandb", False))

    set_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[finetune] {num_rounds} rounds × {offline_steps} offline steps/round, "
        f"B={B}, batch_size≤{batch_size} → {output_root.resolve()}"
    )

    wandb_run = None
    wandb_step = 0
    if use_wandb:
        try:
            import wandb

            wandb_cfg = {
                k: p[k]
                for k in (
                    "agent_type",
                    "B",
                    "num_rounds",
                    "offline_steps_per_round",
                    "spawn_xy_radius_max_m",
                    "seed",
                    "gym_task_id",
                )
                if k in p
            }
            wandb_run = wandb.init(
                project=str(p.get("wandb_project", "ctrlsac-finetune")),
                name=p.get("wandb_name") or None,
                entity=p.get("wandb_entity") or None,
                config=wandb_cfg,
            )
        except Exception as e:
            print(f"[finetune] wandb init failed ({e}); continuing without logging.")
            wandb_run = None

    def _refine_cfg():
        return make_refine_env_cfg(
            coeff_rows,
            num_envs=B,
            initial_pose_xy_radius_max_m=spawn_r,
        )

    cli_args: list[str] = []
    if hq:
        cli_args.extend(["--experience", "isaaclab.python.headless.rendering.hq.kit"])

    # One sim app for all rounds: ``load_isaaclab_env`` starts AppLauncher / Omniverse; a second
    # call in-process typically aborts, which previously made only round_0000 succeed.
    raw_env = load_isaaclab_env(
        task_name=task,
        num_envs=B,
        cli_args=cli_args,
        show_cfg=False,
        env_cfg_factory=_refine_cfg if "Refine" in task else None,
    )
    env = wrap_env(raw_env)
    round_ckpt = Path(folder).expanduser() / ckpt
    try:
        cumulative_offline_step = 0
        for r in range(num_rounds):
            device = env.device
            agent, _mem = build_loaded_eval_agent(
                env,
                device,
                agent_type=agent_type,
                checkpoint_path=round_ckpt,
                memory_size=memory_fill_size,
            )
            agent.cfg["learning_starts"] = 0
            agent.cfg["random_timesteps"] = 0
            agent.cfg["eval"] = False
            agent.eval = False

            offline_algo = str(p.get("offline_algorithm", "awr")).lower()
            if offline_algo == "cql_awr":
                print(
                    "[finetune] offline_algorithm 'cql_awr' is no longer supported; using 'awr' instead."
                )
                offline_algo = "awr"
            if offline_algo == "awr":
                fn = getattr(agent, "configure_offline_finetune", None)
                if callable(fn):
                    fn(
                        critic_lr=float(p.get("offline_critic_learning_rate", 3e-5)),
                        actor_lr=float(p.get("offline_actor_learning_rate", 3e-5)),
                        awr_beta=float(p.get("awr_beta", 1.0)),
                        awr_weight_max=float(p.get("awr_weight_max", 20.0)),
                        awr_num_action_samples=int(p.get("awr_num_action_samples", 8)),
                    )

            find_trajectory_env(env).eval_mode()
            bundle = run_b_rollouts(
                env,
                agent,
                task_gym_id=task,
                checkpoint_path=str(round_ckpt),
                success_radius_m=float(p.get("success_radius", 0.3)),
                success_criterion=str(p.get("success_criterion", "combined_default")),
                max_steps=max_steps if max_steps > 0 else None,
                seed=seed + r,
                policy_stochastic=bool(p.get("policy_stochastic", False)),
                policy_train_mode=bool(p.get("policy_train_mode", False)),
                predefined_task_coeff=coeff_rows,
            )
            if bundle.transitions is None:
                raise RuntimeError(f"Round {r}: no transitions collected")
            n = bundle.transitions["states"].shape[0]
            bs = min(batch_size, max(1, n))
            agent.cfg["batch_size"] = int(bs)
            agent._batch_size = int(bs)
            agent.memory = fill_random_memory_from_transition_dict(bundle.transitions, device, agent)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "finetune/collection/mean_total_reward": bundle.metrics.mean_total_reward,
                        "finetune/collection/std_total_reward": bundle.metrics.std_total_reward,
                        "finetune/collection/crash_rate": 1.0 - bundle.metrics.success_rate,
                        "finetune/collection/success_rate": bundle.metrics.success_rate,
                        "finetune/collection/num_transitions": float(n),
                        "finetune/collection/round": float(r),
                    },
                    step=wandb_step,
                )
                wandb_step += 1
            off_logs = offline_train_steps(
                agent,
                offline_steps,
                timestep_offset=cumulative_offline_step,
                wandb_run=wandb_run,
                wandb_step_start=wandb_step,
                wandb_prefix="finetune",
            )
            cumulative_offline_step += len(off_logs)
            if wandb_run is not None:
                wandb_step += len(off_logs)
            round_dir = output_root / f"round_{r:04d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_ckpt = round_dir / "agent_finetuned.pt"
            agent.save(str(round_ckpt))
            with open(round_dir / "round_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round": r,
                        "mean_total_reward": bundle.metrics.mean_total_reward,
                        "crash_rate": 1.0 - bundle.metrics.success_rate,
                        "transitions": n,
                        "checkpoint": str(round_ckpt.resolve()),
                    },
                    f,
                    indent=2,
                )
            print(
                f"[finetune] round {r + 1}/{num_rounds} saved {round_ckpt} "
                f"(n={n}, mean_R={bundle.metrics.mean_total_reward:.4f})"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


_FINETUNE_ONLINE_SKIP_KEYS = frozenset(
    {
        "mode",
        "agent_type",
        "checkpoint_dir",
        "checkpoint_name",
        "gym_task_id",
        "coeff_json",
        "trajectory_json",
        "B",
        "timesteps",
        "spawn_xy_radius_max_m",
        "memory_size",
        "output_root",
        "seed",
        "hq_viewport_render",
        "video",
        "video_follow_height",
        "video_interval",
        "video_length",
        "memory_load_path",
    }
)


def run_finetune_online_from_preset(
    p: dict[str, Any],
    *,
    preset_file: Path | None = None,
    argv_remainder: list[str] | None = None,
) -> None:
    """Load checkpoint and run on-policy ``SequentialTrainer`` on a Refine (or other) task."""
    _ = argv_remainder
    mode = str(p.get("mode", "finetune-online"))
    if mode != "finetune-online":
        raise ValueError(f"finetune-online driver expects mode 'finetune-online', got {mode!r}")

    agent_type = str(p["agent_type"])
    folder = str(Path(p["checkpoint_dir"]).expanduser())
    ckpt = str(p["checkpoint_name"])
    task = str(p.get("gym_task_id", "Isaac-Quadcopter-Refine-Trajectory-Direct-v0"))
    B = int(p["B"])
    seed = int(p.get("seed", 42))
    timesteps = int(p.get("timesteps", 50_000))
    spawn_r = float(p.get("spawn_xy_radius_max_m", 0.0))
    memory_size = int(p.get("memory_size", 100_000))
    hq = bool(p.get("hq_viewport_render", False))
    video = bool(p.get("video", True))
    follow_h = float(p.get("video_follow_height", 2.0))
    video_interval = int(p.get("video_interval", 1000))
    video_length = int(p.get("video_length", 400))

    coeff_raw = p.get("coeff_json") or p.get("trajectory_json")
    if not coeff_raw:
        raise ValueError("finetune-online preset requires coeff_json or trajectory_json")
    coeff_path = _resolve_against_preset(str(coeff_raw), preset_file)
    coeff_rows = load_coeff_json(coeff_path)

    out_root_raw = p.get("output_root", "runs/finetune-online")
    output_root = Path(out_root_raw).expanduser()
    if not output_root.is_absolute() and preset_file is not None:
        output_root = (preset_file.parent / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    skrl_dir = Path(__file__).resolve().parent
    exp = load_experiment_json(skrl_dir / "configs" / "default_experiment.json")
    for k, v in p.items():
        if k in _FINETUNE_ONLINE_SKIP_KEYS:
            continue
        exp[k] = v
    exp["experiment_directory"] = str(output_root)

    set_seed(seed)
    ckpt_path = Path(folder).expanduser() / ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    memory_load_path: str | None = None
    mlp = p.get("memory_load_path") or p.get("memory_path")
    if mlp is not None and str(mlp).strip():
        memory_load_path = str(_resolve_against_preset(str(mlp).strip(), preset_file))
        if not Path(memory_load_path).is_file():
            raise FileNotFoundError(f"memory_load_path not found: {memory_load_path}")
        print(f"[finetune-online] replay buffer: {memory_load_path}")

    def _refine_cfg():
        return make_refine_env_cfg(
            coeff_rows,
            num_envs=B,
            initial_pose_xy_radius_max_m=spawn_r,
        )

    cli_args: list[str] = []
    if hq:
        cli_args.extend(["--experience", "isaaclab.python.headless.rendering.hq.kit"])
    if video:
        cli_args.extend(["--video", "--enable_cameras"])

    print(
        f"[finetune-online] agent={agent_type} task={task} B={B} timesteps={timesteps} "
        f"-> {output_root.resolve()}"
    )

    raw_env = load_isaaclab_env(
        task_name=task,
        num_envs=B,
        cli_args=cli_args,
        show_cfg=False,
        env_cfg_factory=_refine_cfg if "Refine" in task else None,
    )
    if video:
        apply_follower_viewport(
            raw_env,
            env_index=0,
            eye_offset=(0.0, 0.0, follow_h),
            lookat_offset=(0.0, 0.0, 0.0),
        )
        video_dir = output_root / "videos" / "finetune-online"
        video_dir.mkdir(parents=True, exist_ok=True)
        raw_env = wrap_record_video(
            raw_env,
            video_folder=str(video_dir),
            video_length=video_length,
            step_trigger=lambda step, _every=video_interval: step % _every == 0,
        )
        print(
            f"[finetune-online] follower camera (eval-one style), videos every {video_interval} steps "
            f"-> {video_dir}"
        )
    env = wrap_env(raw_env)
    device = env.device
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    if agent_type == "SAC":
        agent = build_sac_agent(
            env,
            memory,
            device,
            finetune=True,
            task_name=task,
            exp=exp,
        )
        agent.load(str(ckpt_path))
        # apply_cfg_learning_rates_to_optimizers(agent)
    elif agent_type in ("CTRLSAC", "CTRLSAC-multi"):
        multitask = agent_type == "CTRLSAC-multi"
        agent = build_ctrlsac_agent(
            env,
            memory,
            device,
            multitask=multitask,
            finetune=True,
            task_name=task,
            ckpt_path=str(ckpt_path),
            memory_path=memory_load_path,
            exp=exp,
        )
    else:
        raise ValueError(f"Unsupported agent_type for finetune-online: {agent_type!r}")

    cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    exp_dir = agent.cfg["experiment"]["directory"]
    memory.save(exp_dir, format="pt")
    try:
        trainer.train()
    finally:
        if video and hasattr(raw_env, "close_video_recorder"):
            try:
                raw_env.close_video_recorder()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
