"""CLI entry: train / eval / eval-one / eval-batch / finetune (prepend subcommand or default ``train-ctrlsac``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.utils.dict import print_dict
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from evaluation.bundle import save_bundle
from evaluation.rollout import find_trajectory_env, run_b_rollouts
from evaluation.rendering.topdown import apply_follower_viewport, apply_topdown_viewport, wrap_record_video
from factory import (
    build_ctrlsac_agent,
    build_env,
    build_loaded_eval_agent,
    build_sac_agent,
    load_experiment_json,
    make_refine_env_cfg,
)
from presets import load_coeff_json
from utils.utils import load_isaaclab_env

_SUB = frozenset({"train-ctrlsac", "train-sac", "eval", "eval-batch", "eval-one", "finetune"})


def _checkpoint_path_or_raise(folder: str, ckpt: str) -> Path:
    """Resolve ``folder/ckpt`` and raise with a helpful message if missing."""
    folder_p = Path(folder).expanduser()
    ckpt_p = folder_p / ckpt
    if ckpt_p.is_file():
        return ckpt_p
    pt = sorted(p.name for p in folder_p.glob("*.pt")) if folder_p.is_dir() else []
    hint = (
        f"Checkpoint not found: {ckpt_p}\n"
        f"  folder: {folder_p} ({'exists' if folder_p.is_dir() else 'missing or not a directory'})\n"
        f"  available .pt files: {pt if pt else '(none)'}"
    )
    raise FileNotFoundError(hint)


def _pop_subcommand(argv: list[str]) -> tuple[str, list[str]]:
    if len(argv) > 1 and argv[1] in _SUB:
        return argv[1], [argv[0]] + argv[2:]
    return "train-ctrlsac", argv


def run_train_ctrlsac(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(description="Train CTRLSAC on quadcopter trajectory task.")
    parser.add_argument("--env_version", type=str, default="legtrain")
    parser.add_argument("--multitask", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path for finetuning.")
    parser.add_argument(
        "--experiment_json",
        type=str,
        default="",
        help="Optional JSON file merged into agent hyperparameters.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv[1:])

    finetune = args.env_version == "legtrain-finetune"
    task_name = f"Isaac-Quadcopter-{args.env_version}-Trajectory-Direct-v0"
    num_envs = 1 if finetune else 32
    set_seed(42)

    exp = load_experiment_json(Path(__file__).resolve().parent / "configs" / "default_experiment.json")
    exp.update(load_experiment_json(args.experiment_json or None))

    video_folder = os.path.join(f"runs/torch/{args.env_version}/", "videos", "train", "CTRLSAC")
    env = build_env(
        task_name,
        num_envs=num_envs,
        cli_args=["--video"],
        record_video=True,
        video_folder=video_folder,
    )
    device = env.device
    memory = RandomMemory(memory_size=int(1e5), num_envs=env.num_envs, device=device)

    agent = build_ctrlsac_agent(
        env,
        memory,
        device,
        multitask=args.multitask,
        finetune=finetune,
        task_name=task_name,
        ckpt_path=args.ckpt or None,
        exp=exp,
    )

    timesteps = int(3e5) if not finetune else int(5e4)
    cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    exp_dir = agent.cfg["experiment"]["directory"]
    memory.save(exp_dir, format="pt")
    trainer.train()


def run_train_sac(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(description="Train SAC baseline on quadcopter trajectory task.")
    parser.add_argument("--env_version", type=str, default="legtrain")
    parser.add_argument("--experiment_json", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv[1:])

    finetune = args.env_version == "legtrain-finetune"
    task_name = f"Isaac-Quadcopter-{args.env_version}-Trajectory-Direct-v0"
    num_envs = 1 if finetune else 32
    set_seed(42)

    exp = load_experiment_json(Path(__file__).resolve().parent / "configs" / "default_experiment.json")
    exp.update(load_experiment_json(args.experiment_json or None))
    video_folder = os.path.join(f"runs/torch/{args.env_version}/", "videos", "train", "SAC")
    vk = {
        "video_folder": video_folder,
        "step_trigger": lambda step: step % 10000 == 0,
        "video_length": 400,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(vk, nesting=4)
    env = load_isaaclab_env(task_name=task_name, num_envs=num_envs, cli_args=["--video", "--enable_cameras"])
    env = gym.wrappers.RecordVideo(env, **vk)
    from skrl.envs.wrappers.torch import wrap_env

    env = wrap_env(env)
    device = env.device
    memory = RandomMemory(memory_size=int(1e5), num_envs=env.num_envs, device=device)
    agent = build_sac_agent(env, memory, device, finetune=finetune, task_name=task_name, exp=exp)
    timesteps = int(5e5) if not finetune else int(25e4)
    cfg_trainer = {"timesteps": timesteps, "headless": True, "environment_info": "log"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    exp_dir = agent.cfg["experiment"]["directory"]
    memory.save(exp_dir, format="pt")
    trainer.train()


def run_eval(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    experiments = {
        "legeval": [515, False, 100],
        "legeval-predef": [515, False, 1],
        "legood": [500, True, 1],
        "OOD": [500, True, 1],
        "legtrain": [3000, True, 1],
    }
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--experiment", type=str, default="legeval-predef", choices=list(experiments.keys()))
    parser.add_argument(
        "--agent_type",
        type=str,
        default="CTRLSAC-multi",
        choices=["CTRLSAC-multi", "SAC", "CTRLSAC"],
    )
    parser.add_argument("--ckpt", type=str, default="-1")
    parser.add_argument("--folder", type=str, required=True, help="Directory containing checkpoint.")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv[1:])

    from skrl.envs.wrappers.torch import wrap_env

    task = args.experiment
    agent_type = args.agent_type
    ckpt = args.ckpt
    folder = args.folder
    set_seed(42)

    experiment_length = experiments[task][0]
    record_video = experiments[task][1]
    folder_name = Path(folder).parts[-4] if len(Path(folder).parts) >= 4 else Path(folder).name
    output_dir = f"runs/experiments/{task}/{agent_type}/{folder_name}"
    num_envs = experiments[task][2]
    video_kwargs = {
        "video_folder": os.path.join(output_dir, "videos"),
        "step_trigger": lambda step: step % 10000 == 0,
        "video_length": experiment_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = load_isaaclab_env(
        task_name=f"Isaac-Quadcopter-{task}-Trajectory-Direct-v0",
        num_envs=num_envs,
        cli_args=["--video"],
    )
    if record_video:
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = wrap_env(env)
    device = env.device

    ckpt_path = Path(folder).expanduser() / ckpt
    agent, _memory = build_loaded_eval_agent(
        env,
        device,
        agent_type=agent_type,
        checkpoint_path=ckpt_path,
        memory_size=experiment_length,
    )
    cfg_trainer = {"timesteps": experiment_length, "headless": True}
    find_trajectory_env(env).eval_mode()
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.eval()
    os.makedirs(output_dir, exist_ok=True)
    results_payload = {}
    cur = env
    for _ in range(32):
        if hasattr(cur, "results"):
            results_payload = cur.results
            break
        nxt = getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    out_name = ckpt[:-3] if len(ckpt) > 3 else ckpt
    torch.save(results_payload, f"{output_dir}/{out_name}.pth")


def run_eval_one(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(description="Single-trajectory eval: follower video + metrics.json (preset JSON/YAML).")
    parser.add_argument("--preset", type=str, required=True, help="Path to preset file.")
    parser.add_argument("--folder", type=str, default="", help="Override checkpoint_dir.")
    parser.add_argument("--ckpt", type=str, default="", help="Override checkpoint_name.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--output_root", type=str, default="", help="Override output_root.")
    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args(argv[1:])

    from eval_driver import run_eval_one_from_preset
    from presets import load_preset_file, merge_preset

    p = load_preset_file(args.preset)
    ov: dict = {}
    if args.folder.strip():
        ov["checkpoint_dir"] = args.folder
    if args.ckpt.strip():
        ov["checkpoint_name"] = args.ckpt
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.output_root.strip():
        ov["output_root"] = args.output_root
    run_eval_one_from_preset(merge_preset(p, ov), preset_file=Path(args.preset).resolve())


def run_finetune(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(description="Few-shot offline finetune rounds (preset JSON/YAML).")
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--num_rounds", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args(argv[1:])

    from eval_driver import run_finetune_from_preset
    from presets import load_preset_file, merge_preset

    p = load_preset_file(args.preset)
    ov: dict = {}
    if args.folder.strip():
        ov["checkpoint_dir"] = args.folder
    if args.ckpt.strip():
        ov["checkpoint_name"] = args.ckpt
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.output_root.strip():
        ov["output_root"] = args.output_root
    if args.B is not None:
        ov["B"] = args.B
    if args.num_rounds is not None:
        ov["num_rounds"] = args.num_rounds
    run_finetune_from_preset(merge_preset(p, ov), preset_file=Path(args.preset).resolve())


def run_eval_batch(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(description="Batch eval: B parallel rollouts, metrics, EvalRolloutBundle.")
    parser.add_argument("--preset", type=str, default="", help="JSON/YAML preset (optional; replaces most flags).")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Quadcopter-Refine-Trajectory-Direct-v0",
        help="Gym task id (Refine recommended for custom coeffs).",
    )
    parser.add_argument("--B", type=int, default=4, help="Number of parallel envs (rollouts).")
    parser.add_argument("--folder", type=str, default="", help="Directory containing checkpoint.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint filename (e.g. agent_300000.pt).")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="CTRLSAC",
        choices=["CTRLSAC-multi", "SAC", "CTRLSAC"],
    )
    parser.add_argument(
        "--success_radius",
        type=float,
        default=0.3,
        help="Stored in bundle meta only; success_rate is survival-only (not env terminated/death).",
    )
    parser.add_argument(
        "--success_criterion",
        type=str,
        default="combined_default",
        choices=[
            "combined_default",
            "max_xy_below_radius",
            "mean_xy_below_radius",
            "terminal_xy_below_radius",
            "terminal_xyz_below_radius",
        ],
        help="Stored in bundle meta only; success_rate does not use tracking thresholds.",
    )
    parser.add_argument("--max_steps", type=int, default=0, help="Cap steps per rollout (0 = env max + margin).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coeff_json",
        type=str,
        default="",
        help='JSON path for predefined_task_coeff (e.g. [[0,0,0,0,0,1,0]]). Default: Refine cfg row.',
    )
    parser.add_argument("--out", type=str, default="", help="Output path for EvalRolloutBundle (.pt).")
    parser.add_argument(
        "--spawn_xy_radius_max_m",
        type=float,
        default=0.0,
        help="XY disk spawn around trajectory start (Refine env only).",
    )
    parser.add_argument("--save_transitions", type=str, default="", help="Optional path for transitions dict (.pt).")
    parser.add_argument("--save_memory", type=str, default="", help="Optional path for skrl RandomMemory (.pt).")
    parser.add_argument("--policy_stochastic", action="store_true", help="Sample actions (exploration).")
    parser.add_argument("--policy_train_mode", action="store_true", help="policy.train() for rollout.")
    parser.add_argument("--record_topdown_video", action="store_true", help="Record viewport MP4 (fixed top-down).")
    parser.add_argument(
        "--record_follow_video",
        action="store_true",
        help="Record viewport MP4 with camera following robot root (mutually exclusive with --record_topdown_video).",
    )
    parser.add_argument("--video_env_index", type=int, default=0)
    parser.add_argument("--video_eye_z", type=float, default=10.0)
    parser.add_argument("--video_follow_height", type=float, default=2.0, help="World Z offset above root for follow camera.")
    parser.add_argument("--video_folder", type=str, default="", help="Folder for RecordVideo (default: next to --out).")
    parser.add_argument(
        "--hq_viewport_render",
        action="store_true",
        help=(
            "Load isaaclab.python.headless.rendering.hq.kit (more RTX samples, DLSS Quality). "
            "Use with --headless --enable_cameras; slower per frame. Cannot combine with --experience."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv[1:])

    if args.preset and str(args.preset).strip():
        from eval_driver import run_eval_batch_from_preset
        from presets import load_preset_file, merge_preset

        p = load_preset_file(args.preset)
        ov: dict = {}
        if args.folder.strip():
            ov["checkpoint_dir"] = args.folder
        if args.ckpt.strip():
            ov["checkpoint_name"] = args.ckpt
        if args.out.strip():
            ov["output_bundle"] = args.out
        joined = " ".join(argv)
        if "--seed" in joined:
            ov["seed"] = args.seed
        if "--B" in joined:
            ov["B"] = args.B
        if "--spawn_xy_radius_max_m" in joined:
            ov["spawn_xy_radius_max_m"] = args.spawn_xy_radius_max_m
        run_eval_batch_from_preset(merge_preset(p, ov), preset_file=Path(args.preset).resolve())
        return

    if not args.folder or not args.ckpt or not args.out:
        parser.error("Without --preset, --folder, --ckpt, and --out are required.")

    if args.record_topdown_video and args.record_follow_video:
        parser.error("Use only one of --record_topdown_video or --record_follow_video.")
    if args.hq_viewport_render and str(getattr(args, "experience", "") or "").strip():
        parser.error("--hq_viewport_render cannot be combined with --experience.")

    from skrl.envs.wrappers.torch import wrap_env

    set_seed(args.seed)

    if args.coeff_json:
        predefined_task_coeff = load_coeff_json(args.coeff_json)
    else:
        predefined_task_coeff = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    predefined_task_coeff = [[float(x) for x in row] for row in predefined_task_coeff]

    # Refine cfg must be built *after* AppLauncher (inside load_isaaclab_env); otherwise imports like
    # omni.isaac.core run before the sim app exists.
    env_cfg_factory = None
    if "Refine" in args.task:

        def _refine_cfg():
            return make_refine_env_cfg(
                predefined_task_coeff,
                num_envs=args.B,
                initial_pose_xy_radius_max_m=args.spawn_xy_radius_max_m,
            )

        env_cfg_factory = _refine_cfg

    cli_args: list[str] = []
    if args.hq_viewport_render:
        cli_args.extend(["--experience", "isaaclab.python.headless.rendering.hq.kit"])
    if args.record_topdown_video or args.record_follow_video:
        cli_args.extend(["--video", "--enable_cameras"])

    raw_env = load_isaaclab_env(
        task_name=args.task,
        num_envs=args.B,
        cli_args=cli_args,
        show_cfg=False,
        env_cfg_factory=env_cfg_factory,
    )

    video_path = None
    if args.record_topdown_video:
        apply_topdown_viewport(raw_env, env_index=args.video_env_index, eye_z=args.video_eye_z)
        vfolder = args.video_folder or str(Path(args.out).parent / "videos_batch")
        max_steps_v = args.max_steps if args.max_steps > 0 else 2048
        raw_env = wrap_record_video(
            raw_env,
            video_folder=vfolder,
            video_length=max_steps_v,
            step_trigger=lambda step: step == 0,
        )
        video_path = vfolder
    elif args.record_follow_video:
        apply_follower_viewport(
            raw_env,
            env_index=args.video_env_index,
            eye_offset=(0.0, 0.0, args.video_follow_height),
            lookat_offset=(0.0, 0.0, 0.0),
        )
        vfolder = args.video_folder or str(Path(args.out).parent / "videos_batch")
        max_steps_v = args.max_steps if args.max_steps > 0 else 2048
        raw_env = wrap_record_video(
            raw_env,
            video_folder=vfolder,
            video_length=max_steps_v,
            step_trigger=lambda step: step == 0,
        )
        video_path = vfolder

    env = wrap_env(raw_env)
    device = env.device

    ckpt_path = Path(args.folder).expanduser() / args.ckpt
    agent, _memory = build_loaded_eval_agent(
        env,
        device,
        agent_type=args.agent_type,
        checkpoint_path=ckpt_path,
        memory_size=1024,
    )

    # Match ``run_eval``: trajectory env + generator eval mode (deterministic task cycling, no torch.randint).
    # Call on unwrapped Isaac env — ``env.eval_mode`` via Gymnasium wrapper emits deprecation warnings.
    find_trajectory_env(env).eval_mode()

    max_steps = args.max_steps if args.max_steps > 0 else None

    bundle = run_b_rollouts(
        env,
        agent,
        task_gym_id=args.task,
        checkpoint_path=str(ckpt_path),
        success_radius_m=args.success_radius,
        success_criterion=args.success_criterion,
        max_steps=max_steps,
        seed=args.seed,
        init_noise_enabled=False,
        init_noise_config=None,
        policy_stochastic=args.policy_stochastic,
        policy_train_mode=args.policy_train_mode,
        predefined_task_coeff=predefined_task_coeff,
        save_memory_path=args.save_memory or None,
        save_transitions_path=args.save_transitions or None,
    )
    # RecordVideo with video_length>0 only auto-closes after N frames, not at episode end; flush MP4 here.
    if (args.record_topdown_video or args.record_follow_video) and hasattr(raw_env, "close_video_recorder"):
        raw_env.close_video_recorder()

    if video_path:
        bundle.video_path = video_path

    save_bundle(bundle, args.out)
    print(
        f"[eval-batch] mean_total_reward={bundle.metrics.mean_total_reward:.4f} "
        f"std_total_reward={bundle.metrics.std_total_reward:.4f} -> {args.out}"
    )


def main() -> None:
    cmd, argv = _pop_subcommand(sys.argv)
    sys.argv = argv
    if cmd == "train-ctrlsac":
        run_train_ctrlsac()
    elif cmd == "train-sac":
        run_train_sac()
    elif cmd == "eval":
        run_eval()
    elif cmd == "eval-one":
        run_eval_one()
    elif cmd == "eval-batch":
        run_eval_batch()
    elif cmd == "finetune":
        run_finetune()
    else:
        raise RuntimeError(f"Unknown command {cmd}")
