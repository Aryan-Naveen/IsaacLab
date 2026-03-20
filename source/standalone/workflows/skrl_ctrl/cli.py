"""CLI entry: ``train-ctrlsac`` | ``train-sac`` | ``eval`` (prepend command or use default ``train-ctrlsac``)."""

from __future__ import annotations

import argparse
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

from factory import (
    build_ctrlsac_agent,
    build_env,
    build_sac_agent,
    load_experiment_json,
)
from utils.utils import load_isaaclab_env

_SUB = frozenset({"train-ctrlsac", "train-sac", "eval"})


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

    from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.envs.wrappers.torch import wrap_env

    from agents.ctrlsac_agent import CTRLSACAgent
    from networks.actor import StochasticActor
    from networks.critic import Critic, SACCritic
    from networks.feature import Mu, Phi, Theta

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
    memory = RandomMemory(memory_size=experiment_length, num_envs=env.num_envs, device=device)

    actor_hidden_dim = 256
    actor_hidden_depth = 3
    feature_dim = 512
    feature_hidden_dim = 1024
    cdim = 512
    task_state_dim = 67
    drone_state_dim = 13
    multitask = agent_type == "CTRLSAC-multi"
    if agent_type == "CTRLSAC-multi":
        agent_type = "CTRLSAC"

    agentclasses = {"CTRLSAC": CTRLSACAgent, "SAC": SAC}
    models: dict = {"SAC": {}, "CTRLSAC": {}}
    models["SAC"]["policy"] = StochasticActor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=actor_hidden_dim,
        hidden_depth=actor_hidden_depth,
        log_std_bounds=[-5.0, 2.0],
        device=device,
    )
    for n in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        models["SAC"][n] = SACCritic(env.observation_space, env.action_space, feature_dim=feature_dim, device=device)

    models["CTRLSAC"]["policy"] = StochasticActor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=actor_hidden_dim,
        hidden_depth=actor_hidden_depth,
        log_std_bounds=[-5.0, 2.0],
        device=device,
    )
    for n in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        models["CTRLSAC"][n] = Critic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            task_state_dim=task_state_dim,
            cdim=cdim,
            multitask=multitask,
            device=device,
        )
    models["CTRLSAC"]["phi"] = Phi(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        hidden_dim=feature_hidden_dim,
        drone_state_dim=drone_state_dim,
        multitask=multitask,
        device=device,
    )
    models["CTRLSAC"]["frozen_phi"] = Phi(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        hidden_dim=feature_hidden_dim,
        drone_state_dim=drone_state_dim,
        multitask=multitask,
        device=device,
    )
    models["CTRLSAC"]["theta"] = Theta(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        device=device,
    )
    models["CTRLSAC"]["mu"] = Mu(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        hidden_dim=feature_hidden_dim,
        drone_state_dim=drone_state_dim,
        multitask=multitask,
        device=device,
    )

    import copy

    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 256
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 1e-4
    cfg["critic_learning_rate"] = 1e-4
    cfg["weight_decay"] = 0
    cfg["feature_learning_rate"] = 5e-5
    cfg["grad_norm_clip"] = 1.0
    cfg["learn_entropy"] = True
    cfg["entropy_learning_rate"] = 1e-4
    cfg["initial_entropy_value"] = 1.0
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 1e6
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["use_feature_target"] = False
    cfg["extra_feature_steps"] = 1
    cfg["target_update_period"] = 1
    cfg["eval"] = True
    cfg["alpha"] = None

    AgentClass = agentclasses[agent_type]
    agent = AgentClass(
        models=models[agent_type],
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    agent.load(f"{folder}/{ckpt}")
    cfg_trainer = {"timesteps": experiment_length, "headless": True}
    env.eval_mode()
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


def main() -> None:
    cmd, argv = _pop_subcommand(sys.argv)
    sys.argv = argv
    if cmd == "train-ctrlsac":
        run_train_ctrlsac()
    elif cmd == "train-sac":
        run_train_sac()
    elif cmd == "eval":
        run_eval()
    else:
        raise RuntimeError(f"Unknown command {cmd}")
