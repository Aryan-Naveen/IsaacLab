"""Shared builders for Isaac Lab envs, CTRLSAC, and SAC (used by train / eval / refinement)."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from agents.ctrlsac_agent import CTRLSACAgent
from agents.sac_offline_awr_agent import OfflineAWRSAC
from networks.actor import StochasticActor
from networks.critic import Critic, SACCritic
from networks.feature import Mu, Phi, Theta
from utils.utils import load_isaaclab_env


def isaaclab_repo_root() -> Path:
    """Ascend from ``factory.py`` until a directory contains ``source/standalone`` (Isaac Lab repo root)."""
    p = Path(__file__).resolve()
    for d in p.parents:
        if (d / "source" / "standalone").is_dir():
            return d
    return p.parents[4]


def ensure_record_video_dir(video_folder: str | Path) -> str:
    """Create a folder for :class:`gymnasium.wrappers.RecordVideo`, with a temp-dir fallback.

    ``runs/torch/...`` under the repo can raise ``FileNotFoundError`` when ``runs`` is a broken
    symlink or the bind mount omits that path.

    Set env ``ISAACLAB_SKRL_VIDEO_ROOT`` to a writable directory to mirror paths under the repo
    root (same relative path as under ``isaaclab_repo_root()``).
    """
    raw = Path(video_folder).expanduser()
    override = os.environ.get("ISAACLAB_SKRL_VIDEO_ROOT", "").strip()
    if override:
        root = isaaclab_repo_root()
        try:
            primary = Path(override) / raw.resolve().relative_to(root.resolve())
        except ValueError:
            primary = Path(override) / raw.name
    else:
        primary = raw

    try:
        os.makedirs(primary, mode=0o755, exist_ok=True)
        return str(primary.resolve())
    except OSError as exc:
        tail_parts = raw.parts[-6:] if len(raw.parts) >= 6 else raw.parts
        alt = Path(tempfile.gettempdir()) / "isaaclab_skrl_recordings" / Path(*tail_parts)
        try:
            os.makedirs(alt, mode=0o755, exist_ok=True)
        except OSError as exc2:
            raise RuntimeError(
                f"Could not create RecordVideo folder {primary} ({exc!s}) nor fallback {alt} ({exc2!s})"
            ) from exc2
        warnings.warn(
            f"Could not mkdir RecordVideo path {primary} ({exc!s}); using {alt}",
            UserWarning,
            stacklevel=2,
        )
        return str(alt.resolve())


def observation_vector_dim(observation_space: gym.Space) -> int:
    """Length of one environment's observation vector (handles ``(D,)`` and batched ``(N, D)`` boxes)."""
    shape = getattr(observation_space, "shape", None)
    if not shape:
        raise ValueError(f"Cannot infer observation size from space: {observation_space!r}")
    if len(shape) == 1:
        return int(shape[0])
    if len(shape) == 2:
        return int(shape[-1])
    raise ValueError(f"Unsupported observation_space.shape {shape}")


def resolve_task_state_dim(
    observation_space: gym.Space,
    drone_state_dim: int,
    *,
    override: int | None = None,
) -> int:
    """CTRL critic task head input size: full observation minus leading drone block (unless overridden)."""
    if override is not None:
        return int(override)
    obs_d = observation_vector_dim(observation_space)
    d_drone = int(drone_state_dim)
    task_d = obs_d - d_drone
    if task_d <= 0:
        raise ValueError(
            f"Derived task_state_dim = obs_dim - drone_state_dim = {obs_d} - {d_drone} is not positive."
        )
    return task_d


# Optional: programmatic refinement cfg (requires Isaac app already configured)
def make_refine_env_cfg(
    predefined_task_coeff: list[list[float]],
    num_envs: int | None = None,
    *,
    initial_pose_xy_radius_max_m: float | None = None,
    initial_pose_xy_when_not_eval: bool | None = None,
    freeze_on_done_in_eval: bool | None = None,
):
    """Return a copy of :class:`QuadcopterTrajectoryRefineEnvCfg` with custom coefficients.

    Use after ``AppLauncher`` / ``import omni.isaac.lab_tasks`` so the config class is available.
    """
    from omni.isaac.lab_tasks.direct.quadcopter.configs import QuadcopterTrajectoryRefineEnvCfg

    cfg = QuadcopterTrajectoryRefineEnvCfg()
    cfg.predefined_task_coeff = predefined_task_coeff
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    if initial_pose_xy_radius_max_m is not None:
        cfg.initial_pose_xy_radius_max_m = float(initial_pose_xy_radius_max_m)
    if initial_pose_xy_when_not_eval is not None:
        cfg.initial_pose_xy_when_not_eval = bool(initial_pose_xy_when_not_eval)
    if freeze_on_done_in_eval is not None:
        cfg.freeze_on_done_in_eval = bool(freeze_on_done_in_eval)
    return cfg


def eval_agent_memory_config() -> dict[str, Any]:
    """SAC/CTRLSAC kwargs matching historical ``cli.py`` eval rollouts (no training)."""
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
    return cfg


def build_loaded_eval_agent(
    env: Any,
    device: torch.device,
    *,
    agent_type: str,
    checkpoint_path: str | Path,
    memory_size: int = 1024,
    actor_hidden_dim: int = 256,
    actor_hidden_depth: int = 3,
    feature_dim: int = 512,
    feature_hidden_dim: int = 1024,
    cdim: int = 512,
    task_state_dim: int | None = None,
    drone_state_dim: int = 13,
) -> tuple[Any, RandomMemory]:
    """Build SAC or CTRLSAC eval agent and load weights (matches ``cli.py`` eval stack)."""
    multitask = agent_type == "CTRLSAC-multi"
    resolved_type = "CTRLSAC" if multitask else agent_type
    task_state_dim = resolve_task_state_dim(
        env.observation_space, drone_state_dim, override=task_state_dim
    )

    agentclasses = {"CTRLSAC": CTRLSACAgent, "SAC": OfflineAWRSAC}
    if resolved_type not in agentclasses:
        raise ValueError(f"Unknown agent_type {agent_type!r} (use SAC, CTRLSAC, CTRLSAC-multi)")

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

    cfg = eval_agent_memory_config()
    cfg["drone_state_dim"] = int(drone_state_dim)
    memory = RandomMemory(memory_size=int(memory_size), num_envs=env.num_envs, device=device)
    AgentClass = agentclasses[resolved_type]
    agent = AgentClass(
        models=models[resolved_type],
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    agent.load(str(checkpoint_path))
    return agent, memory


def load_experiment_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def build_env(
    task_name: str,
    *,
    num_envs: int | None,
    cli_args: list[str] | None = None,
    record_video: bool = False,
    video_folder: str | None = None,
    video_step_trigger: int = 10000,
    video_length: int = 400,
) -> Any:
    cli_args = list(cli_args or [])
    env = load_isaaclab_env(task_name=task_name, num_envs=num_envs, cli_args=cli_args)
    if record_video and video_folder:
        vpath_str = ensure_record_video_dir(video_folder)
        vk = {
            "video_folder": vpath_str,
            "step_trigger": lambda step, _s=video_step_trigger: step % _s == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **vk)
    return wrap_env(env)


def build_ctrlsac_models(
    env: Any,
    device: torch.device,
    *,
    multitask: bool = False,
    actor_hidden_dim: int = 512,
    actor_hidden_depth: int = 3,
    feature_dim: int = 512,
    feature_hidden_dim: int = 1024,
    cdim: int = 512,
    task_state_dim: int | None = None,
    drone_state_dim: int = 13,
) -> dict[str, Any]:
    task_state_dim = resolve_task_state_dim(
        env.observation_space, drone_state_dim, override=task_state_dim
    )
    models: dict[str, Any] = {}
    models["policy"] = StochasticActor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=actor_hidden_dim,
        hidden_depth=actor_hidden_depth,
        log_std_bounds=[-5.0, 2.0],
        device=device,
    )
    for name in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        models[name] = Critic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            task_state_dim=task_state_dim,
            cdim=cdim,
            multitask=multitask,
            device=device,
        )
    models["phi"] = Phi(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        hidden_dim=feature_hidden_dim,
        drone_state_dim=drone_state_dim,
        multitask=multitask,
        device=device,
    )
    models["frozen_phi"] = Phi(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        hidden_dim=feature_hidden_dim,
        drone_state_dim=drone_state_dim,
        multitask=multitask,
        device=device,
    )
    models["theta"] = Theta(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        device=device,
    )
    models["mu"] = Mu(
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_dim=feature_dim,
        hidden_dim=feature_hidden_dim,
        drone_state_dim=drone_state_dim,
        multitask=multitask,
        device=device,
    )
    return models


def build_ctrlsac_agent(
    env: Any,
    memory: RandomMemory,
    device: torch.device,
    *,
    multitask: bool = False,
    finetune: bool = False,
    task_name: str = "",
    ckpt_path: str | None = None,
    memory_path: str | None = None,
    exp: dict[str, Any] | None = None,
) -> CTRLSACAgent:
    exp = exp or {}
    drone_sd = int(exp.get("drone_state_dim", 13))
    task_override = exp.get("task_state_dim")
    task_override_i = int(task_override) if task_override is not None else None
    models = build_ctrlsac_models(
        env,
        device,
        multitask=multitask,
        actor_hidden_dim=int(exp.get("actor_hidden_dim", 512)),
        actor_hidden_depth=int(exp.get("actor_hidden_depth", 3)),
        feature_dim=int(exp.get("feature_dim", 512)),
        feature_hidden_dim=int(exp.get("feature_hidden_dim", 1024)),
        cdim=int(exp.get("cdim", 512)),
        task_state_dim=task_override_i,
        drone_state_dim=drone_sd,
    )
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg["gradient_steps"] = int(exp.get("gradient_steps", 1))
    cfg["batch_size"] = int(exp.get("batch_size", 256))
    cfg["discount_factor"] = float(exp.get("discount_factor", 0.99))
    cfg["polyak"] = float(exp.get("polyak", 0.005))
    cfg["actor_learning_rate"] = float(exp.get("actor_learning_rate", 3e-4 if finetune else 1e-4))
    cfg["critic_learning_rate"] = float(exp.get("critic_learning_rate", 3e-4 if finetune else 1e-4))
    cfg["weight_decay"] = float(exp.get("weight_decay", 0))
    cfg["feature_learning_rate"] = float(exp.get("feature_learning_rate", 1e-5 if finetune else 1e-4))
    cfg["random_timesteps"] = float(exp.get("random_timesteps", 0 if finetune else 12e3))
    cfg["learning_starts"] = float(exp.get("learning_starts", 25e3 if finetune else 12e3))
    cfg["grad_norm_clip"] = float(exp.get("grad_norm_clip", 1.0))
    cfg["learn_entropy"] = bool(exp.get("learn_entropy", not finetune))
    cfg["entropy_learning_rate"] = float(exp.get("entropy_learning_rate", 1e-4 if multitask else 1e-5))
    cfg["initial_entropy_value"] = float(exp.get("initial_entropy_value", 0.06 if finetune else 1.0))
    cfg["experiment"]["write_interval"] = int(exp.get("write_interval", 1000))
    cfg["experiment"]["checkpoint_interval"] = int(exp.get("checkpoint_interval", 1000 if finetune else 10000))
    cfg["use_feature_target"] = bool(exp.get("use_feature_target", True))
    cfg["extra_feature_steps"] = int(exp.get("extra_feature_steps", 0))
    # Critic optimizer steps per policy step: ``extra_critic_steps + 1`` (see CTRLSACAgent._update).
    cfg["extra_critic_steps"] = int(exp.get("extra_critic_steps", 5))
    cfg["target_update_period"] = int(exp.get("target_update_period", 1))
    cfg["eval"] = bool(exp.get("eval", False))
    cfg["experiment"]["wandb"] = bool(exp.get("wandb", True))
    _seed = int(exp.get("seed", 42))
    _default_wk: dict[str, Any] = {
        "project": exp.get("wandb_project", "CDC"),
        "name": (exp.get("wandb_name") or f"CTRL-SAC-multitask-{multitask}"),
        "config": {
            "task": task_name,
            "num_envs": getattr(env, "num_envs", None),
            "multitask": multitask,
            "seed": _seed,
        },
    }
    _user_wk = exp.get("wandb_kwargs")
    if _user_wk:
        _merged = {**_default_wk, **_user_wk}
        if isinstance(_user_wk.get("config"), dict):
            _merged["config"] = {**_default_wk["config"], **_user_wk["config"]}
        cfg["experiment"]["wandb_kwargs"] = _merged
    else:
        cfg["experiment"]["wandb_kwargs"] = _default_wk
    cfg["experiment"]["directory"] = exp.get("experiment_directory", f"runs/torch/{task_name}/CTRL-SAC-{multitask}/")
    cfg["alpha"] = float(exp.get("alpha", 1e-3))
    cfg["memory"] = None
    cfg["drone_state_dim"] = drone_sd
    cfg["multitask"] = bool(multitask)
    cfg["policy_phased_learning_rate"] = bool(exp.get("policy_phased_learning_rate", False))
    cfg["policy_phased_lr_phase1"] = float(exp.get("policy_phased_lr_phase1", 1e-6))
    cfg["policy_phased_lr_phase2"] = float(exp.get("policy_phased_lr_phase2", 1e-6))

    if memory_path:
        memory.load(memory_path, format="pt")

    agent = CTRLSACAgent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    if ckpt_path:
        agent.load(ckpt_path)
    _phased = cfg["policy_phased_learning_rate"] and multitask
    agent.policy_optimizer.param_groups[0]["lr"] = (
        cfg["policy_phased_lr_phase1"] if _phased else cfg["actor_learning_rate"]
    )
    agent.critic_optimizer.param_groups[0]['lr'] = cfg['critic_learning_rate']
    agent.feature_optimizer.param_groups[0]['lr'] = cfg['feature_learning_rate']
    
    return agent


def build_sac_models(
    env: Any,
    device: torch.device,
    *,
    actor_hidden_dim: int = 512,
    actor_hidden_depth: int = 3,
) -> dict[str, Any]:
    return {
        "policy": StochasticActor(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_dim=actor_hidden_dim,
            hidden_depth=actor_hidden_depth,
            log_std_bounds=[-5.0, 2.0],
            device=device,
        ),
        "critic_1": SACCritic(env.observation_space, env.action_space, feature_dim=512, device=device),
        "critic_2": SACCritic(env.observation_space, env.action_space, feature_dim=512, device=device),
        "target_critic_1": SACCritic(env.observation_space, env.action_space, feature_dim=512, device=device),
        "target_critic_2": SACCritic(env.observation_space, env.action_space, feature_dim=512, device=device),
    }


def build_sac_agent(
    env: Any,
    memory: RandomMemory,
    device: torch.device,
    *,
    finetune: bool = False,
    task_name: str = "",
    exp: dict[str, Any] | None = None,
) -> OfflineAWRSAC:
    exp = exp or {}
    models = build_sac_models(
        env,
        device,
        actor_hidden_dim=int(exp.get("actor_hidden_dim", 512)),
        actor_hidden_depth=int(exp.get("actor_hidden_depth", 3)),
    )
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg["gradient_steps"] = int(exp.get("gradient_steps", 1))
    cfg["batch_size"] = int(exp.get("batch_size", 256))
    cfg["discount_factor"] = float(exp.get("discount_factor", 0.99))
    cfg["polyak"] = float(exp.get("polyak", 0.005))
    cfg["actor_learning_rate"] = float(exp.get("actor_learning_rate", 1e-4))
    cfg["critic_learning_rate"] = float(exp.get("critic_learning_rate", 1e-4))
    cfg["random_timesteps"] = float(exp.get("random_timesteps", 0 if finetune else 12e3))
    cfg["learning_starts"] = float(exp.get("learning_starts", 25e3 if finetune else 12e3))
    cfg["grad_norm_clip"] = float(exp.get("grad_norm_clip", 0))
    cfg["learn_entropy"] = bool(exp.get("learn_entropy", not finetune))
    cfg["entropy_learning_rate"] = float(exp.get("entropy_learning_rate", 1e-4))
    cfg["initial_entropy_value"] = float(exp.get("initial_entropy_value", 0.06 if finetune else 1.0))
    cfg["experiment"]["write_interval"] = int(exp.get("write_interval", 1000))
    cfg["experiment"]["checkpoint_interval"] = int(exp.get("checkpoint_interval", 1000 if finetune else 10000))
    cfg["experiment"]["directory"] = exp.get("experiment_directory", f"runs/torch/{task_name}/SAC")
    cfg["experiment"]["experiment_name"] = exp.get("experiment_name", f"{task_name}-SAC")
    cfg["experiment"]["wandb"] = bool(exp.get("wandb", True))
    _seed = int(exp.get("seed", 42))
    _default_wk = {
        "project": exp.get("wandb_project", "CDC"),
        "name": (exp.get("wandb_name") or "SAC-baseline"),
        "config": {"task": task_name, "num_envs": getattr(env, "num_envs", None), "seed": _seed},
    }
    _user_wk = exp.get("wandb_kwargs")
    if _user_wk:
        _merged = {**_default_wk, **_user_wk}
        if isinstance(_user_wk.get("config"), dict):
            _merged["config"] = {**_default_wk["config"], **_user_wk["config"]}
        cfg["experiment"]["wandb_kwargs"] = _merged
    else:
        cfg["experiment"]["wandb_kwargs"] = _default_wk
    return OfflineAWRSAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
