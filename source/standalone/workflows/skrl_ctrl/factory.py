"""Shared builders for Isaac Lab envs, CTRLSAC, and SAC (used by train / eval / refinement)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from agents.ctrlsac_agent import CTRLSACAgent
from networks.actor import StochasticActor
from networks.critic import Critic, SACCritic
from networks.feature import Mu, Phi, Theta
from utils.utils import load_isaaclab_env

# Optional: programmatic refinement cfg (requires Isaac app already configured)
def make_refine_env_cfg(
    predefined_task_coeff: list[list[float]],
    num_envs: int | None = None,
):
    """Return a copy of :class:`QuadcopterTrajectoryRefineEnvCfg` with custom coefficients.

    Use after ``AppLauncher`` / ``import omni.isaac.lab_tasks`` so the config class is available.
    """
    from omni.isaac.lab_tasks.direct.quadcopter.configs import QuadcopterTrajectoryRefineEnvCfg

    cfg = QuadcopterTrajectoryRefineEnvCfg()
    cfg.predefined_task_coeff = predefined_task_coeff
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


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
        vk = {
            "video_folder": video_folder,
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
    task_state_dim: int = 67,
    drone_state_dim: int = 13,
) -> dict[str, Any]:
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
    exp: dict[str, Any] | None = None,
) -> CTRLSACAgent:
    exp = exp or {}
    models = build_ctrlsac_models(
        env,
        device,
        multitask=multitask,
        actor_hidden_dim=int(exp.get("actor_hidden_dim", 512)),
        actor_hidden_depth=int(exp.get("actor_hidden_depth", 3)),
        feature_dim=int(exp.get("feature_dim", 512)),
        feature_hidden_dim=int(exp.get("feature_hidden_dim", 1024)),
        cdim=int(exp.get("cdim", 512)),
        task_state_dim=int(exp.get("task_state_dim", 67)),
        drone_state_dim=int(exp.get("drone_state_dim", 13)),
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
    cfg["extra_critic_steps"] = int(exp.get("extra_critic_steps", 2))
    cfg["target_update_period"] = int(exp.get("target_update_period", 1))
    cfg["eval"] = bool(exp.get("eval", False))
    cfg["experiment"]["wandb"] = bool(exp.get("wandb", True))
    cfg["experiment"]["wandb_kwargs"] = exp.get(
        "wandb_kwargs",
        {
            "project": exp.get("wandb_project", "CDC"),
            "name": f"CTRL-SAC-multitask-{multitask}",
            "config": {"task": task_name, "num_envs": getattr(env, "num_envs", None), "multitask": multitask},
        },
    )
    cfg["experiment"]["directory"] = exp.get("experiment_directory", f"runs/torch/{task_name}/CTRL-SAC-{multitask}/")
    cfg["alpha"] = float(exp.get("alpha", 1e-3))
    cfg["memory"] = None

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
) -> SAC:
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
    cfg["experiment"]["wandb_kwargs"] = exp.get(
        "wandb_kwargs",
        {
            "project": exp.get("wandb_project", "CDC"),
            "name": "SAC-baseline",
            "config": {"task": task_name, "num_envs": getattr(env, "num_envs", None)},
        },
    )
    return SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
