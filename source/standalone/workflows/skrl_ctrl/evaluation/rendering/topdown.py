"""Viewport top-down framing for one env clone (optional ``RecordVideo``)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


def find_direct_rl_env(env: Any) -> Any:
    cur = env
    for _ in range(64):
        if hasattr(cur, "viewport_camera_controller"):
            return cur
        if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
            cur = cur.unwrapped
        elif hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "_env"):
            cur = cur._env
        else:
            break
    raise RuntimeError("Could not find a DirectRLEnv with viewport_camera_controller")


def apply_topdown_viewport(
    env: Any,
    *,
    env_index: int = 0,
    eye_z: float = 10.0,
) -> None:
    """Point the viewport camera at ``env_index`` origin with a top-down offset.

    Uses :class:`ViewportCameraController` when available (requires RTX / partial rendering).
    """
    base = find_direct_rl_env(env)
    vc = getattr(base, "viewport_camera_controller", None)
    if vc is None:
        return
    vc.set_view_env_index(env_index)
    vc.update_view_location(eye=(0.0, 0.0, eye_z), lookat=(0.0, 0.0, 0.0))


def apply_follower_viewport(
    env: Any,
    *,
    env_index: int = 0,
    asset_name: str = "robot",
    eye_offset: tuple[float, float, float] = (0.0, 0.0, 1.0),
    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Track ``asset_name`` (world-root) with camera eye / look-at as world-axis offsets from that root.

    Requires ``viewport_camera_controller`` (partial rendering / ``--video`` + ``--enable_cameras`` on batch eval).
    """
    base = find_direct_rl_env(env)
    vc = getattr(base, "viewport_camera_controller", None)
    if vc is None:
        return
    vc.set_view_env_index(env_index)
    vc.update_view_to_asset_root(asset_name)
    vc.update_view_location(eye=eye_offset, lookat=lookat_offset)


def wrap_record_video(
    env: Any,
    *,
    video_folder: str,
    video_length: int,
    step_trigger: Any | None = None,
) -> gym.Env:
    """Wrap with :class:`gymnasium.wrappers.RecordVideo` (expects ``render_mode='rgb_array'``)."""
    if step_trigger is None:
        step_trigger = lambda step: step == 0
    return gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        step_trigger=step_trigger,
        video_length=video_length,
        disable_logger=True,
    )
