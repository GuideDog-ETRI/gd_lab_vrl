"""Single path for gym task registration. Task IDs are generated, never hand-written."""

from __future__ import annotations

import gymnasium as gym

PREFIX = "Gd"

_ALL_IDS: list[str] = []

_VALID_MODES = (None, "Play", "Gamepad")


def make_task_id(*, task: str, robot: str, method: str, mode: str | None = None) -> str:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    parts = [PREFIX, task, robot, method]
    if mode:
        parts.append(mode)
    return "-".join(parts) + "-v0"


def register_task(
    *,
    task: str,
    robot: str,
    method: str,
    env_cfg: str,
    agent_cfg: str,
    mode: str | None = None,
) -> str:
    """Register one task combination and return its generated gym ID.

    ``env_cfg`` / ``agent_cfg`` are ``"<module>:<attr>"`` entry-point strings.
    """
    task_id = make_task_id(task=task, robot=robot, method=method, mode=mode)
    if task_id in _ALL_IDS:
        raise ValueError(f"Task ID '{task_id}' registered twice")
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": agent_cfg,
        },
    )
    _ALL_IDS.append(task_id)
    return task_id


def all_ids() -> list[str]:
    return list(_ALL_IDS)
