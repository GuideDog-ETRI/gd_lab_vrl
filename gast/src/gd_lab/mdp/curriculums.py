"""Command-range curricula for the base-velocity command term.

Promotion decisions accumulate the per-reset ``Episode_Reward`` contribution
across every reset event and evaluate the running mean every
``max_episode_length`` steps, instead of sampling ``episode_sums`` only on
steps where ``common_step_counter % max_episode_length == 0`` (where the
resetting subset can be small or biased after early-termination desync).

The terrain-gated penalty-weight ramp lives in ``terrain_curriculums.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _accumulate(env: ManagerBasedRLEnv, state_key: str, reward_term_name: str, env_ids) -> None:
    """Add this reset event's contribution to a windowed Episode_Reward accumulator."""
    sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    n = int(sums.numel())
    if n == 0:
        return
    state = getattr(env, state_key)
    state["sum"] += float(sums.sum()) / env.max_episode_length_s
    state["n"] += n


def _ready_to_eval(env: ManagerBasedRLEnv, state_key: str) -> bool:
    state = getattr(env, state_key)
    return env.common_step_counter >= state["next_check"] and state["n"] > 0


def _consume_window(env: ManagerBasedRLEnv, state_key: str) -> float:
    state = getattr(env, state_key)
    avg = state["sum"] / state["n"]
    state["sum"] = 0.0
    state["n"] = 0
    state["next_check"] += env.max_episode_length
    return avg


def _terrain_throttle(env: ManagerBasedRLEnv, scale: float | None, floor: float = 0.0) -> float:
    """Soft delta multiplier in ``[floor, 1]`` from ``mean(terrain_levels) / scale``.

    ``scale=None`` disables the throttle (returns 1.0). A nonzero ``floor``
    prevents a collapsed mean terrain level from freezing command growth
    forever (which would in turn prevent terrain progress from resuming).
    """
    if scale is None or scale <= 0:
        return 1.0
    if not hasattr(env.scene, "terrain") or env.scene.terrain is None:
        return 1.0
    level = float(env.scene.terrain.terrain_levels.float().mean().item())
    return float(max(floor, min(1.0, level / scale)))


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    promote_threshold: float = 0.8,
    delta: float = 0.1,
    terrain_throttle_scale: float | None = None,
    terrain_throttle_floor: float = 0.0,
) -> torch.Tensor:
    """Linear-velocity command-range curriculum.

    Each window of ``max_episode_length`` steps, if the windowed mean of
    ``Episode_Reward/<reward_term_name>`` exceeds ``promote_threshold *
    weight``, expand the symmetric ``lin_vel_x``/``lin_vel_y`` ranges by
    ``+-delta_eff``, clamped to ``original * range_multiplier[1]``.

    ``terrain_throttle_scale`` scales ``delta_eff`` down while mean terrain
    level lags (soft coupling, never blocks promotion outright).
    """
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    weight = env.reward_manager.get_term_cfg(reward_term_name).weight

    if not hasattr(env, "_lv_state"):
        env._lvx_orig = torch.tensor(ranges.lin_vel_x, device=env.device)
        env._lvy_orig = torch.tensor(ranges.lin_vel_y, device=env.device)
        env._lvx_final = env._lvx_orig * range_multiplier[1]
        env._lvy_final = env._lvy_orig * range_multiplier[1]
        ranges.lin_vel_x = (env._lvx_orig * range_multiplier[0]).tolist()
        ranges.lin_vel_y = (env._lvy_orig * range_multiplier[0]).tolist()
        env._lv_state = {"sum": 0.0, "n": 0, "next_check": int(env.max_episode_length)}
        return torch.tensor(ranges.lin_vel_x[1], device=env.device)

    _accumulate(env, "_lv_state", reward_term_name, env_ids)

    if _ready_to_eval(env, "_lv_state"):
        avg = _consume_window(env, "_lv_state")
        if avg > promote_threshold * weight:
            throttle = _terrain_throttle(env, terrain_throttle_scale, terrain_throttle_floor)
            delta_eff = delta * throttle
            step = torch.tensor([-delta_eff, delta_eff], device=env.device)
            new_x = torch.tensor(ranges.lin_vel_x, device=env.device) + step
            new_y = torch.tensor(ranges.lin_vel_y, device=env.device) + step
            new_x = torch.clamp(new_x, min=env._lvx_final[0], max=env._lvx_final[1])
            new_y = torch.clamp(new_y, min=env._lvy_final[0], max=env._lvy_final[1])
            ranges.lin_vel_x = new_x.tolist()
            ranges.lin_vel_y = new_y.tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    promote_threshold: float = 0.8,
    delta: float = 0.1,
    terrain_throttle_scale: float | None = None,
    terrain_throttle_floor: float = 0.0,
) -> torch.Tensor:
    """Angular-velocity command-range curriculum; mirrors :func:`command_levels_lin_vel`.

    The distance-ratio gate does not apply to yaw rate, so only the terrain
    throttle is offered here.
    """
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    weight = env.reward_manager.get_term_cfg(reward_term_name).weight

    if not hasattr(env, "_av_state"):
        env._avz_orig = torch.tensor(ranges.ang_vel_z, device=env.device)
        env._avz_final = env._avz_orig * range_multiplier[1]
        ranges.ang_vel_z = (env._avz_orig * range_multiplier[0]).tolist()
        env._av_state = {"sum": 0.0, "n": 0, "next_check": int(env.max_episode_length)}
        return torch.tensor(ranges.ang_vel_z[1], device=env.device)

    _accumulate(env, "_av_state", reward_term_name, env_ids)

    if _ready_to_eval(env, "_av_state"):
        avg = _consume_window(env, "_av_state")
        if avg > promote_threshold * weight:
            throttle = _terrain_throttle(env, terrain_throttle_scale, terrain_throttle_floor)
            delta_eff = delta * throttle
            step = torch.tensor([-delta_eff, delta_eff], device=env.device)
            new_z = torch.tensor(ranges.ang_vel_z, device=env.device) + step
            new_z = torch.clamp(new_z, min=env._avz_final[0], max=env._avz_final[1])
            ranges.ang_vel_z = new_z.tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
