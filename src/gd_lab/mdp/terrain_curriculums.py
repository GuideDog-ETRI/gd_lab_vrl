"""Terrain-level promotion, terrain-gated schedules, and per-family monitoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from gd_lab.mdp.terrain_families import family_column_masks

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ratcheted_level(env: ManagerBasedRLEnv, state_attr: str) -> float | None:
    """Running max of the mean terrain level, or ``None`` on a plane / play terrain."""
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None) if terrain is not None else None
    if levels is None:
        return None
    ratchet = max(float(getattr(env, state_attr, 0.0)), float(levels.float().mean().item()))
    setattr(env, state_attr, ratchet)
    return ratchet


def _level_frac(level: float | None, level_start: float, level_end: float) -> float:
    if level is None:
        return 1.0
    span = max(level_end - level_start, 1e-6)
    return min(1.0, max(0.0, (level - level_start) / span))


def terrain_levels_vel_cmd_aware(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    min_moving_time: float = 2.0,
) -> torch.Tensor:
    """Terrain promotion graded against the distance actually COMMANDED.

    The stock criterion breaks under the pulse train in both directions: a fixed
    ``size/2`` promote bar starves an env that obeyed its zero holds, and grading
    demotion against the reset-instant command demotes it for obeying.

    Here the promote bar scales with the episode's moving fraction, the demote
    bar is half the integral of the command actually given, and envs commanded to
    move for less than ``min_moving_time`` are frozen - no gait evidence to
    grade. Falls back to the stock formula without the accumulators.
    """
    terrain = env.scene.terrain
    if terrain.cfg.terrain_generator is None:
        return torch.mean(terrain.terrain_levels.float())
    asset = env.scene["robot"]
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    tile_size = terrain.cfg.terrain_generator.size[0]
    cmd_term = env.command_manager.get_term("base_velocity")

    if not hasattr(cmd_term, "cmd_dist_integral"):
        command = env.command_manager.get_command("base_velocity")
        move_up = distance > tile_size / 2
        move_down = (distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5) & ~move_up
    else:
        moving_time = cmd_term.cmd_moving_time[env_ids]
        moving_frac = (moving_time / cmd_term.cmd_elapsed_time[env_ids].clamp_min(1e-6)).clamp(0.0, 1.0)
        eligible = moving_time > min_moving_time
        move_up = eligible & (distance > tile_size * 0.5 * moving_frac)
        move_down = eligible & (distance < 0.5 * cmd_term.cmd_dist_integral[env_ids]) & ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def pulse_prob_terrain_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    level_start: float = 3.0,
    level_end: float = 8.0,
) -> dict:
    """Ramp the command pulse train in with terrain competence (easy-first).

    Learn to walk the terrain first, then to stop and re-launch on it. Companion
    to, not a replacement for, :func:`terrain_levels_vel_cmd_aware`: this removes
    the early-discovery drag, that one removes the grading bias.

    The declared value is captured once and set absolutely, so the schedule is
    idempotent. On resume the ratchet restarts at 0 and re-ramps.
    """
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(env, "_pulse_prob_target"):
        env._pulse_prob_target = float(cmd_term.cfg.pulse_prob)
    frac = _level_frac(_ratcheted_level(env, "_pulse_terrain_level_max"), level_start, level_end)
    cmd_term.cfg.pulse_prob = env._pulse_prob_target * frac
    return {"pulse_prob": cmd_term.cfg.pulse_prob}


def penalty_weight_terrain_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: list[str],
    kappa_min: float = 0.2,
    level_start: float = 3.0,
    level_end: float = 8.0,
) -> dict:
    """Scale regularization reward weights by ratcheted terrain-level progress.

    ``weight = declared * kappa``, ``kappa = kappa_min + (1 - kappa_min) * frac``
    over the running-max mean terrain level - a ratchet, so kappa stays monotonic
    even if a level dip follows a penalty increase. Style shaping must not fight
    gait discovery early on; the declared weights are the ``kappa = 1`` targets.

    Weights are captured once, pristine, and set absolutely every call, so this
    is resume-safe and idempotent; keep ``term_names`` disjoint from any other
    absolute-weight schedule.
    """
    frac = _level_frac(_ratcheted_level(env, "_penalty_terrain_level_max"), level_start, level_end)
    kappa = kappa_min + (1.0 - kappa_min) * frac

    key = "_penalty_terrain_base_weights_" + "_".join(term_names)
    if not hasattr(env, key):
        setattr(env, key, {n: float(env.reward_manager.get_term_cfg(n).weight) for n in term_names})
    for name, w0 in getattr(env, key).items():
        env.reward_manager.get_term_cfg(name).weight = w0 * kappa
    return {"penalty_terrain_kappa": kappa}


def terrain_levels_per_family(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, float]:
    """Mean terrain level per sub-terrain family. Monitoring only - a family
    stalled at level 4 is invisible in an overall mean of 9."""
    terrain = env.scene.terrain
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
        return {}
    types = terrain.terrain_types
    levels = levels.float()
    out = {}
    for name, cols in family_column_masks(env).items():
        mask = torch.isin(types, cols)
        out[name] = levels[mask].mean().item() if mask.any() else 0.0
    return out


def termination_per_family(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str = "base_contact",
    ema_alpha: float = 0.02,
) -> dict[str, float]:
    """Per-family EMA rate of a termination term. Monitoring only - shows WHERE
    the falls happen.

    The EMA alpha is per-sample, so a rare family converges at the same
    per-episode rate as a common one. No-ops when ``term_name`` is inactive: the
    gamepad variant nulls ``base_contact``, and monitoring must not crash play.
    """
    manager = env.termination_manager
    families = family_column_masks(env)
    types = getattr(env.scene.terrain, "terrain_types", None)
    if term_name not in manager.active_terms or types is None or not families:
        return {}
    hit = manager.get_term(term_name)[env_ids].float()
    types = types[env_ids]

    key = f"_termination_per_family_{term_name}"
    state = getattr(env, key, None)
    if state is None:
        state = {}
        setattr(env, key, state)
    for name, cols in families.items():
        mask = torch.isin(types, cols)
        n = int(mask.sum())
        if n == 0:
            state.setdefault(name, 0.0)
            continue
        alpha = min(1.0, ema_alpha * n)
        state[name] = state.get(name, 0.0) + alpha * (hit[mask].mean().item() - state.get(name, 0.0))
    return dict(state)
