"""Terrain-gated penalty-weight ramp."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def penalty_weight_terrain_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: list[str],
    kappa_min: float = 0.5,
    level_start: float = 1.5,
    level_end: float = 5.0,
    iter_start: float | None = None,
    iter_end: float | None = None,
    steps_per_iter: int = 48,
    kappa_mid: float | None = None,
    post_ramp_iters: float | None = None,
) -> dict:
    """Scale regularization reward weights by ratcheted terrain-level progress.

    ``weight = declared * kappa``, ``kappa = kappa_min + (1 - kappa_min) *
    clamp((L - level_start) / (level_end - level_start), 0, 1)`` where ``L``
    is the running-max mean of ``terrain.terrain_levels`` (a ratchet, so
    kappa stays monotonic even if a level dip follows a penalty increase).
    Falls back to ``kappa=1`` when terrain levels are unavailable (play /
    flat plane). Declared weights are captured once, pristine, and set
    absolutely every call, so the schedule is resume-safe and idempotent;
    keep ``term_names`` disjoint from any other absolute-weight schedule.

    Optional time gate (``iter_start``/``iter_end``, in learning iterations
    derived from ``common_step_counter / steps_per_iter``, which must equal
    the runner's ``num_steps_per_env``): the ramp becomes ``min(terrain_frac,
    time_frac)``. Not resume-safe on its own, unlike the terrain ratchet.

    Two-stage variant (``kappa_mid`` and ``post_ramp_iters`` both set):
    terrain progress ramps ``kappa_min -> kappa_mid``; once the ratcheted
    level first reaches ``level_end`` that iteration is latched, and kappa
    then ramps ``kappa_mid -> 1`` over the next ``post_ramp_iters``
    iterations.
    """
    terrain = env.scene.terrain
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
        level = float(level_end)  # play / flat plane: final (full) weights
    else:
        level = levels.float().mean().item()
    ratchet = max(float(getattr(env, "_penalty_terrain_level_max", 0.0)), level)
    env._penalty_terrain_level_max = ratchet

    if level_end <= level_start:
        frac = 1.0
    else:
        frac = min(1.0, max(0.0, (ratchet - level_start) / (level_end - level_start)))

    if iter_start is not None and iter_end is not None and iter_end > iter_start:
        it = env.common_step_counter / max(1, steps_per_iter)
        time_frac = min(1.0, max(0.0, (it - iter_start) / (iter_end - iter_start)))
        frac = min(frac, time_frac)

    if kappa_mid is not None and post_ramp_iters is not None and post_ramp_iters > 0:
        if levels is None:
            kappa = 1.0  # play / flat plane: full declared weights, no ramp
        else:
            latch_key = "_penalty_post_ramp_it0_" + "_".join(term_names)
            it = env.common_step_counter / max(1, steps_per_iter)
            if frac >= 1.0 and not hasattr(env, latch_key):
                setattr(env, latch_key, it)
            if hasattr(env, latch_key):
                post_frac = min(1.0, max(0.0, (it - getattr(env, latch_key)) / post_ramp_iters))
                kappa = kappa_mid + (1.0 - kappa_mid) * post_frac
            else:
                kappa = kappa_min + (kappa_mid - kappa_min) * frac
    else:
        kappa = kappa_min + (1.0 - kappa_min) * frac

    key = "_penalty_terrain_base_weights_" + "_".join(term_names)
    if not hasattr(env, key):
        setattr(env, key, {n: float(env.reward_manager.get_term_cfg(n).weight) for n in term_names})
    for name, w0 in getattr(env, key).items():
        env.reward_manager.get_term_cfg(name).weight = w0 * kappa
    return {"penalty_terrain_kappa": kappa}
