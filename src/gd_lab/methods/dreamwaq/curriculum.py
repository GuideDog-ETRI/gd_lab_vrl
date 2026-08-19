"""Curriculum for blind DreamWaQ.

Terrain promotion is driven by the inherited ``terrain_levels`` term (upstream
``terrain_levels_vel``); the terms here ramp command range and penalty weights.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

import gd_lab.mdp.curriculums as gd_cur
import gd_lab.mdp.terrain_curriculums as gd_tcur

# Regularization terms ramped from half strength to full as the ratcheted mean
# terrain level climbs; style shaping must not fight gait discovery early on.
_PENALTY_RAMP_TERMS = [
    "smoothness",
    "action_rate_l2",
    "dof_acc_l2",
    "dof_torques_l2",
    "dof_vel",
    "ang_vel_xy_l2",
]


@configclass
class DreamwaqCurriculumCfg(CurriculumCfg):
    # Command-range ramp 0.1x -> 1.0x of the terminal range, gated on the
    # tracking reward clearing promote_threshold * weight.
    command_levels_lin_vel = CurrTerm(
        func=gd_cur.command_levels_lin_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.1, 1.0),
            "promote_threshold": 0.45,
            "terrain_throttle_scale": 5.0,
            "terrain_throttle_floor": 0.2,
        },
    )
    command_levels_ang_vel = CurrTerm(
        func=gd_cur.command_levels_ang_vel,
        params={
            "reward_term_name": "track_ang_vel_z_exp",
            "range_multiplier": (0.1, 1.0),
            "promote_threshold": 0.4,
            "terrain_throttle_scale": 5.0,
            "terrain_throttle_floor": 0.2,
        },
    )
    penalty_terrain_schedule = CurrTerm(
        func=gd_tcur.penalty_weight_terrain_schedule,
        params={
            "term_names": list(_PENALTY_RAMP_TERMS),
            "kappa_min": 0.5,
            "level_start": 3.0,
            "level_end": 8.0,
        },
    )
    feet_touchdown_schedule = CurrTerm(
        func=gd_tcur.penalty_weight_terrain_schedule,
        params={
            "term_names": ["feet_touchdown"],
            "kappa_min": 0.5,
            "kappa_mid": 0.5,
            "post_ramp_iters": 1000,
            "level_start": 3.0,
            "level_end": 8.0,
        },
    )
