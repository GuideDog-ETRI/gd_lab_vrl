"""Curriculum for blind DreamWaQ.

All four schedules read the same terrain level, so harder terrain, wider
commands, fuller style penalties and a less forgiving command signal arrive
together as the policy earns them.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

import gd_lab.mdp.curriculums as gd_cur
import gd_lab.mdp.terrain_curriculums as gd_tcur

# Regularization terms ramped from kappa_min to full strength as the ratcheted
# mean terrain level climbs; style shaping must not fight gait discovery early on.
_PENALTY_RAMP_TERMS = [
    "smoothness",
    "action_rate_l2",
    "dof_acc_l2",
    "dof_torques_l2",
    "joint_power",
    "dof_vel",
    "feet_touchdown",
]


@configclass
class DreamwaqCurriculumCfg(CurriculumCfg):
    # Replaces the inherited stock term, whose distance test the pulse poisons.
    terrain_levels = CurrTerm(func=gd_tcur.terrain_levels_vel_cmd_aware, params={"min_moving_time": 2.0})
    pulse_prob_schedule = CurrTerm(
        func=gd_tcur.pulse_prob_terrain_schedule,
        params={"command_name": "base_velocity", "level_start": 3.0, "level_end": 8.0},
    )
    # Ramps 0.1x -> 1.0x of the terminal range once the tracking reward clears
    # promote_threshold * weight. Both thresholds sit below the observed plateau
    # of their gated reward; the stock 0.8 default freezes each at 0.1x.
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
            "kappa_min": 0.2,
            "level_start": 3.0,
            "level_end": 8.0,
        },
    )
    # Monitoring only.
    terrain_levels_per_family = CurrTerm(func=gd_tcur.terrain_levels_per_family)
    termination_per_family = CurrTerm(
        func=gd_tcur.termination_per_family,
        params={"term_name": "base_contact"},
    )
