"""Reward recipe for blind DreamWaQ rough locomotion.

Weights are final values. Terms declared at weight 0.0 stay declared and are
stripped at runtime by the env's zero-weight pass, so re-enabling one is a
config override, not a code change. Bodies and joints are referenced only by
name patterns; no robot type appears here.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

import gd_lab.mdp.rewards as gd_rew
from gd_lab.core.types import FOOT_ORDER


@configclass
class DreamwaqRewardsCfg(RewardsCfg):
    # -- tracking and lin_vel_z penalty are inherited unchanged from the
    # upstream velocity RewardsCfg.

    # -- velocity penalties
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.15)

    # -- joint / action regularization (ramped by penalty_terrain_schedule)
    dof_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-2.5e-5)
    dof_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    dof_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-5e-4)
    smoothness = RewTerm(func=gd_rew.action_acceleration_l2, weight=-0.02)

    # -- root height, terrain-relative. The target sits just above the nominal
    # standing trunk height so the strong weight does not pull the trunk up.
    base_height = RewTerm(
        func=gd_rew.base_height_l2_clamped,
        weight=-15.0,
        params={"target_height": 0.52, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    # -- stand still
    stand_still = RewTerm(
        func=gd_rew.stand_still_joint_deviation_with_yaw_command_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # -- gait
    # Upstream's feet_air_time targets ANYmal body names (".*FOOT") and is
    # superseded by the cadence bound below.
    feet_air_time = None
    # Dense cadence bound: each foot pays the seconds its current air/stance
    # phase exceeds the target. One-sided caps bound each phase from above
    # without pinning a duty factor; 0.25 s air + 0.35 s contact = 0.6 s period.
    feet_cadence_overrun = RewTerm(
        func=gd_rew.feet_cadence_overrun,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "target_air_time": 0.25,
            "target_contact_time": 0.35,
            "cmd_threshold": 0.1,
        },
    )
    # Swing clearance measured against the terrain under the foot, so the
    # requirement rises automatically on a stair riser.
    foot_clearance = RewTerm(
        func=gd_rew.foot_clearance_terrain,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_ORDER), preserve_order=True),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_ORDER), preserve_order=True),
            "scanner_names": [f"height_scanner_feet_{f[:2].lower()}" for f in FOOT_ORDER],
            "min_clearance": 0.10,
            "contact_threshold": 1.0,
        },
    )
    feet_slipping = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_touchdown = RewTerm(
        func=gd_rew.feet_touchdown_vel,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # -- contacts. The trunk is deliberately not listed: trunk contact is the
    # base_contact termination, and taxing it here only double-counts falls.
    undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=0.0)

    # -- orientation: roll hard-locked, pitch follows the local terrain incline.
    flat_orientation_roll_l2 = RewTerm(func=gd_rew.flat_orientation_roll_l2, weight=-2.0)
    flat_orientation_pitch_l2 = RewTerm(
        func=gd_rew.slope_aligned_pitch_l2,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
    )

    # -- hip shaping: the abduction joints are held near their default, keeping
    # the stance width inside the nominal footprint.
    joint_deviation_hip_l1 = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HIP"])},
    )
