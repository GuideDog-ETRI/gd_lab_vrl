"""Reward recipe for blind DreamWaQ rough locomotion.

Weights are final values. Terms declared at weight 0.0 stay declared and are
stripped at runtime by the env's zero-weight pass, so re-enabling one is a
config override, not a code change. Bodies and joints are referenced only by
name patterns; no robot type appears here.

Several terms carry per-terrain-family scales: a penalty tuned on flat ground is
the wrong price on a stair, most sharply ``lin_vel_z_l2``.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

import gd_lab.mdp.rewards as gd_rew

# The four families where a flat-nominal standing pose is unreachable: the feet
# sit on different step levels, so a pose pull leaves the downhill foot hovering.
_STAIR_FAMILIES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "pyramid_stairs_nose",
    "pyramid_stairs_inv_nose",
)
# Stairs and slopes need sustained vertical base motion by geometry alone.
_CLIMB_FAMILY_SCALES = {f: 0.25 for f in (*_STAIR_FAMILIES, "hf_pyramid_slope", "hf_pyramid_slope_inv")}


@configclass
class DreamwaqRewardsCfg(RewardsCfg):
    # The command-relative low-speed gate is load-bearing: ungated, standing
    # under a 0.1 command pays ~0.96 and the command curriculum never promotes.
    track_lin_vel_xy_exp = RewTerm(
        func=gd_rew.track_lin_vel_xy_velocity_scaled_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_base": 0.3,
            "alpha": 0.2,
            "cmd_threshold": 0.1,
            "vel_frac_low": 0.4,
            "vel_frac_high": 0.8,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=gd_rew.track_ang_vel_z_velocity_scaled_exp,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "std_base": 0.3,
            "alpha": 0.3,
            "cmd_threshold": 0.1,
            "vel_frac_low": 0.4,
            "vel_frac_high": 0.8,
        },
    )

    # Effective -0.5 on stairs and slopes: descending requires a negative v_z.
    lin_vel_z_l2 = RewTerm(
        func=gd_rew.lin_vel_z_l2_scaled,
        weight=-2.0,
        params={"family_weight_scales": dict(_CLIMB_FAMILY_SCALES)},
    )
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.15)

    # -- joint / action regularization (ramped by penalty_terrain_schedule)
    dof_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-2.5e-5)
    dof_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    joint_power = RewTerm(func=gd_rew.joint_power_l1, weight=-2e-5)
    dof_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-5e-4)
    smoothness = RewTerm(func=gd_rew.action_acceleration_l2, weight=-0.02)

    # -- root height, terrain-relative. The target sits just above the nominal
    # standing trunk height so the strong weight does not pull the trunk up.
    base_height = RewTerm(
        func=gd_rew.base_height_l2_clamped,
        weight=-15.0,
        params={"target_height": 0.52, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    # Two complementary halves: the pose pull mis-targets on stairs, the contact
    # count is valid there.
    stand_still = RewTerm(
        func=gd_rew.stand_still_joint_deviation_with_yaw_command_l1,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "exclude_terrain_families": _STAIR_FAMILIES,
        },
    )
    stand_still_feet_contact = RewTerm(
        func=gd_rew.stand_still_feet_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "force_threshold": 10.0,
        },
    )

    # Upstream's feet_air_time targets ANYmal body names (".*FOOT") and is
    # superseded by the cadence bound below.
    feet_air_time = None
    # 0.3 air + 0.4 contact = 0.7 s period (~1.43 Hz). One-sided, so the caps
    # bound each phase from above without pinning a duty factor; asymmetric so
    # the swing tightens while rough terrain keeps its longer ground contact.
    feet_cadence_overrun = RewTerm(
        func=gd_rew.feet_cadence_overrun,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "target_air_time": 0.3,
            "target_contact_time": 0.4,
            "overrun_cap": 0.5,
            "cmd_threshold": 0.1,
        },
    )
    # 0.52 base-height target + (-0.43) = 0.09 m effective ground clearance;
    # move the two together.
    foot_clearance = RewTerm(
        func=gd_rew.foot_clearance_swing,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.43,
            "command_name": "base_velocity",
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
    # Half price where landing on a lower tread is unavoidably harder.
    feet_touchdown = RewTerm(
        func=gd_rew.feet_touchdown_vel,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "family_weight_scales": {"pyramid_stairs": 0.5, "pyramid_stairs_nose": 0.5},
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
    # -0.04 per fall after weight x dt, on top of the dominant implicit cost of
    # losing all future value. Small on purpose: a large explicit fall price
    # buys "stand at the top rather than risk the descent", and at a high fall
    # rate it drives per-step return negative, making a fast death optimal.
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-2.0)

    # -- orientation: roll hard-locked, pitch follows the local terrain incline.
    flat_orientation_roll_l2 = RewTerm(func=gd_rew.flat_orientation_roll_l2, weight=-2.0)
    flat_orientation_pitch_l2 = RewTerm(
        func=gd_rew.slope_aligned_pitch_l2,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
    )

    # Holds stance width inside the nominal footprint. Exactly one hip
    # regulator: a hip both velocity-damped and pose-anchored cannot make the
    # wide abduction step a stair descent needs.
    joint_deviation_hip_l1 = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HIP"])},
    )
