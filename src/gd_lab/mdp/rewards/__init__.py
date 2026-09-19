"""Public reward-term functions, for use as ``RewTermCfg(func=...)`` targets."""

from __future__ import annotations

from .gait import feet_cadence_overrun, feet_touchdown_vel, foot_clearance_swing
from .posture import base_height_l2_clamped, flat_orientation_roll_l2, slope_aligned_pitch_l2
from .regularization import joint_power_l1, lin_vel_z_l2_scaled
from .smoothness import action_acceleration_l2
from .standstill import stand_still_feet_contact, stand_still_joint_deviation_with_yaw_command_l1
from .tracking import track_ang_vel_z_velocity_scaled_exp, track_lin_vel_xy_velocity_scaled_exp

__all__ = [
    "action_acceleration_l2",
    "base_height_l2_clamped",
    "feet_cadence_overrun",
    "feet_touchdown_vel",
    "flat_orientation_roll_l2",
    "foot_clearance_swing",
    "joint_power_l1",
    "lin_vel_z_l2_scaled",
    "slope_aligned_pitch_l2",
    "stand_still_feet_contact",
    "stand_still_joint_deviation_with_yaw_command_l1",
    "track_ang_vel_z_velocity_scaled_exp",
    "track_lin_vel_xy_velocity_scaled_exp",
]
