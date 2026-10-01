"""Velocity-tracking reward kernels with a command-relative anti-sit-still gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _velocity_gate(
    cmd_mag: torch.Tensor,
    vel_mag: torch.Tensor,
    cmd_threshold: float,
    vel_frac_low: float,
    vel_frac_high: float,
) -> torch.Tensor:
    """Ramp from 0 at ``vel_frac_low * cmd`` to 1 at ``vel_frac_high * cmd``.

    Relative to the commanded speed, not absolute: the command curriculum starts
    at 0.1x of the terminal range, so any absolute floor above ~0.14 m/s makes
    perfect tracking earn zero and the range never grows. Open below
    ``cmd_threshold`` - a stand command has no speed to clear.
    """
    lo = vel_frac_low * cmd_mag
    hi = vel_frac_high * cmd_mag
    ramp = ((vel_mag - lo) / (hi - lo).clamp(min=1.0e-6)).clamp(0.0, 1.0)
    return torch.where(cmd_mag > cmd_threshold, ramp, torch.ones_like(ramp))


def track_lin_vel_xy_velocity_scaled_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_base: float = 0.3,
    alpha: float = 0.2,
    cmd_threshold: float = 0.1,
    vel_frac_low: float = 0.4,
    vel_frac_high: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Planar velocity tracking with a command-scaled kernel width.

    ``std = std_base + alpha * ||cmd||``: tight at low command, loose at high
    command so the policy may shed speed on hard terrain without a reward cliff.
    The looser kernel would pay near-full reward for standing, which the gate
    removes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_mag = torch.norm(cmd, dim=1)
    lin_vel_xy = asset.data.root_lin_vel_b[:, :2]

    std = std_base + alpha * cmd_mag
    reward = torch.exp(-torch.sum(torch.square(cmd - lin_vel_xy), dim=1) / (std * std))
    gate = _velocity_gate(cmd_mag, torch.norm(lin_vel_xy, dim=1), cmd_threshold, vel_frac_low, vel_frac_high)
    return reward * gate


def track_ang_vel_z_velocity_scaled_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_base: float = 0.3,
    alpha: float = 0.3,
    cmd_threshold: float = 0.1,
    vel_frac_low: float = 0.4,
    vel_frac_high: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-rate analogue of :func:`track_lin_vel_xy_velocity_scaled_exp`."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    cmd_mag = torch.abs(cmd_yaw)
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]

    std = std_base + alpha * cmd_mag
    reward = torch.exp(-torch.square(cmd_yaw - ang_vel_z) / (std * std))
    gate = _velocity_gate(cmd_mag, torch.abs(ang_vel_z), cmd_threshold, vel_frac_low, vel_frac_high)
    return reward * gate
