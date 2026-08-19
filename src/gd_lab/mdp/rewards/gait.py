"""Contact-sensor gait shaping and foot-clearance reward terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_cadence_overrun(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_air_time: float = 0.2,
    target_contact_time: float = 0.2,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Dense per-step penalty for the current air/contact phase exceeding its target.

    Unlike the event-paid air/contact-time terms (which fire only at
    touchdown/lift-off), this prices the overrun every step the phase is
    active, so a long stance accrues penalty continuously. One-sided: phases
    shorter than target are free.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    stance = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    over = (air - target_air_time).clamp(min=0.0) + (stance - target_contact_time).clamp(min=0.0)
    command = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return torch.sum(over, dim=1) * (cmd_mag > cmd_threshold)


def feet_touchdown_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot linear speed at the moment of touchdown."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(foot_vel.norm(dim=-1) * first_contact, dim=1)


def foot_clearance_terrain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    scanner_names: Sequence[str] = (),
    min_clearance: float = 0.10,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Terrain-relative, one-sided, contact-gated foot clearance penalty.

    Each swing foot must clear the terrain under it by ``min_clearance``; only
    the shortfall is penalized, and only while the foot is out of contact.
    ``scanner_names`` name the per-foot ray casters in the same body order as
    ``asset_cfg``/``contact_cfg``, which must both set ``preserve_order=True``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    if len(scanner_names) != foot_pos.shape[1]:
        raise ValueError(f"Expected one scanner per foot, got {len(scanner_names)} for {foot_pos.shape[1]} feet")

    # Highest ray hit under each foot: the clearance that matters is over the
    # nearest obstacle, not the mean of a patch spanning a step edge.
    terrain_z = []
    for name in scanner_names:
        hits_z = env.scene.sensors[name].data.ray_hits_w[..., 2]
        hits_z = torch.nan_to_num(hits_z, nan=-1.0e6, posinf=-1.0e6, neginf=-1.0e6)
        terrain_z.append(hits_z.max(dim=1).values)
    terrain_z = torch.stack(terrain_z, dim=1)

    shortfall = torch.clamp(min_clearance - (foot_pos[..., 2] - terrain_z), min=0.0)
    contact_f = env.scene.sensors[contact_cfg.name].data.net_forces_w[:, contact_cfg.body_ids]
    in_swing = (contact_f.norm(dim=-1) < contact_threshold).float()
    return torch.sum(torch.square(shortfall) * in_swing, dim=1)
