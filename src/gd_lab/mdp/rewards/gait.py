"""Contact-sensor gait shaping and foot-clearance reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from gd_lab.mdp.terrain_families import terrain_family_scale

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_cadence_overrun(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_air_time: float = 0.3,
    target_contact_time: float = 0.4,
    cmd_threshold: float = 0.1,
    overrun_cap: float | None = 0.5,
) -> torch.Tensor:
    """Dense per-step penalty for the current air/contact phase exceeding its target.

    Event-paid air/contact-time terms fire only at touchdown/lift-off; pricing
    the overrun every step it is happening is what actually moves the gait.
    One-sided: phases shorter than target are free.

    ``overrun_cap`` bounds the rising-edge burst - ``current_contact_time``
    accrues while the command gate is closed, so uncapped, the first step after
    a 2 s pulse hold is charged for the whole standstill.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    stance = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    over = (air - target_air_time).clamp(min=0.0) + (stance - target_contact_time).clamp(min=0.0)
    if overrun_cap is not None:
        over = over.clamp(max=overrun_cap)
    command = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return torch.sum(over, dim=1) * (cmd_mag > cmd_threshold)


def feet_touchdown_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    family_weight_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize foot linear speed at the moment of touchdown.

    ``family_weight_scales`` softens it where a landing is inherently harder
    than on flat ground, so the anti-slamming pressure does not bias against
    committing to a step down.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    penalty = torch.sum(foot_vel.norm(dim=-1) * first_contact, dim=1)
    return penalty * terrain_family_scale(env, family_weight_scales)


def foot_clearance_swing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    target_height: float = -0.43,
    contact_force_threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    upright_threshold: float = 0.7,
) -> torch.Tensor:
    r"""One-sided, linear, swing-gated foot clearance penalty in the BODY frame.

    ``sum_k clamp(target_height - p_fz,k, min=0) * 1[foot k in swing]`` with
    ``p_fz`` in the base frame, so ``target_height`` is negative. Effective
    ground clearance is ``base_height target + target_height`` - move the two
    together.

    Linear, not squared: a squared shortfall's gradient vanishes as the foot
    nears the target, stalling the swing a few cm short.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    translated = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)
    quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, translated.shape[1], -1)
    foot_z_body = quat_apply_inverse(quat, translated)[..., 2]
    shortfall = torch.clamp(target_height - foot_z_body, min=0.0)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_swing = (net_forces.norm(dim=-1).max(dim=1).values <= contact_force_threshold).float()

    command = env.command_manager.get_command(command_name)
    move_gate = (torch.norm(command, dim=1) > command_threshold).float()
    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, upright_threshold) / upright_threshold
    return torch.sum(shortfall * in_swing, dim=1) * move_gate * upright_gate
