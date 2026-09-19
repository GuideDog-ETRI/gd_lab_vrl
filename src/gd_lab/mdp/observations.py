"""Observation terms: terrain-relative feet and privileged dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def feet_around_height_from_terrain(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
    sensor_cfg_fl: SceneEntityCfg | None = None,
    sensor_cfg_fr: SceneEntityCfg | None = None,
    sensor_cfg_rl: SceneEntityCfg | None = None,
    sensor_cfg_rr: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Per-ray foot heights relative to local terrain under each foot.

    ``asset_cfg`` must list the feet with ``preserve_order=True`` so its body
    order matches the sensor arguments; rays are not reduced, so four feet of
    four rays give ``(num_envs, 16)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    sensor_cfgs = [sensor_cfg_fl, sensor_cfg_fr, sensor_cfg_rl, sensor_cfg_rr]
    if any(cfg is None for cfg in sensor_cfgs):
        raise ValueError("feet_around_height_from_terrain requires all four foot sensor cfgs")
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [B, num_feet]

    terrain_z_list = []
    for cfg in sensor_cfgs:
        ray_hits_z = env.scene.sensors[cfg.name].data.ray_hits_w[..., 2]
        terrain_z_list.append(ray_hits_z.reshape(ray_hits_z.shape[0], -1))  # [B, rays_per_foot]

    terrain_z = torch.stack(terrain_z_list, dim=1)  # [B, num_feet, rays_per_foot]
    rel_height = foot_z.unsqueeze(-1).expand_as(terrain_z) - terrain_z
    return rel_height.reshape(rel_height.shape[0], -1)


def feet_contact_on_terrain(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Binary contact state of feet with terrain."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0] > threshold
    return contact.float()


"""
Privileged dynamics (asymmetric-critic-only ground truth of quantities the policy
otherwise has to infer through proprioceptive history).
"""


def feet_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-foot net contact force vectors in the base frame, ``(N, 12)``.

    ``sensor_cfg`` must list the feet with ``preserve_order=True`` so the body
    order is stable; world-frame forces are rotated into the base frame so the
    observation is heading-invariant.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 4, 3)
    quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, forces_w.shape[1], -1)
    return quat_apply_inverse(quat, forces_w).reshape(env.num_envs, -1)


def friction_coeff(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shape-mean ``(static, dynamic)`` friction of the robot, ``(N, 2)``.

    Friction is randomized per shape once at startup, so the physx-view read
    (CPU tensors) happens on the first call only; the shape-mean is cached on
    the env and served from device memory afterwards.
    """
    cached = getattr(env, "_gd_friction_coeff", None)
    if cached is None:
        asset: Articulation = env.scene[asset_cfg.name]
        mats = asset.root_physx_view.get_material_properties()  # (N, shapes, 3), CPU
        cached = mats[..., :2].mean(dim=1).to(env.device)
        env._gd_friction_coeff = cached
    return cached


def base_mass_offset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Startup base-mass domain-randomization delta in kg, ``(N, 1)``.

    Cached: the startup delta is constant for the run. Reads the payload event's
    post-startup baseline when present, so the per-episode payload is not folded
    into the startup delta.
    """
    cached = getattr(env, "_gd_base_mass_offset", None)
    if cached is None:
        asset: Articulation = env.scene[asset_cfg.name]
        # (N, bodies), CPU
        masses = getattr(env, "_payload_base_mass", None)
        if masses is None:
            masses = asset.root_physx_view.get_masses()
        delta = (masses.sum(dim=1) - asset.data.default_mass.sum(dim=1)).to(env.device)
        cached = delta.unsqueeze(1)
        env._gd_base_mass_offset = cached
    return cached


def actuator_gain_scale(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Live PD-gain scale vs. cfg defaults, ``(N, 24)`` = [kp x12, kd x12].

    Actuator gains are rescaled from the default joint stiffness/damping on
    every reset, so this is recomputed each call (no cache): per actuator
    group, current / default, written in global joint order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    kp = torch.ones(env.num_envs, asset.num_joints, device=env.device)
    kd = torch.ones_like(kp)
    for act in asset.actuators.values():
        ids = act.joint_indices
        kp[:, ids] = act.stiffness / asset.data.default_joint_stiffness[:, ids]
        kd[:, ids] = act.damping / asset.data.default_joint_damping[:, ids]
    return torch.cat((kp, kd), dim=1)


def payload_mass(env: ManagerBasedEnv, scale: float = 0.2) -> torch.Tensor:
    """Commanded trunk payload as ``payload_kg * scale``, ``(N, 1)``.

    An observable, not a privileged estimate: the operator enters the mounted
    payload at deploy, so it carries no noise.
    """
    payload = getattr(env, "payload_kg", None)
    if payload is None:
        payload = torch.zeros(env.num_envs, device=env.device)
    return (payload * scale).unsqueeze(1)


def push_delta_v(env: ManagerBasedEnv, hold_s: float = 0.2) -> torch.Tensor:
    """Base-frame linear velocity delta the interval push applied, ``(N, 3)``.

    Privileged, critic-only. Held ``hold_s`` seconds after each push, clipped to
    the current episode so a push before a reset does not leak into the next.
    Reads zeros with no push event, keeping the critic width stable.
    """
    buf = getattr(env, "push_delta_v_buf", None)
    if buf is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    hold_steps = max(int(round(hold_s / env.step_dt)), 1)
    age = int(env.common_step_counter) - env.push_step_buf
    fresh = (age >= 0) & (age < hold_steps) & (age <= env.episode_length_buf)
    return torch.where(fresh.unsqueeze(1), buf, torch.zeros_like(buf))
