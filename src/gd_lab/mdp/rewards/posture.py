"""Base-height, orientation, and stance-pose reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_l2_clamped(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    max_ray_height: float = 10.0,
) -> torch.Tensor:
    """Terrain-relative base-height L2 penalty.

    Upstream ``base_height_l2`` with the ray heights clamped: a ray that misses
    returns a non-finite hit, and an unclamped mean propagates that into the
    penalty for the whole episode.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    root_z = torch.clamp(asset.data.root_pos_w[:, 2], min=-max_ray_height, max=max_ray_height)
    if sensor_cfg is None:
        return torch.square(root_z - target_height)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_z = torch.clamp(sensor.data.ray_hits_w[..., 2], min=-max_ray_height, max=max_ray_height)
    return torch.square(root_z - (target_height + torch.mean(ray_z, dim=1)))


def flat_orientation_roll_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Roll-only flat-orientation penalty: lateral gravity tilt held flat on all terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def slope_aligned_pitch_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_slope: float = 0.7,
) -> torch.Tensor:
    r"""Penalize body pitch deviation from the local terrain incline (not from flat).

    Fits a weighted least-squares plane to the height-scanner hits in the
    heading (yaw-only) frame to get the forward slope ``a = dz/dx``; the
    target gravity-x component is ``-a / sqrt(1 + a^2)`` (= ``-sin(atan(a))``).
    Reduces to the flat-pitch penalty on flat ground.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    rel = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    yaw = yaw_quat(asset.data.root_quat_w)
    rel_b = quat_apply_inverse(yaw.unsqueeze(1).expand(-1, rel.shape[1], -1), rel)
    # Missed ray casts return non-finite in all components; zero the whole ray
    # before weighting so 0 * inf does not poison the plane fit.
    valid = torch.isfinite(rel_b).all(dim=-1)
    rel_b = torch.where(valid.unsqueeze(-1), rel_b, torch.zeros_like(rel_b))
    w = valid.float()
    x, y, z = rel_b[..., 0], rel_b[..., 1], rel_b[..., 2]
    a_mat = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    a_weighted = a_mat * w.unsqueeze(-1)
    ata = a_weighted.transpose(1, 2) @ a_mat
    atz = a_weighted.transpose(1, 2) @ z.unsqueeze(-1)
    eye = torch.eye(3, device=env.device).expand(ata.shape[0], -1, -1)
    coeff = torch.linalg.solve(ata + 1e-4 * eye, atz).squeeze(-1)
    a = coeff[:, 0].clamp(-max_slope, max_slope)
    target_gx = -a / torch.sqrt(1.0 + a * a)
    return torch.nan_to_num(torch.square(asset.data.projected_gravity_b[:, 0] - target_gx), nan=0.0)
