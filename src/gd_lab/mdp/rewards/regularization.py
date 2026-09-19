"""Effort and body-velocity regularization terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from gd_lab.mdp.terrain_families import terrain_family_scale

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power cost ``sum |tau| * |qd|``.

    L1, not squared: a squared cost overprices the high-torque low-speed phase,
    which is exactly what a controlled stair descent is.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        * torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def lin_vel_z_l2_scaled(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    family_weight_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """``lin_vel_z_l2`` with a per-terrain-family weight scale.

    Stairs and slopes need sustained vertical base motion by geometry alone, so
    at full weight the cheapest response to a descent command is to not descend.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * terrain_family_scale(env, family_weight_scales)
