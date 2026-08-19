"""Standstill-only stance-hold reward term."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stand_still_joint_deviation_with_yaw_command_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Nominal-pose L1 pull at standstill, with a yaw-aware command gate.

    Gates on ``||cmd_xy|| + |cmd_yaw||`` (not just ``||cmd_xy||``) so a
    turn-in-place command is not misclassified as standing.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    gate = (command_norm <= command_threshold).float()
    joint_deviation = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return joint_deviation * gate
