"""Standstill-only stance-hold reward terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from gd_lab.mdp.terrain_families import terrain_family_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _standstill_gate(env: ManagerBasedRLEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """1.0 while the command is ~zero. Gates on ``||cmd_xy|| + |cmd_yaw|`` so a
    turn-in-place command is not misclassified as standing."""
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (command_norm <= command_threshold).float()


def stand_still_joint_deviation_with_yaw_command_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    exclude_terrain_families: Sequence[str] = (),
) -> torch.Tensor:
    """Nominal-pose L1 pull at standstill, with a yaw-aware command gate.

    ``exclude_terrain_families`` zeroes the pull where the flat-ground nominal
    pose is unreachable: on a staircase it leaves the downhill foot hovering
    above the lower tread. :func:`stand_still_feet_contact` covers those.
    """
    asset = env.scene[asset_cfg.name]
    joint_deviation = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    gate = _standstill_gate(env, command_name, command_threshold)
    return joint_deviation * gate * terrain_family_gate(env, exclude_terrain_families)


def stand_still_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    force_threshold: float = 10.0,
    exclude_terrain_families: Sequence[str] = (),
) -> torch.Tensor:
    """Count of feet off the ground at standstill (0..4); assign a NEGATIVE weight.

    Reads contact, not joint pose, so it stays valid on stairs where the pose
    pull is disabled: it removes the unstable 3-foot standstill there.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1).values > force_threshold
    num_lifted = torch.sum((~in_contact).float(), dim=1)
    gate = _standstill_gate(env, command_name, command_threshold)
    return num_lifted * gate * terrain_family_gate(env, exclude_terrain_families)
