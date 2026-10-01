"""Gamepad-teleop velocity command term.

:mod:`gd_lab.devices` (and its `inputs` pip dependency) is imported lazily in
``__init__``, so importing this module does not require the gamepad backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GamepadVelocityCommand(CommandTerm):
    r"""Command generator that forwards SE(2) velocity commands from a gamepad.

    The command comprises the base linear and angular velocity: (v_x, v_y, \omega_z).
    The gamepad input is broadcast to all environments.
    """

    cfg: GamepadVelocityCommandCfg

    def __init__(self, cfg: GamepadVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Imported lazily: `inputs` (the gamepad backend) is a training-machine-only
        # extra, so importing this module must not require it to be installed.
        from gd_lab.devices import Se2Controller, Se2ControllerCfg

        self._device = Se2Controller(
            Se2ControllerCfg(
                sim_device=self.device,
                ranges=cfg.ranges,
                dead_zone=cfg.dead_zone,
                yaw_dead_zone=cfg.yaw_dead_zone,
                input_device=cfg.input_device,
            )
        )

    def __str__(self) -> str:
        msg = "GamepadVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        vel_err_xy = torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
            dim=-1,
        )
        vel_err_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self.metrics["error_vel_xy"] = vel_err_xy
        self.metrics["error_vel_yaw"] = vel_err_yaw

    def _resample_command(self, env_ids: Sequence[int]):
        """No-op: a joystick command does not need periodic resampling."""

    def _update_command(self):
        self.vel_command_b[:] = self._device.read_command()  # (vx, vy, wz)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class GamepadVelocityCommandCfg(CommandTermCfg):
    class_type: type = GamepadVelocityCommand

    asset_name: str = MISSING

    ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Gamepad-stick-to-velocity-range mapping."""

    dead_zone: float = 0.05
    """Radial dead zone on the translation stick, as a fraction of full stick travel."""

    yaw_dead_zone: float | None = None
    """Dead zone on the yaw stick. Falls back to :attr:`dead_zone` when None.

    The gamepad only emits events on change, so a stick that does not mechanically
    re-center to zero leaves a standing yaw command latched after release. Keep this
    above the pad's worst re-center error.
    """

    input_device: str = "xbox"

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
