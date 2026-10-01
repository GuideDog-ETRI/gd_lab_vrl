"""SE(2) gamepad controller: maps raw stick state to a scaled velocity command."""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .xbox import XboxController


@configclass
class Se2ControllerCfg:
    sim_device: str = "cpu"
    ranges: UniformVelocityCommandCfg.Ranges = MISSING
    dead_zone: float = 0.01
    """Dead zone applied to the normalized stick value, radially for the translation stick."""
    yaw_dead_zone: float | None = None
    """Dead zone for the yaw stick. Falls back to :attr:`dead_zone` when None."""
    input_device: str = "xbox"


class Se2Controller:
    """A gamepad controller for sending SE(2) velocity commands.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      left stick up              left stick down
        Move along y-axis      left stick right           left stick left
        Rotate along z-axis    right stick right          right stick left
        ====================== ========================= ========================
    """

    def __init__(self, cfg: Se2ControllerCfg):
        self.cfg = cfg
        self.input_device = XboxController()

    def __str__(self) -> str:
        msg = f"Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tDevice: {self.cfg.input_device}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove in X-Y plane: left stick\n"
        msg += "\tRotate in Z-axis: right stick\n"
        msg += "\t----------------------------------------------\n"
        msg += f"\tlin x range: [{self.cfg.ranges.lin_vel_x[0]}, {self.cfg.ranges.lin_vel_x[1]}]\n"
        msg += f"\tlin y range: [{self.cfg.ranges.lin_vel_y[0]}, {self.cfg.ranges.lin_vel_y[1]}]\n"
        msg += f"\tang z range: [{self.cfg.ranges.ang_vel_z[0]}, {self.cfg.ranges.ang_vel_z[1]}]\n"
        msg += f"\tdead zone: {self.cfg.dead_zone} (yaw: {self._yaw_dead_zone})\n"
        return msg


    def read_command(self) -> torch.Tensor:
        # no device attached (unplugged / read error): never latch the last stick value
        if not self.input_device.connected:
            return torch.zeros(3, dtype=torch.float32, device=self.cfg.sim_device)

        # xbox driver returns LeftJoystickX in screen coords (right=+); REP-103 robot
        # body frame has +y = LEFT, so flip the sign for "stick right -> robot right".
        lin_x = self.input_device.LeftJoystickY
        lin_y = -self.input_device.LeftJoystickX
        ang_z = self.input_device.RightJoystickX

        # dead zone: radial for the translation stick, scalar for the yaw stick
        lin_x, lin_y = self._apply_radial_dead_zone(lin_x, lin_y, self.cfg.dead_zone)
        ang_z = self._apply_dead_zone(ang_z, self._yaw_dead_zone)

        ranges = self.cfg.ranges
        command = [
            self._scale(lin_x, ranges.lin_vel_x),
            self._scale(lin_y, ranges.lin_vel_y),
            self._scale(ang_z, ranges.ang_vel_z),
        ]
        return torch.tensor(command, dtype=torch.float32, device=self.cfg.sim_device)


    @property
    def _yaw_dead_zone(self) -> float:
        return self.cfg.dead_zone if self.cfg.yaw_dead_zone is None else self.cfg.yaw_dead_zone

    @staticmethod
    def _apply_dead_zone(value: float, dead_zone: float) -> float:
        """Zero out values inside the dead zone and rescale the rest back to full range."""
        if abs(value) <= dead_zone:
            return 0.0
        scaled = (abs(value) - dead_zone) / max(1.0 - dead_zone, 1e-6)
        return math.copysign(min(scaled, 1.0), value)

    @staticmethod
    def _apply_radial_dead_zone(x: float, y: float, dead_zone: float) -> tuple[float, float]:
        """Dead zone on the stick magnitude, so a diagonal hold is not clipped per-axis."""
        magnitude = math.hypot(x, y)
        if magnitude <= dead_zone:
            return 0.0, 0.0
        scaled = (magnitude - dead_zone) / max(1.0 - dead_zone, 1e-6)
        gain = min(scaled, 1.0) / magnitude
        return x * gain, y * gain

    @staticmethod
    def _scale(value: float, limits: tuple[float, float]) -> float:
        """Map a normalized [-1, 1] stick value onto ``limits``, honouring asymmetric ranges."""
        lower, upper = limits
        return value * upper if value >= 0.0 else -value * lower
