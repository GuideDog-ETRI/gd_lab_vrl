"""Velocity command with a resample deadband and an operator-style pulse train.

The deadband makes "stand" a distinct command instead of a slow crawl the
tracking term and the stand-still gates disagree about. The pulse train toggles
the raw command cmd <-> 0 mid-window, training the stop / re-launch transition a
deploy joystick produces and a constant-per-window command never exercises.

The pulse breaks the stock terrain curriculum, which grades distance against a
fixed bar and the reset-instant command - an env that obeyed its zero holds is
demoted for obeying. The per-episode accumulators here are what
``terrain_levels_vel_cmd_aware`` grades against instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

DEADBAND = 0.1
"""Resampled commands with ``||cmd_xy||`` at or below this are zeroed (m/s)."""


class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` + resample deadband + command-pulse train."""

    cfg: UniformThresholdVelocityCommandCfg

    def __init__(self, cfg: UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pulse_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pulse_zero_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pulse_timer = torch.zeros(self.num_envs, device=self.device)
        self._pulse_base_cmd_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # Read by the terrain curriculum before the command manager resets
        # (CurriculumManager.compute runs first).
        self.cmd_dist_integral = torch.zeros(self.num_envs, device=self.device)
        self.cmd_moving_time = torch.zeros(self.num_envs, device=self.device)
        self.cmd_elapsed_time = torch.zeros(self.num_envs, device=self.device)

    def _hold(self, n: int, hold_range: tuple[float, float]) -> torch.Tensor:
        lo, hi = hold_range
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        moving = torch.norm(self.vel_command_b[env_ids, :2], dim=1) > DEADBAND
        self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)
        if self.cfg.pulse_prob <= 0.0:
            return
        roll = torch.rand(len(env_ids), device=self.device) < self.cfg.pulse_prob
        self._pulse_active[env_ids] = moving & ~self.is_standing_env[env_ids] & roll
        self._pulse_zero_phase[env_ids] = False  # a window starts in the cmd phase
        self._pulse_timer[env_ids] = self._hold(len(env_ids), self.cfg.pulse_hold_range)
        self._pulse_base_cmd_xy[env_ids] = self.vel_command_b[env_ids, :2]

    def _redraw_pulse_command(self, mask: torch.Tensor) -> None:
        """Re-roll the base command for envs re-entering the cmd phase.

        ``cfg.ranges`` is live, so a redraw respects the current curriculum
        range. A draw inside the deadband keeps the previous base command -
        adopting the zeroed value would stall the train for the rest of the
        window.
        """
        redraw = mask & (torch.rand(self.num_envs, device=self.device) < self.cfg.pulse_resample_prob)
        ids = redraw.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        new_x = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.ranges.lin_vel_x)
        new_y = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.ranges.lin_vel_y)
        new_xy = torch.stack((new_x, new_y), dim=1)
        keep = torch.norm(new_xy, dim=1) > DEADBAND
        adopt = ids[keep]
        if len(adopt) == 0:
            return
        self._pulse_base_cmd_xy[adopt] = new_xy[keep]
        if self.cfg.heading_command:
            # wz follows next step via the heading controller in super().
            self.heading_target[adopt] = torch.empty(len(adopt), device=self.device).uniform_(*self.cfg.ranges.heading)

    def _advance_pulse(self) -> None:
        self._pulse_timer[self._pulse_active] -= self._env.step_dt
        flip = self._pulse_active & (self._pulse_timer <= 0.0)
        if flip.any():
            self._pulse_zero_phase[flip] = ~self._pulse_zero_phase[flip]
            to_zero = flip & self._pulse_zero_phase
            to_cmd = flip & ~self._pulse_zero_phase
            for mask, hold_range in (
                (to_zero, self.cfg.pulse_zero_hold_range),
                (to_cmd, self.cfg.pulse_hold_range),
            ):
                if mask.any():
                    self._pulse_timer[mask] = self._hold(int(mask.sum()), hold_range)
            if to_cmd.any():
                self._redraw_pulse_command(to_cmd)
        zero_now = self._pulse_active & self._pulse_zero_phase
        cmd_now = self._pulse_active & ~self._pulse_zero_phase
        if zero_now.any():
            self.vel_command_b[zero_now] = 0.0
        if cmd_now.any():
            self.vel_command_b[cmd_now, :2] = self._pulse_base_cmd_xy[cmd_now]

    def _update_command(self):
        super()._update_command()
        if self.cfg.pulse_prob > 0.0 and self._pulse_active.any():
            self._advance_pulse()
        # After every mutation of the raw command, so the accumulators count
        # what the robot was actually asked to do.
        dt = self._env.step_dt
        cmd_mag = torch.norm(self.vel_command_b[:, :2], dim=1)
        self.cmd_dist_integral += cmd_mag * dt
        self.cmd_moving_time += (cmd_mag > DEADBAND).float() * dt
        self.cmd_elapsed_time += dt

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.cmd_dist_integral[ids] = 0.0
        self.cmd_moving_time[ids] = 0.0
        self.cmd_elapsed_time[ids] = 0.0
        return extras


@configclass
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformThresholdVelocityCommand

    pulse_prob: float = 0.0
    """Per-window probability of a cmd <-> 0 pulse train; 0 disables it (the
    deadband still applies)."""

    pulse_hold_range: tuple[float, float] = (0.3, 1.0)
    """Seconds the CMD phase holds before dropping to zero, redrawn at each entry."""

    pulse_zero_hold_range: tuple[float, float] = (0.2, 2.0)
    """Seconds the ZERO phase holds. Longer tail than the cmd hold: a real stop
    must be held, not ridden out on momentum."""

    pulse_resample_prob: float = 0.5
    """Probability that a cmd-phase return redraws the command - an operator
    re-launch is not always in the old direction."""
