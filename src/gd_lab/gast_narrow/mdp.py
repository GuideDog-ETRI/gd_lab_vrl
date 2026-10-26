"""Narrow-passage MDP terms. Non-narrow (retention) tiles keep the 6566 task terms unchanged.

Command on narrow tiles = a coarse gdm local planner: forward vx (episode constant, 0.35..0.6 m/s), vy = 0, and a
turn rate toward the path tangent 0.5 m ahead, with a per-episode heading bias (+/-8 deg), 5 Hz updates and noise.
It never steers on the lateral error: keeping off the walls is the policy's job (head LiDAR map).
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase

from gd_lab.gast_backgap.tasks import BackGapVelocityCommand
from gd_lab.gast_narrow import geometry
from gd_lab.gast_narrow.state import SUCCESS_HOLD_S, _wrap, narrow_state, tangent_at

VX_RANGE = (0.35, 0.60)
WZ_MAX, WZ_GAIN, LOOKAHEAD = 0.6, 1.5, 0.5
BIAS_DEG, NOISE_WZ, UPDATE_S = 8.0, 0.05, 0.2
SPAWN_Y, SPAWN_YAW_DEG = 0.15, 15.0
CLEAR_MARGIN = 0.05
TOUCH = 0.005


class NarrowVelocityCommand(BackGapVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, d = self.num_envs, self.device
        self.narrow_vx = torch.zeros(n, device=d)
        self.narrow_bias = torch.zeros(n, device=d)
        self.narrow_wz = torch.zeros(n, device=d)
        self.narrow_timer = torch.zeros(n, device=d)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if len(ids):  # per-episode values, drawn before super().reset() resamples (and applies) the command
            lo, hi = VX_RANGE
            self.narrow_vx[ids] = lo + (hi - lo) * torch.rand(len(ids), device=self.device)
            self.narrow_bias[ids] = (torch.rand(len(ids), device=self.device) * 2 - 1) * math.radians(BIAS_DEG)
            self.narrow_timer[ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._apply_narrow()

    def _update_command(self):
        super()._update_command()
        self._apply_narrow()

    def _apply_narrow(self):
        state = narrow_state(self._env)
        mask = state.narrow
        if not bool(mask.any()):
            return
        self.narrow_timer -= self._env.step_dt
        due = mask & (self.narrow_timer <= 0)
        if bool(due.any()):
            target = tangent_at(state.path_e, state.s + LOOKAHEAD) + self.narrow_bias
            wz = (WZ_GAIN * _wrap(target - state.yaw)).clamp(-WZ_MAX, WZ_MAX)
            wz = wz + torch.randn_like(wz) * NOISE_WZ
            self.narrow_wz = torch.where(due, wz, self.narrow_wz)
            self.narrow_timer = torch.where(due, torch.full_like(self.narrow_timer, UPDATE_S), self.narrow_timer)
        done = state.past_exit
        vx = torch.where(done, torch.zeros_like(self.narrow_vx), self.narrow_vx)
        wz = torch.where(done, torch.zeros_like(self.narrow_wz), self.narrow_wz)
        m = mask
        self.vel_command_b[m, 0] = vx[m]
        self.vel_command_b[m, 1] = 0.0
        self.vel_command_b[m, 2] = wz[m]
        self.is_heading_env[m] = False
        self.is_standing_env[m] = False
        self.backward[m] = False
        self._pulse_active[m] = False


def reset_narrow_pose(env, env_ids):
    """After reset_base: narrow-tile envs start before the passage, facing +x, small offset and yaw error."""
    state = getattr(env, "_narrow_state", None)
    if state is not None:
        state.reset_counter += 1
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    state = narrow_state(env)
    ids = ids[state.narrow[ids]]
    if not len(ids):
        return
    robot = env.scene["robot"]
    n = len(ids)
    pose = robot.data.default_root_state[ids, :7].clone()
    origin = env.scene.env_origins[ids]
    pose[:, 0] = origin[:, 0] + geometry.SPAWN_X
    pose[:, 1] = origin[:, 1] + (torch.rand(n, device=env.device) * 2 - 1) * SPAWN_Y
    pose[:, 2] = origin[:, 2] + pose[:, 2]
    yaw = (torch.rand(n, device=env.device) * 2 - 1) * math.radians(SPAWN_YAW_DEG)
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 4:6] = 0.0
    pose[:, 6] = torch.sin(yaw / 2)
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
    state.reset_counter += 1


class NarrowMonitor(ManagerTermBase):
    """Reward term with weight 1 that returns 0: tracks success / wall touch per episode (read by other terms,
    the curriculum, the termination and the Top-5 collector)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self.success = torch.zeros(n, dtype=torch.bool, device=d)
        self.success_time = torch.zeros(n, device=d)
        self.touch_time = torch.zeros(n, device=d)
        self.min_clearance = torch.full((n,), 9.0, device=d)
        self.paid = torch.zeros(n, dtype=torch.bool, device=d)
        self.event = torch.zeros(n, dtype=torch.bool, device=d)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.success[ids] = False
        self.success_time[ids] = 0.0
        self.touch_time[ids] = 0.0
        self.min_clearance[ids] = 9.0
        self.paid[ids] = False
        self.event[ids] = False
        state = getattr(self._env, "_narrow_state", None)
        if state is not None:
            state.reset_counter += 1

    def __call__(self, env):
        state = narrow_state(env)
        dt = env.step_dt
        m = state.narrow
        clearance = state.clearance.amin(-1)
        self.min_clearance = torch.where(m, torch.minimum(self.min_clearance, clearance), self.min_clearance)
        self.touch_time += (m & (clearance < TOUCH)).float() * dt
        self.success |= state.past_exit
        self.success_time += self.success.float() * dt
        self.event = self.success & ~self.paid
        self.paid |= self.event
        return torch.zeros(env.num_envs, device=env.device)


def _monitor(env) -> NarrowMonitor:
    return env.reward_manager.get_term_cfg("narrow_monitor").func


def narrow_success_bonus(env):
    """One-shot bonus when the robot passes the exit line (divided by dt: the manager multiplies by dt)."""
    return _monitor(env).event.float() / env.step_dt


def narrow_wall_proximity(env, margin: float = CLEAR_MARGIN):
    """Per second: sum over robot points of the margin intrusion (0 at >= margin, 1 at contact)."""
    state = narrow_state(env)
    intrusion = ((margin - state.clearance) / margin).clamp(0.0, 1.0).sum(-1)
    return intrusion * state.narrow.float()


def narrow_centering(env, sigma: float = 0.10):
    """Per second: exp(-(lateral / sigma)^2) on narrow tiles before the exit."""
    state = narrow_state(env)
    active = state.narrow & ~state.past_exit
    return torch.exp(-(state.lateral / sigma) ** 2) * active.float()


def narrow_success_done(env, hold_s: float = SUCCESS_HOLD_S):
    """Time-out style end of a narrow episode 0.5 s after the exit line (more passages per env)."""
    return _monitor(env).success_time >= hold_s


def narrow_privileged(env):
    """Critic-only: [d_left, d_right (clip 1 m), lateral, heading error, progress fraction, narrow flag]."""
    state = narrow_state(env)
    m = state.narrow.float()
    progress = (state.s / state.total.clamp_min(1e-6)).clamp(0.0, 1.0)
    out = torch.stack((state.d_left.clamp(0.0, 1.0), state.d_right.clamp(0.0, 1.0), state.lateral.clamp(-1.0, 1.0),
                       state.heading_error, progress, torch.ones_like(m)), -1)
    return out * m[:, None]


def narrow_levels(env, env_ids, min_moving_time: float = 2.0):
    """Narrow tiles: up after a passage with < 0.2 s wall touch, down on failure. Others: the 6566 rule."""
    from gd_lab.mdp.platform_gap_terms import platform_gap_levels
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    state = narrow_state(env)
    mask = state.narrow[ids]
    other = ids[~mask]
    if len(other):
        platform_gap_levels(env, other, min_moving_time)
    nar = ids[mask]
    if len(nar):
        monitor = _monitor(env)
        command = env.command_manager.get_term("base_velocity")
        eligible = command.cmd_moving_time[nar] > min_moving_time
        failed = env.termination_manager.terminated[nar]
        up = eligible & monitor.success[nar] & (monitor.touch_time[nar] < 0.2) & ~failed
        down = eligible & ~up & (failed | ~monitor.success[nar])
        env.scene.terrain.update_env_origins(nar, up, down)
    return env.scene.terrain.terrain_levels.float().mean()
