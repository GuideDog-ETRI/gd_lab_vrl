"""GAST-Narrow v2 MDP terms: clearance-margin input, blocked passages -> stand, wider gdm-like commands.

Per episode (reset event) every env draws a requested side clearance m in [0.05, 0.15] m (policy input, group
``narrow_margin``). A narrow tile is *passable* for that episode when its width w >= ROBOT_WIDTH + 2 m + BAND and
*blocked* when w <= ROBOT_WIDTH + 2 m - BAND (m is redrawn inside the +/-BAND band; dead ends are always blocked).

Command on narrow tiles (gdm local planner stand-in, pushes forward even when the passage is blocked):
- vx episode constant 0.35..0.8 m/s; wz = pure pursuit toward the path point LOOKAHEAD ahead + per-episode heading
  bias (+/-8 deg, 20 % of episodes +/-20 deg), clip +/-0.8 rad/s, update period 0.05/0.1/0.2 s, noise.
- 30 % of episodes: vy windows (+/-0.2 m/s for 1-3 s, alternating with 1-3 s of vy = 0).
- 20 % of passable episodes: one stop window (command 0 for 1-3 s) somewhere in the passage.
- zero after the exit line.
Blocked: within ZONE of the block line the tracking rewards are off; standing still there pays, crossing the line
(robot front past it) ends the episode as a failure; 1.5 s still in the zone = success.
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from gd_lab.gast_narrow import geometry
from gd_lab.gast_narrow.mdp import NarrowVelocityCommand
from gd_lab.gast_narrow.state import SUCCESS_HOLD_S, _wrap, narrow_state

ROBOT_WIDTH = 0.48          # 6566 swept width walking straight (clearance points incl. foot radius), measured
MARGIN_RANGE = (0.05, 0.15)
MARGIN_MID, MARGIN_HALF = 0.10, 0.05
BAND = 0.04                 # 5 cm map cells: no draws within +/-4 cm of the pass/block threshold
VX_RANGE = (0.35, 0.80)
WZ_MAX, WZ_GAIN, LOOKAHEAD = 0.8, 1.5, 0.8
BIAS_DEG, BIG_BIAS_DEG, BIG_BIAS_P = 8.0, 20.0, 0.2
NOISE_WZ, UPDATE_CHOICES = 0.05, (0.05, 0.1, 0.2)
VY_P, VY_MAX, VY_WINDOW = 0.3, 0.2, (1.0, 3.0)
STOP_P, STOP_WINDOW = 0.2, (1.0, 3.0)
SPAWN_X, SPAWN_Y, SPAWN_YAW_DEG = (-3.0, -2.2), 0.40, 35.0
FRONT = 0.45                # trunk front (0.40) + head clearance
ZONE = 1.2
HOLD_S, STILL_SPEED = 1.5, 0.10
TOUCH = 0.005


class NarrowEpisode:
    """Per-env episode draws shared by the command, rewards, observations and Top-5 (``env._narrow_v2``)."""

    def __init__(self, env):
        n, d = env.num_envs, env.device
        self.margin = torch.full((n,), MARGIN_MID, device=d)
        self.blocked = torch.zeros(n, dtype=torch.bool, device=d)
        self.slack = torch.zeros(n, device=d)          # w - (ROBOT_WIDTH + 2 m), narrow tiles


def narrow_episode(env) -> NarrowEpisode:
    episode = getattr(env, "_narrow_v2", None)
    if episode is None:
        episode = NarrowEpisode(env)
        env._narrow_v2 = episode
    return episode


def draw_margin(width, dead_end, generator_rand=torch.rand):
    """width [n], dead_end [n] bool -> (margin [n], blocked [n]) with no draw inside the +/-BAND band."""
    lo, hi = MARGIN_RANGE
    m = lo + (hi - lo) * generator_rand(width.shape, device=width.device)
    threshold = ROBOT_WIDTH + 2 * m
    bad = ~dead_end & ((width - threshold).abs() < BAND)
    for _ in range(8):
        if not bool(bad.any()):
            break
        m = torch.where(bad, lo + (hi - lo) * generator_rand(width.shape, device=width.device), m)
        bad = ~dead_end & ((width - (ROBOT_WIDTH + 2 * m)).abs() < BAND)
    if bool(bad.any()):  # push to the nearer valid side (one always exists: the band is narrower than the range)
        m_pass = (width - BAND - ROBOT_WIDTH) / 2 - 1e-3
        m_block = (width + BAND - ROBOT_WIDTH) / 2 + 1e-3
        fix = torch.where(m_pass >= lo, m_pass, m_block).clamp(lo, hi)
        m = torch.where(bad, fix, m)
    blocked = dead_end | (width < ROBOT_WIDTH + 2 * m)
    return m, blocked


def point_at(path, s):
    """Path point [N,2] at arclength s (clamped to the path end)."""
    seg = path[:, 1:] - path[:, :-1]
    length = seg.norm(dim=-1)
    start = torch.cumsum(torch.cat((torch.zeros_like(length[:, :1]), length[:, :-1]), -1), -1)
    s = s.clamp(min=0.0)
    inside = (s[:, None] >= start) & (length > 1e-6)
    k = (inside.float() * torch.arange(1, seg.shape[1] + 1, device=path.device)).argmax(-1)
    rows = torch.arange(path.shape[0], device=path.device)
    t = ((s - start[rows, k]) / length[rows, k].clamp_min(1e-6)).clamp(0.0, 1.0)
    return path[rows, k] + t[:, None] * seg[rows, k]


def zone_state(env):
    """(state, episode, front arclength, in-zone mask, breached mask)."""
    state = narrow_state(env)
    episode = narrow_episode(env)
    front = state.s + FRONT * torch.cos(state.heading_error)
    blocked = state.narrow & episode.blocked
    zone = blocked & (front >= state.block_s - ZONE)
    breach = blocked & (front > state.block_s + 0.02)
    return state, episode, front, zone, breach


def reset_narrow_v2(env, env_ids):
    """After reset_base: margin/blocked draw for every env, spawn before the passage on narrow tiles."""
    state = getattr(env, "_narrow_state", None)
    if state is not None:
        state.reset_counter += 1
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    state = narrow_state(env)
    episode = narrow_episode(env)
    width = state.width[ids]
    dead_end = state.kind[ids] == geometry.KINDS["dead_end"]
    m, blocked = draw_margin(width, dead_end)
    narrow = state.narrow[ids]
    episode.margin[ids] = m
    episode.blocked[ids] = blocked & narrow
    episode.slack[ids] = torch.where(narrow, width - (ROBOT_WIDTH + 2 * m), torch.zeros_like(m))
    ids = ids[narrow]
    if not len(ids):
        return
    robot = env.scene["robot"]
    n = len(ids)
    rand = lambda: torch.rand(n, device=env.device)  # noqa: E731
    pose = robot.data.default_root_state[ids, :7].clone()
    origin = env.scene.env_origins[ids]
    pose[:, 0] = origin[:, 0] + SPAWN_X[0] + (SPAWN_X[1] - SPAWN_X[0]) * rand()
    pose[:, 1] = origin[:, 1] + (rand() * 2 - 1) * SPAWN_Y
    pose[:, 2] = origin[:, 2] + pose[:, 2]
    yaw = (rand() * 2 - 1) * math.radians(SPAWN_YAW_DEG)
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 4:6] = 0.0
    pose[:, 6] = torch.sin(yaw / 2)
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
    state.reset_counter += 1


class NarrowV2VelocityCommand(NarrowVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, d = self.num_envs, self.device
        self.update_s = torch.full((n,), 0.2, device=d)
        self.vy_episode = torch.zeros(n, dtype=torch.bool, device=d)
        self.vy_on = torch.zeros(n, dtype=torch.bool, device=d)
        self.vy_value = torch.zeros(n, device=d)
        self.vy_timer = torch.zeros(n, device=d)
        self.stop_s = torch.full((n,), 1e6, device=d)
        self.stop_left = torch.zeros(n, device=d)
        self.stop_used = torch.zeros(n, dtype=torch.bool, device=d)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if len(ids):
            n, d = len(ids), self.device
            rand = lambda: torch.rand(n, device=d)  # noqa: E731
            big = rand() < BIG_BIAS_P
            bias = torch.where(big, torch.full((n,), BIG_BIAS_DEG, device=d), torch.full((n,), BIAS_DEG, device=d))
            choices = torch.tensor(UPDATE_CHOICES, device=d)
            self.update_s[ids] = choices[torch.randint(0, len(UPDATE_CHOICES), (n,), device=d)]
            self.vy_episode[ids] = rand() < VY_P
            self.vy_on[ids] = False
            self.vy_value[ids] = 0.0
            self.vy_timer[ids] = VY_WINDOW[0] + (VY_WINDOW[1] - VY_WINDOW[0]) * rand()
            state = narrow_state(self._env)
            episode = narrow_episode(self._env)
            stop = (rand() < STOP_P) & ~episode.blocked[ids]
            self.stop_s[ids] = torch.where(stop, state.block_s[ids] + 2.0 * rand(), torch.full((n,), 1e6, device=d))
            self.stop_left[ids] = STOP_WINDOW[0] + (STOP_WINDOW[1] - STOP_WINDOW[0]) * rand()
            self.stop_used[ids] = False
        out = super().reset(env_ids)
        if len(ids):  # after super(): v1 draws its vx/bias there; v2 replaces them
            lo, hi = VX_RANGE
            self.narrow_vx[ids] = lo + (hi - lo) * torch.rand(len(ids), device=self.device)
            self.narrow_bias[ids] = (torch.rand(len(ids), device=self.device) * 2 - 1) * torch.deg2rad(bias)
        return out

    def _apply_narrow(self):
        env = self._env
        state = narrow_state(env)
        mask = state.narrow
        if not bool(mask.any()):
            return
        dt = env.step_dt
        self.narrow_timer -= dt
        due = mask & (self.narrow_timer <= 0)
        if bool(due.any()):
            goal = point_at(state.path_e, state.s + LOOKAHEAD)
            target = torch.atan2(goal[:, 1] - state.local[:, 1], goal[:, 0] - state.local[:, 0]) + self.narrow_bias
            wz = (WZ_GAIN * _wrap(target - state.yaw)).clamp(-WZ_MAX, WZ_MAX)
            wz = (wz + torch.randn_like(wz) * NOISE_WZ).clamp(-WZ_MAX, WZ_MAX)
            self.narrow_wz = torch.where(due, wz, self.narrow_wz)
            self.narrow_timer = torch.where(due, self.update_s, self.narrow_timer)
        # vy windows
        self.vy_timer -= dt
        flip = self.vy_episode & (self.vy_timer <= 0)
        if bool(flip.any()):
            n = int(flip.sum())
            self.vy_on[flip] = ~self.vy_on[flip]
            self.vy_value[flip] = (torch.rand(n, device=self.device) * 2 - 1) * VY_MAX
            self.vy_timer[flip] = VY_WINDOW[0] + (VY_WINDOW[1] - VY_WINDOW[0]) * torch.rand(n, device=self.device)
        vy = torch.where(self.vy_on, self.vy_value, torch.zeros_like(self.vy_value))
        # one stop window in the passage
        stopping = mask & ~self.stop_used & (state.s >= self.stop_s)
        self.stop_left = torch.where(stopping, self.stop_left - dt, self.stop_left)
        self.stop_used |= stopping & (self.stop_left <= 0)
        zero = state.past_exit | (stopping & ~self.stop_used)
        vx = torch.where(zero, torch.zeros_like(self.narrow_vx), self.narrow_vx)
        wz = torch.where(zero, torch.zeros_like(self.narrow_wz), self.narrow_wz)
        vy = torch.where(zero, torch.zeros_like(vy), vy)
        m = mask
        self.vel_command_b[m, 0] = vx[m]
        self.vel_command_b[m, 1] = vy[m]
        self.vel_command_b[m, 2] = wz[m]
        self.is_heading_env[m] = False
        self.is_standing_env[m] = False
        self.backward[m] = False
        self._pulse_active[m] = False


class NarrowV2Monitor(ManagerTermBase):
    """Weight-1 reward term returning 0: per-episode tracker (pass, hold, breach, wall touch)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        zeros = lambda dtype=torch.float32: torch.zeros(n, dtype=dtype, device=d)  # noqa: E731
        self.passed, self.held, self.breached = zeros(torch.bool), zeros(torch.bool), zeros(torch.bool)
        self.success, self.success_time = zeros(torch.bool), zeros()
        self.hold_time, self.touch_time = zeros(), zeros()
        self.min_clearance = torch.full((n,), 9.0, device=d)
        self.pass_event, self.hold_event, self.breach_event = zeros(torch.bool), zeros(torch.bool), zeros(torch.bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("passed", "held", "breached", "success", "pass_event", "hold_event", "breach_event"):
            getattr(self, name)[ids] = False
        self.success_time[ids] = 0.0
        self.hold_time[ids] = 0.0
        self.touch_time[ids] = 0.0
        self.min_clearance[ids] = 9.0
        state = getattr(self._env, "_narrow_state", None)
        if state is not None:
            state.reset_counter += 1

    def __call__(self, env):
        state, episode, _, zone, breach = zone_state(env)
        dt = env.step_dt
        m = state.narrow
        clearance = state.clearance.amin(-1)
        self.min_clearance = torch.where(m, torch.minimum(self.min_clearance, clearance), self.min_clearance)
        self.touch_time += (m & (clearance < TOUCH)).float() * dt
        passable = m & ~episode.blocked
        new_pass = passable & state.past_exit & ~self.passed
        self.passed |= new_pass
        speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
        still = zone & ~breach & ~self.breached & (speed < STILL_SPEED)
        self.hold_time = torch.where(still, self.hold_time + dt, torch.zeros_like(self.hold_time))
        new_hold = still & (self.hold_time >= HOLD_S) & ~self.held
        self.held |= new_hold
        new_breach = breach & ~self.breached & ~self.held
        self.breached |= new_breach
        self.success = self.passed | self.held
        self.success_time += self.success.float() * dt
        self.pass_event, self.hold_event, self.breach_event = new_pass, new_hold, new_breach
        return torch.zeros(env.num_envs, device=env.device)


def _monitor(env) -> NarrowV2Monitor:
    return env.reward_manager.get_term_cfg("narrow_monitor").func


def narrow_pass_bonus(env):
    return _monitor(env).pass_event.float() / env.step_dt


def narrow_hold_bonus(env):
    return _monitor(env).hold_event.float() / env.step_dt


def narrow_breach_penalty(env):
    return _monitor(env).breach_event.float() / env.step_dt


def narrow_stand_in_zone(env, sigma: float = 0.10):
    """Per second, blocked episodes inside the zone before the line: exp(-(planar speed / sigma)^2)."""
    _, _, _, zone, breach = zone_state(env)
    speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
    return torch.exp(-(speed / sigma) ** 2) * (zone & ~breach).float()


def narrow_wall_proximity_v2(env):
    """Per second: sum of margin intrusions, margin = requested clearance m - BAND/2 (>= 2 cm)."""
    state = narrow_state(env)
    margin = (narrow_episode(env).margin - BAND / 2).clamp_min(0.02)[:, None]
    intrusion = ((margin - state.clearance) / margin).clamp(0.0, 1.0).sum(-1)
    return intrusion * state.narrow.float()


def _not_in_zone(env):
    _, _, _, zone, _ = zone_state(env)
    return (~zone).float()


def track_lin_vel_xy_gated(env, command_name: str, std_base: float = 0.3, alpha: float = 0.2,
                           cmd_threshold: float = 0.1, vel_frac_low: float = 0.4, vel_frac_high: float = 0.8,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    from gd_lab.mdp.rewards import track_lin_vel_xy_velocity_scaled_exp
    reward = track_lin_vel_xy_velocity_scaled_exp(env, command_name, std_base, alpha, cmd_threshold, vel_frac_low,
                                                  vel_frac_high, asset_cfg)
    return reward * _not_in_zone(env)


def track_ang_vel_z_gated(env, command_name: str, std_base: float = 0.3, alpha: float = 0.3,
                          cmd_threshold: float = 0.1, vel_frac_low: float = 0.4, vel_frac_high: float = 0.8,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    from gd_lab.mdp.rewards import track_ang_vel_z_velocity_scaled_exp
    reward = track_ang_vel_z_velocity_scaled_exp(env, command_name, std_base, alpha, cmd_threshold, vel_frac_low,
                                                 vel_frac_high, asset_cfg)
    return reward * _not_in_zone(env)


def narrow_success_done_v2(env, hold_s: float = SUCCESS_HOLD_S):
    return _monitor(env).success_time >= hold_s


def narrow_breach_done(env):
    return _monitor(env).breached


def narrow_margin_obs(env):
    """Policy input [N,1]: (m - 0.10) / 0.05 in [-1, 1]."""
    return ((narrow_episode(env).margin - MARGIN_MID) / MARGIN_HALF)[:, None]


def narrow_privileged_v2(env):
    """Critic-only [N,9]: v1's 6 + [margin (norm), blocked, (block line - front) / 2 clipped]."""
    state, episode, front, _, _ = zone_state(env)
    m = state.narrow.float()
    progress = (state.s / state.total.clamp_min(1e-6)).clamp(0.0, 1.0)
    base = torch.stack((state.d_left.clamp(0.0, 1.0), state.d_right.clamp(0.0, 1.0), state.lateral.clamp(-1.0, 1.0),
                        state.heading_error, progress, torch.ones_like(m)), -1) * m[:, None]
    to_block = ((state.block_s - front) / 2.0).clamp(-1.0, 1.0) * m
    extra = torch.stack(((episode.margin - MARGIN_MID) / MARGIN_HALF, episode.blocked.float(), to_block), -1)
    return torch.cat((base, extra), -1)


def narrow_levels_v2(env, env_ids, min_moving_time: float = 2.0):
    """Narrow tiles: up after success (pass or hold) with < 0.2 s wall touch, down on failure. Others: 6566 rule."""
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
