"""GAST-Narrow v2.4 turn-in-place terms (design: docs/design/GAST_NARROW_V24_TURN_DESIGN.md).

Turn tiles (geometry kind "turn"): the robot starts at the pocket centre and gets vx = vy = 0, wz = direction * rate
until it has turned a target angle. Per episode the requested clearance m in [0.05, 0.15] (the v2 margin input) decides
whether the turn is possible: turnable = r_free >= R_SWEEP + m (no draw within +/-TURN_BAND of the threshold).
- turnable: success = target angle reached with the centre drift <= DRIFT_OK the whole time (-> the v2 pass event, so
  the paced pass bonus pays); the command then goes to 0.
- not turnable: the command still asks to turn; success = HOLD_S still (|wz| < STILL_WZ, speed < STILL_SPEED) with less
  than BREACH_DEG turned (-> the v2 hold event); turning more than BREACH_DEG = breach (-> the v2 breach event, ends).
- drift above DRIFT_FAIL ends the episode (failure) on every turn tile, and so does reaching the target angle with a
  drift above DRIFT_OK on the way (the turn is done but not clean: no pass, no point in standing until the time-out).
The v2 narrow zone logic (blocked passages along a path) is off on turn tiles: episode.blocked stays False there.
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg

from gd_lab.gast_narrow import geometry, mdp_v2
from gd_lab.gast_narrow.state import _wrap, narrow_state, rect_distance

TURN = geometry.KINDS["turn"]
R_SWEEP = 0.46              # radius swept by the trunk / legs (incl. point radii) turning in place, standing stance
TURN_BAND = 0.04
RATE = (0.3, 0.8)           # rad/s
RIGHT_P = 0.7               # right turn (wz < 0); deployment sends right only (handler at the right-rear)
ANGLE_DEG = (45.0, 180.0)
DONE_TOL_DEG = 5.0
PAUSE_P, PAUSE_S = 0.2, (1.0, 2.0)
DRIFT_FREE, DRIFT_OK, DRIFT_FAIL = 0.05, 0.10, 0.20
BREACH_DEG, STILL_WZ = 15.0, 0.15
SPAWN_JITTER, SPAWN_YAW_NOISE_DEG = 0.03, 15.0
# Spawn footprint (trunk + legs in stance, half sizes) for choosing an axis-aligned start yaw that fits the pocket.
FOOTPRINT = [(x, y) for x in (-0.45, 0.0, 0.45) for y in (-0.25, 0.25)]
OVERSPIN_MARGIN = 0.2


class TurnEpisode:
    """Per-env turn draws and tracker shared by the command, monitor, rewards and observations (``env._narrow_turn``)."""

    def __init__(self, env):
        n, d = env.num_envs, env.device
        self.turn = torch.zeros(n, dtype=torch.bool, device=d)
        self.turnable = torch.zeros(n, dtype=torch.bool, device=d)
        self.start = torch.zeros(n, 2, device=d)
        self.last_yaw = torch.zeros(n, device=d)
        self.turned = torch.zeros(n, device=d)        # signed, accumulated (rad)
        self.target = torch.zeros(n, device=d)        # rad, > 0
        self.direction = torch.ones(n, device=d)      # +1 left, -1 right
        self.drift = torch.zeros(n, device=d)
        self.drift_max = torch.zeros(n, device=d)
        self.reached = torch.zeros(n, dtype=torch.bool, device=d)
        self.hold_time = torch.zeros(n, device=d)


def turn_episode(env) -> TurnEpisode:
    episode = getattr(env, "_narrow_turn", None)
    if episode is None:
        episode = TurnEpisode(env)
        env._narrow_turn = episode
    return episode


def draw_turn_margin(radius, generator_rand=torch.rand):
    """radius [n] -> (margin [n], turnable [n]) with no draw inside +/-TURN_BAND of r = R_SWEEP + m."""
    lo, hi = mdp_v2.MARGIN_RANGE
    m = lo + (hi - lo) * generator_rand(radius.shape, device=radius.device)
    bad = (radius - (R_SWEEP + m)).abs() < TURN_BAND
    for _ in range(8):
        if not bool(bad.any()):
            break
        m = torch.where(bad, lo + (hi - lo) * generator_rand(radius.shape, device=radius.device), m)
        bad = (radius - (R_SWEEP + m)).abs() < TURN_BAND
    if bool(bad.any()):  # nearer valid side
        m_turn = radius - TURN_BAND - R_SWEEP - 1e-3
        m_stop = radius + TURN_BAND - R_SWEEP + 1e-3
        m = torch.where(bad, torch.where(m_turn >= lo, m_turn, m_stop).clamp(lo, hi), m)
    return m, radius >= R_SWEEP + m


def _footprint_clearance(rects, xy, yaw):
    """rects [n,K,4], xy [n,2], yaw [n] -> min distance of the FOOTPRINT points to the rects [n]."""
    pts = torch.tensor(FOOTPRINT, device=xy.device)
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    world = torch.stack((xy[:, None, 0] + c * pts[:, 0] - s * pts[:, 1], xy[:, None, 1] + s * pts[:, 0] + c * pts[:, 1]), -1)
    return rect_distance(world, rects).amin(-1)


def reset_turn(env, env_ids):
    """After reset_narrow_v2: turn tiles get their own margin/turnable draw, start pose and turn target."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    state = narrow_state(env)
    turn = turn_episode(env)
    episode = mdp_v2.narrow_episode(env)
    is_turn = state.kind[ids] == TURN
    turn.turn[ids] = is_turn
    ids = ids[is_turn]
    if not len(ids):
        return
    n, d = len(ids), env.device
    rand = lambda: torch.rand(n, device=d)  # noqa: E731
    radius = state.width[ids]
    m, turnable = draw_turn_margin(radius)
    episode.margin[ids] = m
    episode.blocked[ids] = False
    episode.slack[ids] = radius - (R_SWEEP + m)
    turn.turnable[ids] = turnable
    # Start: pocket centre +/- SPAWN_JITTER; yaw = a random axis direction that fits (clearance >= 5 cm), + noise.
    xy = (torch.rand(n, 2, device=d) * 2 - 1) * SPAWN_JITTER
    rects = state.tables.rects[env.scene.terrain.terrain_levels[ids].long(), env.scene.terrain.terrain_types[ids].long()]
    axes = torch.arange(4, device=d) * (math.pi / 2)
    fits = torch.stack([_footprint_clearance(rects, xy, torch.full((n,), float(a), device=d)) for a in axes], -1)
    score = torch.where(fits >= 0.05, torch.rand(n, 4, device=d), fits - 10.0)
    yaw = axes[score.argmax(-1)]
    noisy = yaw + (rand() * 2 - 1) * math.radians(SPAWN_YAW_NOISE_DEG)
    yaw = torch.where(_footprint_clearance(rects, xy, noisy) >= 0.03, noisy, yaw)
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[ids, :7].clone()
    origin = env.scene.env_origins[ids]
    pose[:, :2] = origin[:, :2] + xy
    pose[:, 2] = origin[:, 2] + pose[:, 2]
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 4:6] = 0.0
    pose[:, 6] = torch.sin(yaw / 2)
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=d), env_ids=ids)
    state.reset_counter += 1
    turn.start[ids] = xy
    turn.last_yaw[ids] = yaw
    turn.turned[ids] = 0.0
    lo, hi = (math.radians(a) for a in ANGLE_DEG)
    turn.target[ids] = lo + (hi - lo) * rand()
    turn.direction[ids] = torch.where(rand() < RIGHT_P, -torch.ones(n, device=d), torch.ones(n, device=d))
    turn.drift[ids] = 0.0
    turn.drift_max[ids] = 0.0
    turn.reached[ids] = False
    turn.hold_time[ids] = 0.0


class NarrowV24VelocityCommand:
    """Mixin over the v2.3 command: turn tiles get vx = vy = 0 and the turn rate (see the module doc)."""

    def _init_turn(self):
        n, d = self.num_envs, self.device
        self.turn_rate = torch.zeros(n, device=d)
        self.turn_wz = torch.zeros(n, device=d)
        self.turn_timer = torch.zeros(n, device=d)
        self.pause_at = torch.full((n,), 1e6, device=d)
        self.pause_left = torch.zeros(n, device=d)

    def _reset_turn(self, ids):
        n, d = len(ids), self.device
        rand = lambda: torch.rand(n, device=d)  # noqa: E731
        self.turn_rate[ids] = RATE[0] + (RATE[1] - RATE[0]) * rand()
        self.turn_timer[ids] = 0.0
        self.pause_at[ids] = torch.where(rand() < PAUSE_P, 0.2 + 0.6 * rand(), torch.full((n,), 1e6, device=d))
        self.pause_left[ids] = PAUSE_S[0] + (PAUSE_S[1] - PAUSE_S[0]) * rand()

    def _apply_turn(self):
        env = self._env
        turn = turn_episode(env)
        mask = turn.turn & (narrow_state(env).kind == TURN)
        if not bool(mask.any()):
            return
        dt = env.step_dt
        self.turn_timer -= dt
        due = mask & (self.turn_timer <= 0)
        wz = turn.direction * self.turn_rate + torch.randn_like(self.turn_rate) * mdp_v2.NOISE_WZ
        self.turn_wz = torch.where(due, wz, self.turn_wz)
        self.turn_timer = torch.where(due, self.update_s, self.turn_timer)
        fraction = (turn.turned * turn.direction) / turn.target.clamp_min(1e-3)
        pausing = (fraction >= self.pause_at) & (self.pause_left > 0)
        self.pause_left = torch.where(mask & pausing, self.pause_left - dt, self.pause_left)
        zero = turn.reached | pausing
        command = torch.where(zero, torch.zeros_like(self.turn_wz), self.turn_wz)
        self.vel_command_b[mask, 0] = 0.0
        self.vel_command_b[mask, 1] = 0.0
        self.vel_command_b[mask, 2] = command[mask]


def update_turn(env, monitor):
    """Called by the v2.4 monitor after the v2 update: turn tracker, success / hold / breach events on turn tiles."""
    state = narrow_state(env)
    turn = turn_episode(env)
    mask = turn.turn & (state.kind == TURN)
    if not bool(mask.any()):
        return
    dt = env.step_dt
    yaw = state.yaw
    turn.turned = torch.where(mask, turn.turned + _wrap(yaw - turn.last_yaw), turn.turned)
    turn.last_yaw = yaw
    drift = (state.local - turn.start).norm(dim=-1)
    turn.drift = torch.where(mask, drift, turn.drift)
    turn.drift_max = torch.where(mask, torch.maximum(turn.drift_max, drift), turn.drift_max)
    progress = turn.turned * turn.direction
    reached = mask & turn.turnable & ~turn.reached & (progress >= turn.target - math.radians(DONE_TOL_DEG))
    turn.reached |= reached
    new_pass = reached & (turn.drift_max <= DRIFT_OK) & ~monitor.passed
    monitor.passed |= new_pass
    monitor.pass_event |= new_pass
    robot = env.scene["robot"].data
    wz = robot.root_ang_vel_b[:, 2].abs()
    speed = robot.root_lin_vel_b[:, :2].norm(dim=-1)
    stop = mask & ~turn.turnable
    over = turn.turned.abs() > math.radians(BREACH_DEG)
    still = stop & ~over & ~monitor.breached & (wz < STILL_WZ) & (speed < mdp_v2.STILL_SPEED) & (drift <= DRIFT_OK)
    turn.hold_time = torch.where(still, turn.hold_time + dt, torch.zeros_like(turn.hold_time))
    new_hold = still & (turn.hold_time >= mdp_v2.HOLD_S) & ~monitor.held
    monitor.held |= new_hold
    monitor.hold_event |= new_hold
    new_breach = stop & over & ~monitor.breached & ~monitor.held
    monitor.breached |= new_breach
    monitor.breach_event |= new_breach
    monitor.success = monitor.passed | monitor.held


class NarrowV24Monitor(mdp_v2.NarrowV2Monitor):
    def __call__(self, env):
        out = super().__call__(env)
        update_turn(env, self)
        return out


def _turn_mask(env):
    turn = turn_episode(env)
    return turn.turn & (narrow_state(env).kind == TURN)


def turn_stand(env, sigma_wz: float = 0.2, sigma_v: float = 0.10):
    """Per second on turn tiles where the turn is not possible: exp(-(wz/0.2)^2 - (v/0.1)^2)."""
    turn = turn_episode(env)
    robot = env.scene["robot"].data
    wz = robot.root_ang_vel_b[:, 2]
    speed = robot.root_lin_vel_b[:, :2].norm(dim=-1)
    active = _turn_mask(env) & ~turn.turnable & ~mdp_v2._monitor(env).breached
    return torch.exp(-(wz / sigma_wz) ** 2 - (speed / sigma_v) ** 2) * active.float()


def turn_drift(env):
    """Per second on turn tiles: centre drift above DRIFT_FREE (m)."""
    return (turn_episode(env).drift - DRIFT_FREE).clamp_min(0.0) * _turn_mask(env).float()


def turn_overspin(env):
    """Per second on turn tiles: |wz| above |commanded wz| + OVERSPIN_MARGIN (rad/s)."""
    command = env.command_manager.get_command("base_velocity")[:, 2].abs()
    wz = env.scene["robot"].data.root_ang_vel_b[:, 2].abs()
    return (wz - command - OVERSPIN_MARGIN).clamp_min(0.0) * _turn_mask(env).float()


def turn_drift_done(env):
    turn = turn_episode(env)
    drifted_turn = turn.reached & (turn.drift_max > DRIFT_OK) & ~mdp_v2._monitor(env).passed
    return _turn_mask(env) & ((turn.drift > DRIFT_FAIL) | drifted_turn)


def track_ang_vel_z_turn_gated(env, command_name: str, std_base: float = 0.3, alpha: float = 0.3,
                               cmd_threshold: float = 0.1, vel_frac_low: float = 0.4, vel_frac_high: float = 0.8,
                               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """v2's gated yaw tracking, also off on turn tiles where the turn is not possible (standing must not cost it)."""
    reward = mdp_v2.track_ang_vel_z_gated(env, command_name, std_base, alpha, cmd_threshold, vel_frac_low,
                                          vel_frac_high, asset_cfg)
    blocked = _turn_mask(env) & ~turn_episode(env).turnable
    return reward * (~blocked).float()


def narrow_centering_no_turn(env, sigma: float = 0.10):
    """v1 centering, off on turn tiles (no path there; turn_drift keeps the centre)."""
    from gd_lab.gast_narrow.mdp import narrow_centering
    return narrow_centering(env, sigma) * (~_turn_mask(env)).float()


def narrow_privileged_v24(env):
    """v2's 9 critic inputs; on turn tiles slots 2 / 4 / 7 / 8 carry the turn state (same mirror rule as v2:
    slot 2 = signed remaining turn / pi (negated by the mirror), 4 = turned fraction, 7 = not turnable,
    8 = drift / DRIFT_FAIL)."""
    priv = mdp_v2.narrow_privileged_v2(env)
    mask = _turn_mask(env)
    if not bool(mask.any()):
        return priv
    turn = turn_episode(env)
    progress = turn.turned * turn.direction
    remaining = ((turn.target - progress).clamp_min(0.0) * turn.direction / math.pi).clamp(-1.0, 1.0)
    remaining = torch.where(turn.turnable & ~turn.reached, remaining, torch.zeros_like(remaining))
    fraction = (progress / turn.target.clamp_min(1e-3)).clamp(0.0, 1.0)
    turn_cols = torch.stack((remaining, fraction, (~turn.turnable).float(), (turn.drift / DRIFT_FAIL).clamp(0.0, 1.0)), -1)
    out = priv.clone()
    out[:, [2, 4, 7, 8]] = torch.where(mask[:, None], turn_cols, priv[:, [2, 4, 7, 8]])
    return out


def narrow_levels_v24(env, env_ids, min_moving_time: float = 2.0):
    """v2 rule off the turn tiles. Turn tiles: up after success (turn or stand) with < 0.2 s wall touch, else down.
    (cmd_moving_time counts linear commands only, so the v2 eligibility never holds on a turn tile.)"""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    on_turn = _turn_mask(env)[ids]
    other = ids[~on_turn]
    if len(other):
        mdp_v2.narrow_levels_v2(env, other, min_moving_time)
    turning = ids[on_turn]
    if len(turning):
        monitor = mdp_v2._monitor(env)
        failed = env.termination_manager.terminated[turning]
        up = monitor.success[turning] & (monitor.touch_time[turning] < 0.2) & ~failed
        env.scene.terrain.update_env_origins(turning, up, ~up)
    return env.scene.terrain.terrain_levels.float().mean()
