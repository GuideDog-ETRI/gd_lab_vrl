"""GAST-Narrow v2.5: cliff tiles -- an uncrossable drop on all four sides, the robot must stop before the edge.

Design: docs/design/GAST_NARROW_V25_DESIGN.md section 3. A cliff tile is a square deck (half-size L, 2.2 -> 1.6 m over
the curriculum) raised CLIFF_H above the tile floor; the robot spawns near the deck centre with a random yaw and a
constant body-frame velocity command toward one of its four sides (front 40 %, back / left / right 20 % each). Within
ZONE of the edge the linear tracking reward is off and standing pays (cliff_stand, the v2 hold bonus after HOLD_S of
filtered stillness); a foot or trunk point past the edge is a breach (-4, episode over), a drop of the trunk below the
deck is a fall (episode over). No new policy input: the drop is seen by the height scan (x +-0.8, y +-0.5 m -> 0.35 m
ahead / behind and 0.25 m beside the feet) and the belly Depth+IR; the LiDAR map does not encode drops (its height band
is 0.08..1.0 m above the feet), so ZONE is set to what the robot can see (section 3.3).

The v2 monitor sets nothing on a tile without walls (exit FAR, no rects): update_cliff writes held / breached and the
events itself, as update_turn does for turn tiles, so the v2 hold bonus, breach penalty, success termination and the
online collector see cliff episodes like blocked passages.
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg

from gd_lab.gast_narrow import geometry, mdp_v2
from gd_lab.gast_narrow.state import narrow_state

CLIFF = geometry.KINDS["cliff"]
ZONE = 0.6                  # m from the edge: standing pays, tracking off (what the height scan can see)
SPAWN_JITTER = 0.3
DIR_P = (0.4, 0.2, 0.2, 0.2)  # front, back, left, right (body frame)
SPEED_FB, SPEED_LR = (0.3, 0.6), (0.2, 0.3)
BREACH_TOL = 0.02
FALL_DROP = 0.3             # trunk below deck top by this much = fell
STILL_WZ = 0.15
TRUNK_XY = [(x, y) for x in (-0.40, 0.0, 0.40) for y in (-0.10, 0.10)]
_DIR_VEC = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


class CliffEpisode:
    """Per-env cliff draws and tracker (``env._narrow_cliff``)."""

    def __init__(self, env):
        n, d = env.num_envs, env.device
        robot = env.scene["robot"]
        self.foot_ids = torch.tensor(robot.find_bodies(".*_foot")[0], device=d, dtype=torch.long)
        self.trunk_xy = torch.tensor(TRUNK_XY, device=d)
        self.dir_vec = _DIR_VEC.to(d)
        self.cliff = torch.zeros(n, dtype=torch.bool, device=d)
        self.half = torch.zeros(n, device=d)
        self.direction = torch.zeros(n, dtype=torch.long, device=d)
        self.speed = torch.zeros(n, device=d)
        self.e = torch.full((n,), 9.0, device=d)      # edge margin of the closest point (m, < 0 = over the edge)
        self.e_min = torch.full((n,), 9.0, device=d)
        self.fell = torch.zeros(n, dtype=torch.bool, device=d)
        self.hold_time = torch.zeros(n, device=d)
        self.hold_max = torch.zeros(n, device=d)
        self.wz_f = torch.zeros(n, device=d)
        self.v_f = torch.zeros(n, 2, device=d)


def cliff_episode(env) -> CliffEpisode:
    episode = getattr(env, "_narrow_cliff", None)
    if episode is None:
        episode = CliffEpisode(env)
        env._narrow_cliff = episode
    return episode


def _cliff_mask(env):
    return cliff_episode(env).cliff & (narrow_state(env).kind == CLIFF)


def reset_cliff(env, env_ids):
    """After reset_narrow_v2 / reset_turn: cliff envs re-spawn on the deck; direction and speed draw; v2 blocked off."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    state = narrow_state(env)
    cliff = cliff_episode(env)
    v2 = mdp_v2.narrow_episode(env)
    is_cliff = state.kind[ids] == CLIFF
    cliff.cliff[ids] = is_cliff
    ids = ids[is_cliff]
    if not len(ids):
        return
    n, d = len(ids), env.device
    rand = lambda: torch.rand(n, device=d)  # noqa: E731
    v2.blocked[ids] = False  # the v2 zone / stall logic must not see a blocked passage here
    v2.slack[ids] = 1.0
    cliff.half[ids] = state.width[ids]
    p = torch.tensor(DIR_P, device=d)
    cliff.direction[ids] = torch.multinomial(p.expand(n, -1), 1).squeeze(-1)
    fb = cliff.direction[ids] < 2
    lo = torch.where(fb, torch.full((n,), SPEED_FB[0], device=d), torch.full((n,), SPEED_LR[0], device=d))
    hi = torch.where(fb, torch.full((n,), SPEED_FB[1], device=d), torch.full((n,), SPEED_LR[1], device=d))
    cliff.speed[ids] = lo + (hi - lo) * rand()
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[ids, :7].clone()
    origin = env.scene.env_origins[ids]
    pose[:, :2] = origin[:, :2] + (torch.rand(n, 2, device=d) * 2 - 1) * SPAWN_JITTER
    pose[:, 2] = origin[:, 2] + pose[:, 2]
    yaw = rand() * 2 * math.pi
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 4:6] = 0.0
    pose[:, 6] = torch.sin(yaw / 2)
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=d), env_ids=ids)
    state.reset_counter += 1
    cliff.e[ids] = 9.0
    cliff.e_min[ids] = 9.0
    cliff.fell[ids] = False
    cliff.hold_time[ids] = 0.0
    cliff.hold_max[ids] = 0.0
    cliff.wz_f[ids] = 0.0
    cliff.v_f[ids] = 0.0


class NarrowV25CliffCommandMixin:
    """Mixin over the v2.4 command: cliff tiles get a constant body-frame velocity toward one side, wz = 0."""

    def _apply_cliff(self):
        env = self._env
        mask = _cliff_mask(env)
        if not bool(mask.any()):
            return
        cliff = cliff_episode(env)
        v = cliff.dir_vec[cliff.direction] * cliff.speed[:, None]
        self.vel_command_b[mask, 0] = v[mask, 0]
        self.vel_command_b[mask, 1] = v[mask, 1]
        self.vel_command_b[mask, 2] = 0.0
        self.is_heading_env[mask] = False
        self.is_standing_env[mask] = False


def _edge_margin(env, cliff, state):
    """[N] deck half-size minus the largest |x| / |y| over the trunk points and the feet (tile frame)."""
    robot = env.scene["robot"].data
    origin = env.scene.env_origins[:, :2]
    c, s = torch.cos(state.yaw), torch.sin(state.yaw)
    tx, ty = cliff.trunk_xy[:, 0], cliff.trunk_xy[:, 1]
    trunk = torch.stack((robot.root_pos_w[:, 0:1] + c[:, None] * tx - s[:, None] * ty,
                         robot.root_pos_w[:, 1:2] + s[:, None] * tx + c[:, None] * ty), -1)  # [N,6,2]
    feet = robot.body_pos_w[:, cliff.foot_ids, :2]                                           # [N,4,2]
    pts = torch.cat((trunk, feet), 1) - origin[:, None]
    return cliff.half - pts.abs().amax(-1).amax(-1)


def update_cliff(env, monitor):
    """Called by the v2.5 monitor after the v2 / turn updates: edge margin, breach, fall, filtered hold."""
    state = narrow_state(env)
    cliff = cliff_episode(env)
    mask = cliff.cliff & (state.kind == CLIFF)
    if not bool(mask.any()):
        return
    dt = env.step_dt
    robot = env.scene["robot"].data
    e = _edge_margin(env, cliff, state)
    cliff.e = torch.where(mask, e, cliff.e)
    cliff.e_min = torch.where(mask, torch.minimum(cliff.e_min, e), cliff.e_min)
    cliff.fell |= mask & (robot.root_pos_w[:, 2] < env.scene.env_origins[:, 2] - FALL_DROP)
    new_breach = mask & (e < -BREACH_TOL) & ~monitor.breached & ~monitor.held
    monitor.breached |= new_breach
    monitor.breach_event |= new_breach
    alpha = min(1.0, dt / mdp_v2.STILL_TAU)
    cliff.wz_f = cliff.wz_f + alpha * (robot.root_ang_vel_b[:, 2] - cliff.wz_f)
    cliff.v_f = cliff.v_f + alpha * (robot.root_lin_vel_b[:, :2] - cliff.v_f)
    zone = mask & (e < ZONE) & ~monitor.breached & ~cliff.fell
    still = zone & (cliff.wz_f.abs() < STILL_WZ) & (cliff.v_f.norm(dim=-1) < mdp_v2.STILL_SPEED)
    cliff.hold_time = torch.where(still, cliff.hold_time + dt, torch.zeros_like(cliff.hold_time))
    cliff.hold_max = torch.maximum(cliff.hold_max, cliff.hold_time)
    new_hold = still & (cliff.hold_time >= mdp_v2.HOLD_S) & ~monitor.held
    monitor.held |= new_hold
    monitor.hold_event |= new_hold
    monitor.success = monitor.passed | monitor.held


def cliff_zone(env):
    """[N] bool: on a cliff tile within ZONE of the edge, not breached / fallen (standing is the task here)."""
    cliff = cliff_episode(env)
    monitor = mdp_v2._monitor(env)
    return _cliff_mask(env) & (cliff.e < ZONE) & ~monitor.breached & ~cliff.fell


def cliff_stand(env, sigma_wz: float = 0.2, sigma_v: float = 0.10):
    """Per second in the cliff zone: exp(-(wz/0.2)^2 - (v/0.1)^2) (turn_stand's kernel)."""
    robot = env.scene["robot"].data
    wz = robot.root_ang_vel_b[:, 2]
    speed = robot.root_lin_vel_b[:, :2].norm(dim=-1)
    return torch.exp(-(wz / sigma_wz) ** 2 - (speed / sigma_v) ** 2) * cliff_zone(env).float()


def cliff_fall_done(env):
    return cliff_episode(env).fell & _cliff_mask(env)


def base_height_cliff(env, target_height, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    """platform_gap_base_height, except that on cliff tiles the terrain reference ignores rays over the void
    (ray heights clamped to the deck top - FALL_DROP): with 20-50 % of the rays over the drop the plain mean lowers the
    reference by 0.2-0.5 m, a -0.6..-3.8 /s penalty for standing at the edge (larger than cliff_stand)."""
    from gd_lab.mdp.platform_gap_terms import platform_gap_base_height
    usual = platform_gap_base_height(env, target_height, sensor_cfg, asset_cfg)
    mask = _cliff_mask(env)
    if not bool(mask.any()):
        return usual
    sensor = env.scene[sensor_cfg.name]
    deck = env.scene.env_origins[:, 2:3]
    ray_z = torch.clamp(sensor.data.ray_hits_w[..., 2], min=-10.0, max=10.0)
    ray_z = torch.maximum(ray_z, deck - FALL_DROP)
    root_z = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    on_deck = torch.square(root_z - (target_height + ray_z.mean(dim=1)))
    return torch.where(mask, on_deck, usual)


def narrow_privileged_v25(env):
    """v2.4's critic inputs; on cliff tiles slot 7 = 1 (must stand) and slot 8 = edge margin / 2."""
    from gd_lab.gast_narrow.mdp_turn import narrow_privileged_v24
    priv = narrow_privileged_v24(env)
    mask = _cliff_mask(env)
    if not bool(mask.any()):
        return priv
    cliff = cliff_episode(env)
    out = priv.clone()
    out[:, 7] = torch.where(mask, torch.ones_like(priv[:, 7]), priv[:, 7])
    out[:, 8] = torch.where(mask, (cliff.e / 2.0).clamp(-1.0, 1.0), priv[:, 8])
    return out


def cliff_levels(env, ids):
    """Cliff tiles: up after a hold, down on breach / fall / time-out without a hold (no moving-time eligibility: a
    hold can end the episode at 2.0 s, the v2 rule's threshold)."""
    if not len(ids):
        return
    monitor = mdp_v2._monitor(env)
    cliff = cliff_episode(env)
    up = monitor.held[ids] & ~monitor.breached[ids] & ~cliff.fell[ids]
    env.scene.terrain.update_env_origins(ids, up, ~up)
