"""Gait quality v3 reward terms (docs/design/GAIT_V3_DESIGN.md section 3).

Calibrated on the 3906 forward replay (11-01): with the first values (cap 2.0 m/s, impact 1.5x, floor 0.15 s, 8 cm)
3906's normal flat gait lost 0.64 /s; with these it should lose about 0.3 /s, most of it the clearance push.

Measured on 3906 / 6566 / 1944 (rl_replay, 11-01): the flat-ground reward never constrains the trunk pitch
(flat_orientation_pitch_l2 is -0.002 /s at 3 deg against +0.99 /s tracking), the body-frame foot clearance pays
+0.006..0.012 /s for a lifted nose, feet_touchdown is -0.001 /s although feet land at 0.5-1.3 m/s (p90 2 m/s at
gaps), and at gaps the trailing feet stay planted until they are 30 cm behind the hip, then fly at 3-4.5 m/s.
These terms price exactly those quantities. Rates are per second; "event" terms return value / step_dt so that
their weight is per touchdown (the RewardManager multiplies by step_dt).

Motion direction = sign of the commanded vx (|vx| < 0.1: sign of the body vx; both small: 0 = no direction).
"""
from __future__ import annotations

import math

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from gd_lab.mdp.platform_gap_terms import boarding_family_mask
from gd_lab.mdp.terrain_families import terrain_family_gate

STAIR_FAMILIES = ("pyramid_stairs", "pyramid_stairs_wide", "pyramid_stairs_nose")
FOOT_RADIUS = 0.02
CONTACT_N = 1.0


def _feet_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # [N, F, 3]


def _in_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return _feet_force(env, sensor_cfg).norm(dim=-1) > CONTACT_N


def _moving(env, command_name: str = "base_velocity", threshold: float = 0.1) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()) > threshold


def motion_direction(env, command_name: str = "base_velocity") -> torch.Tensor:
    """+1 forward, -1 backward, 0 none (per env)."""
    vx = env.command_manager.get_command(command_name)[:, 0]
    body_vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    d = torch.sign(vx) * (vx.abs() >= 0.1)
    fallback = torch.sign(body_vx) * (body_vx.abs() >= 0.1)
    return torch.where(d != 0, d, fallback)


def stairs_mask(env) -> torch.Tensor:
    return terrain_family_gate(env, STAIR_FAMILIES) == 0


def nose_up_angle(env) -> torch.Tensor:
    """Trunk front-up pitch (rad), positive = nose up; asin(-g_x) with g the projected gravity in the body frame."""
    g = env.scene["robot"].data.projected_gravity_b
    return torch.asin((-g[:, 0]).clamp(-1.0, 1.0))


def ground_pitch(env, scanner: str = "height_scanner", reach: float = 0.5, inner: float = 0.1,
                 half_width: float = 0.25) -> torch.Tensor:
    """Terrain rise along the heading (rad, positive = ground higher ahead) under the trunk: median height-scanner hit
    in front (``inner``..``reach`` m ahead, |y| < ``half_width``) minus the median behind, over their distance.
    Medians so that a gap slot under one half does not read as a slope; 0 where a half has no finite hit."""
    robot = env.scene["robot"]
    hits = env.scene.sensors[scanner].data.ray_hits_w  # [N, R, 3]
    rel = hits - robot.data.root_pos_w[:, None, :]
    q = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, hits.shape[1], -1)
    local = quat_apply_inverse(q.reshape(-1, 4), torch.nan_to_num(rel).reshape(-1, 3)).view_as(rel)
    ok = torch.isfinite(hits[..., 2]) & (local[..., 1].abs() < half_width)
    front = ok & (local[..., 0] > inner) & (local[..., 0] < reach)
    back = ok & (local[..., 0] < -inner) & (local[..., 0] > -reach)
    z = hits[..., 2]
    zf = torch.where(front, z, torch.full_like(z, float("nan"))).nanmedian(dim=1).values
    zb = torch.where(back, z, torch.full_like(z, float("nan"))).nanmedian(dim=1).values
    xf = torch.where(front, local[..., 0], torch.full_like(z, float("nan"))).nanmedian(dim=1).values
    xb = torch.where(back, local[..., 0], torch.full_like(z, float("nan"))).nanmedian(dim=1).values
    pitch = torch.atan2(zf - zb, (xf - xb).clamp_min(0.05))
    return torch.nan_to_num(pitch, nan=0.0)


class _GapSlots:
    """Lazy GapTileGeometry; None when the terrain has no gap tile metadata (then nothing is relaxed)."""

    def __init__(self, env):
        try:
            from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
            self.geometry = GapTileGeometry(env)
        except Exception:  # noqa: BLE001  (no platform_gap sub-terrain)
            self.geometry = None

    def foot_over_slot(self, env, feet_w: torch.Tensor, reach: float = 0.50) -> torch.Tensor:
        """[N] True when any foot x is within ``reach`` of a slot on a gap tile (the approach and the crossing)."""
        if self.geometry is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        slots, _ = self.geometry.for_envs(env)  # [N, 2, 2] env-frame x0, x1
        x = feet_w[..., 0] - env.scene.env_origins[:, None, 0]  # [N, F]
        inside = (x[..., None] > slots[:, None, :, 0] - reach) & (x[..., None] < slots[:, None, :, 1] + reach)
        return inside.any(-1).any(-1) & boarding_family_mask(env)


# R1 ------------------------------------------------------------------------------------------------------------
class TrunkNoseUp(ManagerTermBase):
    """Per second: nose-up above ``limit_deg`` (rad), nose-down below ``-down_deg``; every tile, both directions.
    On gap tiles, while a foot is within 0.5 m of a slot, the limits are ``gap_limit_deg`` up and ``gap_down_deg``
    down: 3906 pitches about +12 deg through the approach and -10 deg as the front lands on the far deck (replay
    11-01, 44 % of the charged gap steps were nose-down); charging that would discourage the crossing itself.
    Replaces the v2 ``backward_nose_up`` (gap tile + backward command only) and the quadratic pitch term's job.
    ``ground_relative``: the pitch is measured against the terrain rise under the trunk (``ground_pitch``), so a
    trunk parallel to a stair flight or slope is not charged. G0 (absolute pitch) charged a stair climb at 3 deg and
    lost stair ascent 94.8 -> 70.1 % (E2 11-01).
    ``level_band`` (with ``ground_relative``): any pitch between level and the terrain rise is free, the limits are
    added outside that band (up to max(0, rise) + limit, down to min(0, rise) - down). G1 (pitch minus rise, band
    +3 / -6 deg around the rise) charged a stair climb with a near-level trunk as "nose-down": 90 % of its R1 cost
    came from stair and slope tiles, the term grew to -0.146 and stair ascent stayed 2.9 points under 6566."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.slots = _GapSlots(env)

    def __call__(self, env, asset_cfg: SceneEntityCfg, limit_deg: float = 3.0, down_deg: float = 6.0,
                 gap_limit_deg: float = 12.0, gap_down_deg: float = 10.0, ground_relative: bool = False,
                 level_band: bool = False):
        nose = nose_up_angle(env)
        top = bottom = torch.zeros_like(nose)
        if ground_relative:
            rise = ground_pitch(env)
            if level_band:
                top, bottom = rise.clamp_min(0.0), rise.clamp_max(0.0)
            else:
                top = bottom = rise
        feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
        relaxed = self.slots.foot_over_slot(env, feet)
        up_limit = torch.where(relaxed, torch.full_like(nose, math.radians(gap_limit_deg)),
                               torch.full_like(nose, math.radians(limit_deg)))
        down_limit = torch.where(relaxed, torch.full_like(nose, math.radians(gap_down_deg)),
                                 torch.full_like(nose, math.radians(down_deg)))
        up = (nose - top - up_limit).clamp_min(0.0)
        down = (bottom - nose - down_limit).clamp_min(0.0)
        return up + down


# R4 ------------------------------------------------------------------------------------------------------------
def feet_touchdown_v3(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, free_speed: float = 0.3,
                      stair_free_speed: float = 0.6):
    """Event (/dt): foot speed above ``free_speed`` at first contact, summed over the feet that just landed.
    ``stair_free_speed`` on stair tiles (stepping down lands faster). No gap relaxation: a gap landing is the
    loud one (p90 2 m/s on 3906 / 6566 / 1944)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    speed = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids].norm(dim=-1)
    free = torch.where(stairs_mask(env), torch.full_like(speed[:, 0], stair_free_speed),
                       torch.full_like(speed[:, 0], free_speed))
    excess = (speed - free[:, None]).clamp_min(0.0) * first
    return excess.sum(dim=1) / env.step_dt


# R5 ------------------------------------------------------------------------------------------------------------
class FeetImpactForce(ManagerTermBase):
    """Event (/dt), paid ``window_steps`` after a touchdown: peak vertical foot force in that window over
    ``free_ratio`` x the static share (mass g / 4), in units of that share, clamped at ``cap``. The sound proxy.
    ``stair_free_ratio`` on stair tiles."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = len(cfg.params["sensor_cfg"].body_ids) if isinstance(cfg.params["sensor_cfg"].body_ids, list) else 4
        self.since = torch.full((env.num_envs, n), -1, dtype=torch.long, device=env.device)  # steps since touchdown
        self.peak = torch.zeros(env.num_envs, n, device=env.device)
        self.static = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.since[ids] = -1
        self.peak[ids] = 0.0

    def _static_force(self, env, asset_cfg):
        robot: RigidObject = env.scene[asset_cfg.name]
        masses = robot.root_physx_view.get_masses().sum(dim=1).to(env.device)  # [N] incl. payload
        return masses * 9.81 / 4.0

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, free_ratio: float = 2.0,
                 stair_free_ratio: float = 2.5, cap: float = 3.0, window_steps: int = 5):
        if self.static is None:
            self.static = self._static_force(env, asset_cfg)
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
        self.since = torch.where(first, torch.zeros_like(self.since), self.since)
        self.peak = torch.where(first, torch.zeros_like(self.peak), self.peak)
        tracking = self.since >= 0
        self.peak = torch.where(tracking, torch.maximum(self.peak, fz), self.peak)
        due = tracking & (self.since == window_steps)
        free = torch.where(stairs_mask(env), torch.full_like(self.static, stair_free_ratio),
                           torch.full_like(self.static, free_ratio))
        ratio = self.peak / self.static[:, None].clamp_min(1.0)
        excess = (ratio - free[:, None]).clamp(0.0, cap) * due
        self.since = torch.where(tracking, self.since + 1, self.since)
        self.since = torch.where(self.since > window_steps, torch.full_like(self.since, -1), self.since)
        return excess.sum(dim=1) / env.step_dt


# R6 ------------------------------------------------------------------------------------------------------------
def foot_clearance_ground(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, target: float = 0.07,
                          phase_s: tuple[float, float] = (0.08, 0.20), command_name: str = "base_velocity",
                          scanner: str = "height_scanner"):
    """Per second: shortfall of the swing foot's sole below ``target`` above the ground, measured against the
    nearest height-scanner hit (ground frame, so tilting the trunk buys nothing), during the mid-swing window
    ``phase_s`` of the current air time. Linear, moving only."""
    robot: RigidObject = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]  # [N, F, 3]
    hits = env.scene.sensors[scanner].data.ray_hits_w  # [N, R, 3]
    d2 = ((feet[:, :, None, :2] - hits[:, None, :, :2]) ** 2).sum(-1)  # [N, F, R]
    nearest = d2.argmin(dim=-1)
    ground = torch.gather(hits[..., 2], 1, nearest)  # [N, F]
    ground = torch.where(torch.isfinite(ground), ground, feet[..., 2] - FOOT_RADIUS)
    clearance = feet[..., 2] - FOOT_RADIUS - ground
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    mid = (air > phase_s[0]) & (air < phase_s[1]) & ~_in_contact(env, sensor_cfg)
    shortfall = (target - clearance).clamp_min(0.0) * mid
    return shortfall.sum(dim=1) * _moving(env, command_name).float()


# R7 ------------------------------------------------------------------------------------------------------------
def swing_time_floor(env, sensor_cfg: SceneEntityCfg, floor_s: float = 0.14, command_name: str = "base_velocity"):
    """Event (/dt): at touchdown, how far the swing that just ended fell short of ``floor_s``. Moving only."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    short = (floor_s - last_air).clamp_min(0.0) * first * (last_air > 0.0)
    return short.sum(dim=1) * _moving(env, command_name).float() / env.step_dt


# R9 ------------------------------------------------------------------------------------------------------------
def stance_overextension(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg,
                         limit: float = 0.18, gap_limit: float = 0.22, gap_scale: float = 1.5):
    """Per second: a stance foot left behind its hip (body frame, along the motion direction) by more than
    ``limit`` m, summed over feet. The "body goes, feet stay planted" state before the rushed catch-up swing.
    Gap tiles: ``gap_limit`` and ``gap_scale`` (a crossing needs a longer stride, but 30 cm is still charged)."""
    robot: RigidObject = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    hips = robot.data.body_pos_w[:, hip_cfg.body_ids]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
    dx = quat_apply_inverse(quat, feet - hips)[..., 0]  # [N, F] foot ahead of hip (+x body)
    d = motion_direction(env)
    trailing = -d[:, None] * dx  # positive = behind the hip in the motion direction
    gap = boarding_family_mask(env)
    lim = torch.where(gap, torch.full_like(d, gap_limit), torch.full_like(d, limit))
    scale = torch.where(gap, torch.full_like(d, gap_scale), torch.ones_like(d))
    excess = (trailing - lim[:, None]).clamp_min(0.0) * _in_contact(env, sensor_cfg)
    return excess.sum(dim=1) * scale * (d != 0).float()


# R10 -----------------------------------------------------------------------------------------------------------
def swing_speed_cap(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, cap: float = 3.0,
                    gap_cap: float = 3.5):
    """Per second: swing-foot speed above ``cap`` m/s (``gap_cap`` on gap tiles), summed over airborne feet."""
    speed = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids].norm(dim=-1)
    c = torch.where(boarding_family_mask(env), torch.full_like(speed[:, 0], gap_cap), torch.full_like(speed[:, 0], cap))
    excess = (speed - c[:, None]).clamp_min(0.0) * ~_in_contact(env, sensor_cfg)
    return excess.sum(dim=1)


# R11 -----------------------------------------------------------------------------------------------------------
def lead_foot_stall(env, sensor_cfg: SceneEntityCfg, stall_s: float = 0.5, cap_s: float = 1.0,
                    min_trunk_speed: float = 0.0):
    """Per second, gap tiles: the longer contact time of the two leading feet (front when forward, hind when backward)
    above ``stall_s``, capped at ``cap_s``. The "leading feet brace at the slot while the trunk goes on" state.
    ``min_trunk_speed`` > 0: only while the trunk moves in the motion direction at least that fast. G1 (no gate)
    also charged a full stop at the slot, and in MuJoCo the robot stopped at a backward slot, then pushed in and
    fell (11-01). Feet order in ``sensor_cfg``: FL, FR, RL, RR."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    stance = sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # [N, 4]
    d = motion_direction(env)
    front = stance[:, :2].max(dim=1).values
    hind = stance[:, 2:].max(dim=1).values
    lead = torch.where(d > 0, front, hind)
    over = (lead - stall_s).clamp(0.0, cap_s)
    going = d * env.scene["robot"].data.root_lin_vel_b[:, 0] >= min_trunk_speed
    return over * (d != 0).float() * going.float() * boarding_family_mask(env).float() * _moving(env).float()


# R15 -----------------------------------------------------------------------------------------------------------
class FootLiftCap(ManagerTermBase):
    """Per second: swing foot sole higher than ``cap`` above the ground (nearest height-scanner hit), summed over the
    feet, moving only. Only where the scanner sees level ground (height range under the trunk below ``flat_range``):
    off on stairs, slopes and at gap slots, where the nearest-hit ground estimate is off by the step and a lift is
    the point. G0 / G1 lifted every foot about 10 cm against R6's 7 cm floor, and the higher drop landed faster
    than 3906 (hind touchdown 0.80 vs 0.63 m/s at 0.8 m/s, replay 11-01); the first version without the gate cost
    -0.26 /s on slope tiles."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.slots = _GapSlots(env)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, cap: float = 0.09,
                 flat_range: float = 0.05, command_name: str = "base_velocity", scanner: str = "height_scanner"):
        robot: RigidObject = env.scene[asset_cfg.name]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
        hits = env.scene.sensors[scanner].data.ray_hits_w
        z = hits[..., 2]
        finite = torch.isfinite(z)
        zf = torch.where(finite, z, torch.zeros_like(z))
        span = torch.where(finite, zf, zf.amin(1, keepdim=True)).amax(1) - torch.where(finite, zf, zf.amax(1, keepdim=True)).amin(1)
        level = (span < flat_range) & finite.all(1)
        d2 = ((feet[:, :, None, :2] - hits[:, None, :, :2]) ** 2).sum(-1)
        ground = torch.gather(hits[..., 2], 1, d2.argmin(dim=-1))
        ground = torch.where(torch.isfinite(ground), ground, feet[..., 2] - FOOT_RADIUS)
        excess = (feet[..., 2] - FOOT_RADIUS - ground - cap).clamp_min(0.0) * ~_in_contact(env, sensor_cfg)
        free = level & ~stairs_mask(env) & ~self.slots.foot_over_slot(env, feet)
        return excess.sum(dim=1) * free.float() * _moving(env, command_name).float()


# R16 -----------------------------------------------------------------------------------------------------------
def backward_overspeed(env, margin: float = 0.2, command_name: str = "base_velocity"):
    """Per second, gap and stair tiles: backward trunk speed above the backward command + ``margin`` (m/s). The
    existing ``overspeed`` checks forward commands only; G1 stopped at a backward slot and then lunged in."""
    cmd = env.command_manager.get_command(command_name)[:, 0]
    speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    excess = (cmd - speed - margin).clamp_min(0.0) * (cmd < 0)
    return excess * (boarding_family_mask(env) | stairs_mask(env)).float()
