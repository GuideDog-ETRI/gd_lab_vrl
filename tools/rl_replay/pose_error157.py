"""UPDATE157 evaluation-only planar odometry. No physics/proprio or global RNG changes.

World-frame xy random walk + distance-proportional xy slip; yaw RW in radians.
At each episode-local 50 Hz sample: odom = delayed GT + current drift + white.
Yaw slip is not added (k_slip is dimensionless xy distance gain). z/roll/pitch
remain GT. Startup delay is clamped to the episode's initial pose, never a prior
episode. Noise streams are keyed by seed/env/episode/sample, independent of level.
"""
from dataclasses import asdict, dataclass, replace
import math
import os

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    level: str
    rw_xy: float
    rw_yaw_deg: float
    k_slip: float
    white_xy: float
    white_yaw_deg: float
    delay_ms: float


LEVELS = {
    'P0': Config('P0', 0., 0., 0., 0., 0., 0.),
    'P1': Config('P1', .005, .1, .01, .003, .1, 20.),
    'P2': Config('P2', .015, .3, .03, .005, .2, 40.),
    'P3': Config('P3', .04, .8, .06, .01, .5, 80.),
}


def from_environ(environ=None):
    e = os.environ if environ is None else environ
    level = e.get('U157_POSE_ERR', '')
    if not level:
        return None
    if level not in LEVELS:
        raise ValueError('U157_POSE_ERR must be P0/P1/P2/P3 or unset')
    cfg = LEVELS[level]
    overrides = {}
    for field in asdict(cfg):
        if field == 'level':
            continue
        key = 'U157_' + field.upper()
        if key in e:
            value = float(e[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
            overrides[field] = value
    return replace(cfg, **overrides)


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def _normal(seed, env, episode, sample):
    """Counter-based tensor RNG, eight normals; never consumes torch global RNG."""
    lane = torch.arange(8, device=env.device, dtype=torch.int64)[None]
    x = (int(seed) + 157 * 1000003 + env[:, None] * 1000033
         + episode[:, None] * 1000037 + sample[:, None] * 1000039
         + lane * 1000081) & 0xffffffff
    def uniform(y):
        y = ((y ^ (y >> 16)) * 1597334677) & 0xffffffff
        y = ((y ^ (y >> 15)) * 1597334677) & 0xffffffff
        y = y ^ (y >> 16)
        return (y.to(torch.float64) + .5) / 4294967296.
    u = uniform(x)
    v = uniform((x + 2246822519) & 0xffffffff)
    return torch.sqrt(-2 * torch.log(u)) * torch.cos(2 * math.pi * v)


class PoseError:
    def __init__(self, cfg, n, dt, seed, device):
        self.cfg, self.n, self.dt, self.seed = cfg, n, float(dt), int(seed)
        self.device = torch.device(device)
        self.period = round(.02 / self.dt)
        self.delay = round(cfg.delay_ms / 1000 / self.dt)
        if self.period < 1 or abs(self.period * self.dt - .02) > 1e-8:
            raise ValueError('50 Hz odom requires an integer number of policy steps')
        if abs(self.delay * self.dt * 1000 - cfg.delay_ms) > 1e-6:
            raise ValueError('pose delay must be an integer number of policy steps')
        self.length = self.delay + 1
        self.env_ids = torch.arange(n, device=self.device)
        self.episode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.ticks = torch.zeros_like(self.episode)
        self.samples = torch.zeros_like(self.episode)
        self.ring = torch.zeros((self.length, n, 3), device=self.device)
        self.drift = torch.zeros((n, 3), device=self.device, dtype=torch.float64)
        self.pose = torch.zeros((n, 3), device=self.device)
        self.fresh = torch.ones(n, dtype=torch.bool, device=self.device)
        self.records = []

    def reset(self, ids):
        self.episode[ids] += 1
        self.ticks[ids] = 0
        self.samples[ids] = 0
        self.drift[ids] = 0
        self.fresh[ids] = True

    def update(self, gt, speed, yaw_rate=None):
        gt = gt.detach()
        new = self.fresh.clone()
        self.ring[:, new] = gt[new]
        self.ring[self.ticks % self.length, self.env_ids] = gt
        emit = self.ticks % self.period == 0
        ids = self.env_ids[emit]
        if len(ids):
            noise = _normal(self.seed, ids, self.episode[ids], self.samples[ids])
            active = (~new[ids]).to(torch.float64)
            angular = torch.zeros_like(speed) if yaw_rate is None else yaw_rate
            moving = ((speed[ids] > .05) | (angular[ids].abs() > .05)).to(torch.float64)
            rw = noise[:, :3] * torch.tensor(
                [self.cfg.rw_xy, self.cfg.rw_xy, math.radians(self.cfg.rw_yaw_deg)],
                device=self.device) * math.sqrt(.02) * moving[:, None]
            rw[:, :2] += noise[:, 3:5] * self.cfg.k_slip * speed[ids, None] * .02
            self.drift[ids] += rw * active[:, None]
            white = noise[:, 5:8] * torch.tensor(
                [self.cfg.white_xy, self.cfg.white_xy, math.radians(self.cfg.white_yaw_deg)],
                device=self.device)
            delayed = self.ring[(self.ticks[ids] - self.delay) % self.length, ids]
            value = delayed.to(torch.float64) + self.drift[ids] + white
            value[:, 2] = wrap(value[:, 2])
            self.pose[ids] = value.to(gt.dtype)
            self.samples[ids] += 1
        error = self.pose - gt
        error[:, 2] = wrap(error[:, 2])
        self.records.append((self.episode.cpu().numpy().copy(), error.cpu().numpy().copy()))
        self.fresh[:] = False
        self.ticks += 1
        return self.pose.clone()

    def summary(self):
        episodes = np.stack([r[0] for r in self.records])
        errors = np.stack([r[1] for r in self.records])
        lag = round(4.8 / self.dt)
        def stats(values):
            xy = np.linalg.norm(values[:, :2], axis=-1)
            yaw = np.abs(np.rad2deg(np.arctan2(np.sin(values[:, 2]), np.cos(values[:, 2]))))
            return dict(samples=len(values), xy_m_max=float(xy.max()), xy_m_p95=float(np.quantile(xy, .95)),
                        yaw_deg_max=float(yaw.max()), yaw_deg_p95=float(np.quantile(yaw, .95)))
        rows = []
        for e in range(self.n):
            for episode in np.unique(episodes[:, e]):
                select = episodes[:, e] == episode
                row = dict(env=e, episode=int(episode), **stats(errors[select, e]))
                indices = np.flatnonzero(select)
                indices = indices[(indices >= lag)]
                indices = indices[episodes[indices-lag, e] == episode]
                row['relative_4p8s'] = stats(errors[indices, e]-errors[indices-lag, e]) if len(indices) else None
                rows.append(row)
        relative = errors[lag:] - errors[:-lag] if len(errors) > lag else np.empty((0, self.n, 3))
        mask = episodes[lag:] == episodes[:-lag] if len(errors) > lag else np.empty((0, self.n), bool)
        return dict(level=self.cfg.level, parameters=asdict(self.cfg), odom_hz=50, delay_steps=self.delay,
                    seed=self.seed, rng='u157_counter_box_muller_v1(seed,env,episode,sample)',
                    z_roll_pitch='GT unchanged', slip='xy only, k_slip * GT planar speed * 0.02s per sample',
                    movement_gate='RW only when GT speed>0.05m/s OR abs(GT yaw-rate)>0.05rad/s; white always',
                    startup='zero accumulated drift; white always; GT delay clamped to same-episode start',
                    summary_rate_hz=1/self.dt, actual_error_includes='RW,slip,white,GT delay,50Hz hold',
                    relative_definition='world displacement error over 4.8s, same episode only; wrapped yaw difference',
                    overall=stats(errors.reshape(-1, 3)),
                    relative_4p8s=stats(relative[mask]) if mask.any() else None, episodes=rows)


def transform_packet(packet, odom_pose):
    """Keep real raycast returns/self-hits; express their capture geometry in odom.

    Full GT roll/pitch and every z are preserved. Only planar translation/yaw
    is changed, rigidly about the capture body's GT origin.
    """
    out = dict(packet)
    gt_pos, gt_q = packet['body_pos'], packet['body_quat']
    w, x, y, z = np.moveaxis(gt_q, -1, 0)
    gt_yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    delta = np.arctan2(np.sin(odom_pose[:, 2]-gt_yaw), np.cos(odom_pose[:, 2]-gt_yaw))
    c, s = np.cos(delta), np.sin(delta)
    def rotate(v):
        shape = (len(v),) + (1,) * (v.ndim-2)
        cc, ss = c.reshape(shape), s.reshape(shape)
        r = v.copy()
        r[..., 0] = cc*v[..., 0]-ss*v[..., 1]
        r[..., 1] = ss*v[..., 0]+cc*v[..., 1]
        return r
    origin = rotate(packet['origins']-gt_pos[:, None, :]) + gt_pos[:, None, :]
    origin[..., :2] += (odom_pose[:, :2]-gt_pos[:, :2])[:, None]
    out['origins'] = origin
    out['dirs'] = rotate(packet['dirs'])
    pos = gt_pos.copy();pos[:, :2] = odom_pose[:, :2];out['body_pos'] = pos
    ca, sa = np.cos(delta/2), np.sin(delta/2)
    out['body_quat'] = np.stack((ca*w-sa*z, ca*x-sa*y, ca*y+sa*x, ca*z+sa*w), -1)
    return out
