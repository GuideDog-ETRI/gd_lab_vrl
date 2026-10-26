"""Head LiDAR obstacle map: Ouster OS1-64 + Livox Mid-360 raycast against the terrain mesh, 10 Hz.

Mounts follow sim_docker ``_sensor_configuration.yaml`` (2026-09-30 calibration, parent base_link = trunk):
- os_sensor at (0.285, 0, 0.046), rpy (0, +0.74, +0.50) deg; rays leave os_lidar = os_sensor + 0.03618 m z.
  OS1-64: 64 beams over +/-22.5 deg, here 256 columns (1024 on the sensor), min range 0.5 m (sim_docker cut).
- livox_frame = os_sensor * ((-0.0207, 0.0387, 0.1426), rpy (+0.02, +4.81, 0) deg). Mid-360: -7..52 deg,
  non-repetitive; here 24 x 160 rays with a fresh random azimuth/elevation jitter per capture, min range 0.1 m.
Per mount: +/-1 cm, +/-1 deg random offset per episode. Range noise 1 cm, 5 % point dropout, 3 % frame drop
(previous frame held). Max range 3 m.

Map (policy input, group ``narrow_lidar``): body-yaw frame, x in [-0.9, 1.5), y in [-0.75, 0.75), 5 cm
-> 30 (y) x 48 (x) cells; 1 = a return with 0.08..1.0 m height above the feet-estimated ground. Two frames:
[previous capture, current capture], flattened [2, 30, 48]. Rows run from -y to +y (mirror = flip rows).
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase

X_MIN, X_MAX, Y_MIN, Y_MAX, CELL = -0.9, 1.5, -0.75, 0.75, 0.05
NX, NY = round((X_MAX - X_MIN) / CELL), round((Y_MAX - Y_MIN) / CELL)
FRAMES = 2
BAND = (0.08, 1.00)
FOOT_RADIUS = 0.03
NOMINAL_HEIGHT = 0.55
PERIOD_S = 0.1
MAX_RANGE = 3.0
OUSTER_XYZ, OUSTER_RPY, OUSTER_LIDAR_Z = (0.285, 0.0, 0.046), (0.0, 0.74, 0.50), 0.03618
LIVOX_REL_XYZ, LIVOX_REL_RPY = (-0.0207, 0.0387, 0.1426), (0.02, 4.81, 0.0)
OUSTER_MIN, LIVOX_MIN = 0.5, 0.1


def _rot(rpy_deg):
    r, p, y = (math.radians(a) for a in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])


def _dirs(elev_deg, azim_deg):
    e, a = torch.deg2rad(elev_deg), torch.deg2rad(azim_deg)
    return torch.stack((e.cos() * a.cos(), e.cos() * a.sin(), e.sin()), -1)


def mounts():
    """(position [2,3], rotation [2,3,3]) of the Ouster and Mid-360 ray origins in the trunk frame."""
    r_os = _rot(OUSTER_RPY)
    p_os = torch.tensor(OUSTER_XYZ)
    p_ouster = p_os + r_os @ torch.tensor((0.0, 0.0, OUSTER_LIDAR_Z))
    r_livox = r_os @ _rot(LIVOX_REL_RPY)
    p_livox = p_os + r_os @ torch.tensor(LIVOX_REL_XYZ)
    return torch.stack((p_ouster, p_livox)), torch.stack((r_os, r_livox))


def ouster_pattern(cols=256):
    el = torch.linspace(-22.5, 22.5, 64)
    az = torch.arange(cols) * (360.0 / cols)
    E, A = torch.meshgrid(el, az, indexing="ij")
    return _dirs(E.flatten(), A.flatten())


def livox_grid(rows=24, cols=160):
    el = torch.linspace(-7.0, 52.0, rows)
    az = torch.arange(cols) * (360.0 / cols)
    E, A = torch.meshgrid(el, az, indexing="ij")
    return E.flatten(), A.flatten()


def _quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


class NarrowLidarMap(ManagerTermBase):
    """Observation term: [N, FRAMES*NY*NX] occupancy, refreshed every PERIOD_S per env (random phase)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from gd_lab.teachers.bivt.raycast_visibility import warp_mesh_cast
        scanner = env.scene["height_scanner"]
        self.cast = warp_mesh_cast(scanner.meshes[scanner.cfg.mesh_prim_paths[0]])
        d = env.device
        self.period = max(1, round(PERIOD_S / env.step_dt))
        p, r = mounts()
        self.mount_p, self.mount_r = p.to(d), r.to(d)
        self.ouster = ouster_pattern().to(d)
        el, az = livox_grid()
        self.livox_el, self.livox_az = el.to(d), az.to(d)
        n = env.num_envs
        self.frames = torch.zeros(n, FRAMES, NY * NX, device=d)
        self.phase = torch.randint(0, self.period, (n,), device=d)
        self.mount_dp = torch.zeros(n, 2, 3, device=d)
        self.mount_dr = torch.eye(3, device=d).repeat(n, 2, 1, 1)
        self.chunk = 256
        self.counts = torch.zeros(3, device=d, dtype=torch.float64)  # [captures, occupied cells, cells]
        self.foot_ids = env.scene["robot"].find_bodies(".*_foot")[0]
        self.pending = torch.ones(n, dtype=torch.bool, device=d)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        self.frames[ids] = 0.0
        self.phase[ids] = torch.randint(0, self.period, (len(ids),), device=self.device)
        self.mount_dp[ids] = (torch.rand(len(ids), 2, 3, device=self.device) * 2 - 1) * 0.01
        angles = (torch.rand(len(ids), 2, 3, device=self.device) * 2 - 1) * math.radians(1.0)
        self.mount_dr[ids] = _small_rotation(angles)
        # Captured at the first observation after the reset (link poses are only valid after the next read).
        self.pending[ids] = True

    def _rays(self, n):
        """Unit directions in each sensor frame: [n, R, 3] and the sensor index of each ray [R]."""
        jit_az = torch.rand(n, 1, device=self.device) * (360.0 / 160)
        jit_el = (torch.rand(n, self.livox_el.numel(), device=self.device) - 0.5) * (59.0 / 23)
        livox = _dirs((self.livox_el + jit_el).clamp(-7.0, 52.0), self.livox_az + jit_az)
        ouster = self.ouster.expand(n, -1, -1)
        return ouster, livox

    def _capture(self, ids, fill_both=False):
        """fill_both: first capture of an episode -- both frames, ground from the trunk height (feet may be stale)."""
        env = self._env
        robot = env.scene["robot"].data
        for start in range(0, len(ids), self.chunk):
            sub = ids[start:start + self.chunk]
            n = len(sub)
            pos, quat = robot.root_pos_w[sub], robot.root_quat_w[sub]
            R_body = _quat_to_matrix(quat)                                         # [n,3,3]
            ouster, livox = self._rays(n)
            occ = torch.zeros(n, NY * NX, device=self.device)
            if fill_both:
                ground = pos[:, 2] - NOMINAL_HEIGHT
            else:
                ground = robot.body_pos_w[sub][:, self.foot_ids, 2].min(-1).values - FOOT_RADIUS
            yaw = robot.heading_w[sub]
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            for k, (dirs, rmin) in enumerate(((ouster, OUSTER_MIN), (livox, LIVOX_MIN))):
                R_mount = self.mount_r[k] @ self.mount_dr[sub, k]                  # [n,3,3]
                R = R_body @ R_mount
                origin = pos + (R_body @ (self.mount_p[k] + self.mount_dp[sub, k])[..., None])[..., 0]
                d_w = dirs @ R.transpose(1, 2)                                     # [n,R,3]
                starts = origin[:, None].expand_as(d_w).reshape(-1, 3).contiguous()
                dist = self.cast(starts, d_w.reshape(-1, 3).contiguous(), MAX_RANGE).reshape(n, -1)
                dist = dist + torch.randn_like(dist) * 0.01
                ok = torch.isfinite(dist) & (dist >= rmin) & (dist < MAX_RANGE)
                ok &= torch.rand_like(dist) >= 0.05
                hit = origin[:, None] + d_w * torch.where(ok, dist, torch.zeros_like(dist))[..., None]
                rel = hit[..., :2] - pos[:, None, :2]
                bx = cy[:, None] * rel[..., 0] + sy[:, None] * rel[..., 1]
                by = -sy[:, None] * rel[..., 0] + cy[:, None] * rel[..., 1]
                h = hit[..., 2] - ground[:, None]
                ix = ((bx - X_MIN) / CELL).floor().long()
                iy = ((by - Y_MIN) / CELL).floor().long()
                keep = ok & (h >= BAND[0]) & (h <= BAND[1]) & (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
                flat = torch.where(keep, iy * NX + ix, torch.zeros_like(ix))
                occ.scatter_reduce_(1, flat, keep.float(), reduce="amax")
            drop = torch.rand(n, device=self.device) < 0.03
            if fill_both:
                self.frames[sub, 0] = occ
                self.frames[sub, 1] = occ
            else:
                new = torch.where(drop[:, None], self.frames[sub, 1], occ)
                self.frames[sub, 0] = self.frames[sub, 1]
                self.frames[sub, 1] = new
            self.counts += torch.tensor([n, float(occ.sum()), float(occ.numel())], device=self.device,
                                        dtype=torch.float64)

    def __call__(self, env):
        pending = self.pending.nonzero().flatten()
        if len(pending):
            self._capture(pending, fill_both=True)
            self.pending[pending] = False
        due = (((env.episode_length_buf + self.phase) % self.period == 0) & ~self.pending).nonzero().flatten()
        due = due[~torch.isin(due, pending)]
        if len(due):
            self._capture(due)
        return self.frames.flatten(1)


def _small_rotation(angles):
    """angles [...,3] (roll, pitch, yaw) rad -> rotation matrices [...,3,3]."""
    r, p, y = angles.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr), -1).reshape(*angles.shape[:-1], 3, 3)
