"""Per-step narrow-passage geometry for every env: wall distances of the robot, path frame, exit/success.

Tables come from ``terrain_generator.narrow_tile_metadata`` (exact, recorded at generation). Everything is computed
once per env step and cached on the env (``env._narrow_state``); the monitor reward term owns the episode tracker.
"""
from __future__ import annotations

import math

import torch

from gd_lab.gast_narrow import geometry

# Trunk collision box 0.8 x 0.20 x 0.12 (rbq10_simple.urdf): edge points in the trunk frame, radius 0.
TRUNK_POINTS = [(x, y, 0.0) for x in (-0.40, 0.0, 0.40) for y in (-0.10, 0.10)]
BODY_RADII = {"_hip": 0.05, "_calf": 0.04, "_foot": 0.03}
SHIN_RADIUS = 0.035
SUCCESS_HOLD_S = 0.5


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _quat_apply(q, v):
    """q [N,4] wxyz, v [N,M,3] -> rotated [N,M,3]."""
    w, xyz = q[:, None, :1], q[:, None, 1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def rect_distance(points, rects):
    """points [N,M,2], rects [N,K,4] (x0,x1,y0,y1) -> [N,M] distance to the nearest rect (0 inside)."""
    px, py = points[..., None, 0], points[..., None, 1]
    x0, x1, y0, y1 = (rects[:, None, :, i] for i in range(4))
    dx = torch.clamp(torch.maximum(x0 - px, px - x1), min=0.0)
    dy = torch.clamp(torch.maximum(y0 - py, py - y1), min=0.0)
    return torch.sqrt(dx * dx + dy * dy).amin(-1)


def project_path(p, path):
    """p [N,2], path [N,P,2] -> (lateral [N] (+left), arclength s [N], tangent yaw [N], total length [N])."""
    a, b = path[:, :-1], path[:, 1:]
    seg = b - a
    length = seg.norm(dim=-1)
    safe = length.clamp_min(1e-6)
    t = (((p[:, None] - a) * seg).sum(-1) / (safe * safe)).clamp(0.0, 1.0)
    closest = a + t[..., None] * seg
    dist = (p[:, None] - closest).norm(dim=-1)
    dist = torch.where(length > 1e-6, dist, torch.full_like(dist, 1e6))
    k = dist.argmin(-1)
    rows = torch.arange(p.shape[0], device=p.device)
    seg_k, a_k, t_k = seg[rows, k], a[rows, k], t[rows, k]
    cross = seg_k[:, 0] * (p[:, 1] - a_k[:, 1]) - seg_k[:, 1] * (p[:, 0] - a_k[:, 0])
    lateral = cross / safe[rows, k]
    before = torch.cumsum(torch.cat((torch.zeros_like(length[:, :1]), length[:, :-1]), -1), -1)
    s = before[rows, k] + t_k * length[rows, k]
    yaw = torch.atan2(seg_k[:, 1], seg_k[:, 0])
    return lateral, s, yaw, length.sum(-1)


def tangent_at(path, s):
    """Tangent yaw of the path at arclength s (clamped to the path)."""
    seg = path[:, 1:] - path[:, :-1]
    length = seg.norm(dim=-1)
    end = torch.cumsum(length, -1)
    k = (end < s[:, None]).sum(-1).clamp(max=seg.shape[1] - 1)
    rows = torch.arange(path.shape[0], device=path.device)
    # Skip zero-length padding segments: use the last real one.
    real = (length > 1e-6).float() * torch.arange(1, seg.shape[1] + 1, device=path.device)
    last_real = real.argmax(-1)
    k = torch.minimum(k, last_real)
    seg_k = seg[rows, k]
    return torch.atan2(seg_k[:, 1], seg_k[:, 0])


class NarrowTables:
    def __init__(self, env):
        gen = env.scene.terrain.cfg.terrain_generator
        meta = getattr(gen, "narrow_tile_metadata", None)
        if meta is None:
            raise ValueError("narrow task needs gd_lab.gast_narrow.terrain.NarrowTerrainGenerator")
        rows, cols = gen.num_rows, gen.num_cols
        rects = torch.zeros(rows, cols, geometry.MAX_RECTS, 4)
        path = torch.zeros(rows, cols, geometry.MAX_PATH, 2)
        exit_ = torch.zeros(rows, cols, 3)
        kind = torch.zeros(rows, cols, dtype=torch.long)
        width = torch.zeros(rows, cols)
        block_s = torch.zeros(rows, cols)
        for r in range(rows):
            for c in range(cols):
                g = meta[r][c]
                rr, pp, ee, kk = geometry.padded(g)
                rects[r, c] = torch.tensor(rr)
                path[r, c] = torch.tensor(pp)
                exit_[r, c] = torch.tensor(ee)
                kind[r, c] = kk
                width[r, c] = g["width"] if g else 0.0
                block_s[r, c] = g.get("block_s", 0.0) if g else 0.0
        d = env.device
        self.rects, self.path, self.exit, self.kind, self.width = (x.to(d) for x in (rects, path, exit_, kind, width))
        self.block_s = block_s.to(d)
        self.narrow_columns = sorted({c for c in range(cols) if any(meta[r][c] for r in range(rows))})


class NarrowState:
    """Computed lazily once per env step (keyed on ``common_step_counter`` + a reset counter)."""

    def __init__(self, env):
        self.env = env
        self.tables = NarrowTables(env)
        robot = env.scene["robot"]
        ids, names, radii, left = [], [], [], []
        for suffix, radius in BODY_RADII.items():
            body_ids, body_names = robot.find_bodies(f".*{suffix}")
            ids += list(body_ids)
            names += list(body_names)
            radii += [radius] * len(body_ids)
            left += [n[1] == "L" for n in body_names]
        # Optional shin points (v2.3: env.cfg.narrow_shin_fracs): calf origin (knee) -> foot, radius SHIN_RADIUS.
        self.shin_fracs = tuple(getattr(env.cfg, "narrow_shin_fracs", ()))
        self.shin_calf, self.shin_foot = [], []
        if self.shin_fracs:
            calf_ids, calf_names = robot.find_bodies(".*_calf")
            foot_ids, foot_names = robot.find_bodies(".*_foot")
            foot_by_leg = {n.split("_")[0]: i for i, n in zip(foot_ids, foot_names)}
            for i, n in zip(calf_ids, calf_names):
                leg = n.split("_")[0]
                self.shin_calf.append(i)
                self.shin_foot.append(foot_by_leg[leg])
                for _ in self.shin_fracs:
                    radii.append(SHIN_RADIUS)
                    left.append(n[1] == "L")
        self.body_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        trunk = torch.tensor(TRUNK_POINTS, device=env.device)
        self.trunk_points = trunk
        self.radii = torch.tensor([0.0] * len(TRUNK_POINTS) + radii, device=env.device)
        self.left = torch.tensor([p[1] > 0 for p in TRUNK_POINTS] + left, device=env.device)
        self._key = None
        self.reset_counter = 0

    def update(self):
        env = self.env
        key = (int(env.common_step_counter), self.reset_counter)
        if key == self._key:
            return self
        self._key = key
        terrain = env.scene.terrain
        r, c = terrain.terrain_levels.long(), terrain.terrain_types.long()
        t = self.tables
        self.kind = t.kind[r, c]
        self.narrow = self.kind > 0
        self.width = t.width[r, c]
        self.block_s = t.block_s[r, c]
        rects, path, exit_ = t.rects[r, c], t.path[r, c], t.exit[r, c]
        self.path_e = path
        robot = env.scene["robot"].data
        origin = env.scene.env_origins
        root = robot.root_pos_w
        quat = robot.root_quat_w
        self.local = root[:, :2] - origin[:, :2]
        self.yaw = robot.heading_w
        trunk_w = root[:, None] + _quat_apply(quat, self.trunk_points.expand(root.shape[0], -1, -1))
        bodies = robot.body_pos_w[:, self.body_ids]
        if self.shin_fracs:
            knee = robot.body_pos_w[:, self.shin_calf]
            foot = robot.body_pos_w[:, self.shin_foot]
            shin = [knee + f * (foot - knee) for f in self.shin_fracs]  # each [N, 4, 3]
            bodies = torch.cat((bodies, torch.stack(shin, 2).flatten(1, 2)), 1)
        pts = torch.cat((trunk_w, bodies), 1)[..., :2] - origin[:, None, :2]
        self.clearance = rect_distance(pts, rects) - self.radii  # [N, M]
        big = torch.full_like(self.clearance, 9.0)
        self.d_left = torch.where(self.left, self.clearance, big).amin(-1)
        self.d_right = torch.where(~self.left, self.clearance, big).amin(-1)
        self.lateral, self.s, self.tangent, self.total = project_path(self.local, path)
        self.heading_error = _wrap(self.yaw - self.tangent)
        axis = exit_[:, 0].long()
        coord = torch.where(axis == 0, self.local[:, 0], self.local[:, 1])
        self.past_exit = self.narrow & (coord * exit_[:, 1] > exit_[:, 2])
        return self


def narrow_state(env) -> NarrowState:
    state = getattr(env, "_narrow_state", None)
    if state is None:
        state = NarrowState(env)
        env._narrow_state = state
    return state.update()


def narrow_mask(env) -> torch.Tensor:
    return narrow_state(env).narrow
