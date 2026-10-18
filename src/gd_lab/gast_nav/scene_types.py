"""UPDATE160 shared scene metadata. All coordinates are relative to env origin.

Only command emulation, critic/reward and oracle may consume GT metadata. It is
not an obstacle-map shortcut and must not be passed as privileged actor input.
"""
from dataclasses import dataclass
import math
from typing import Literal
import torch
from .schema import SEGMENT_TYPES

SCENE_KINDS = ('nav_corridor', 'nav_door', 'nav_turn_room', 'nav_stair_landing', 'legacy')
BOX_KINDS = ('wall', 'frame', 'box', 'pillar', 'rail', 'step')


@dataclass(frozen=True)
class StaticBox:
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float
    kind: str

    def __post_init__(self):
        if len(self.center) != 3 or len(self.half) != 3 or self.kind not in BOX_KINDS:
            raise ValueError('StaticBox shape/kind')
        if not all(math.isfinite(v) for v in (*self.center, *self.half, self.yaw)) or min(self.half) <= 0:
            raise ValueError('StaticBox geometry')


@dataclass(frozen=True)
class SceneTileMeta:
    kind: str
    level: int
    route_polyline: tuple[tuple[float, float], ...]
    width_profile: tuple[tuple[float, float], ...]
    segments: tuple[tuple[str, float, float], ...]
    landing: tuple[tuple[float, float], tuple[float, float]] | None
    turn_spot: tuple[tuple[float, float], float] | None
    obstacles: tuple[StaticBox, ...]
    floor_z_at: str
    train_inertia: bool

    def __post_init__(self):
        if self.kind not in SCENE_KINDS or self.level < 0 or self.floor_z_at not in ('flat', 'stairs'):
            raise ValueError('scene metadata kind/level/floor')
        if len(self.route_polyline) < 2 or not self.width_profile or not self.segments:
            raise ValueError('missing route/profile/segments')
        if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in self.route_polyline):
            raise ValueError('route point')
        if any(math.dist(a, b) <= 0 for a, b in zip(self.route_polyline, self.route_polyline[1:])):
            raise ValueError('zero-length route segment')
        if any(not math.isfinite(s+w) or s < 0 or w < 0 for s, w in self.width_profile):
            raise ValueError('width profile')
        if any(b[0] < a[0] for a, b in zip(self.width_profile, self.width_profile[1:])):
            raise ValueError('width profile must be ordered')
        if any(k not in SEGMENT_TYPES or not math.isfinite(a+b) or a < 0 or b <= a for k, a, b in self.segments):
            raise ValueError('route segment')
        if self.landing is not None and any(a >= b for a, b in zip(*self.landing)):
            raise ValueError('landing bounds')
        if self.turn_spot is not None and (self.turn_spot[1] < 0 or not math.isfinite(self.turn_spot[1])):
            raise ValueError('turn radius')

    @property
    def route_length(self):
        return sum(math.dist(a, b) for a, b in zip(self.route_polyline, self.route_polyline[1:]))


def project_route(meta, xy):
    """Scalar closest polyline projection -> (s, signed-left offset, yaw)."""
    best, out, distance = math.inf, None, 0.
    for a, b in zip(meta.route_polyline, meta.route_polyline[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx/length, dy/length
        along = min(length, max(0., (xy[0]-a[0])*ux+(xy[1]-a[1])*uy))
        px, py = a[0]+along*ux, a[1]+along*uy
        squared = (xy[0]-px)**2+(xy[1]-py)**2
        if squared < best:
            best = squared
            out = (distance+along, -(xy[0]-px)*uy+(xy[1]-py)*ux, math.atan2(dy, dx))
        distance += length
    return out


def route_point(meta, s):
    """Arc-distance point and unit tangent, clamped to the selected route."""
    remaining = min(meta.route_length, max(0., float(s)))
    for a, b in zip(meta.route_polyline, meta.route_polyline[1:]):
        length = math.dist(a, b)
        unit = ((b[0]-a[0])/length, (b[1]-a[1])/length)
        if remaining <= length:
            return (a[0]+remaining*unit[0], a[1]+remaining*unit[1]), unit
        remaining -= length
    return meta.route_polyline[-1], unit


class SceneIndex:
    """Immutable tiles, mutable env->tile bindings (for curriculum reassignment)."""
    def __init__(self, tiles, env_tile_ids=None):
        self.tiles = tuple(tiles)
        if not self.tiles:
            raise ValueError('empty SceneIndex')
        self.env_tile_ids = list(range(len(self.tiles))) if env_tile_ids is None else self._ids(env_tile_ids)
        if any(i < 0 or i >= len(self.tiles) for i in self.env_tile_ids):
            raise IndexError('tile id')

    @staticmethod
    def _ids(value):
        return [int(x) for x in (value.detach().cpu().tolist() if torch.is_tensor(value) else value)]

    def lookup(self, env_ids):
        ids = self._ids(env_ids)
        if any(i < 0 or i >= len(self.env_tile_ids) for i in ids):
            raise IndexError('env id')
        return [self.tiles[self.env_tile_ids[i]] for i in ids]

    def bind(self, env_ids, tile_ids):
        env, tile = self._ids(env_ids), self._ids(tile_ids)
        if len(env) != len(tile) or any(i < 0 or i >= len(self.env_tile_ids) for i in env) or any(i < 0 or i >= len(self.tiles) for i in tile):
            raise ValueError('invalid scene binding')
        for i, j in zip(env, tile): self.env_tile_ids[i] = j

    @classmethod
    def from_grid(cls, grid, terrain_levels, terrain_types):
        """grid[(row,col)] from GastNavTerrainGenerator -> current env bindings."""
        keys = sorted(grid)
        mapping = {key: i for i, key in enumerate(keys)}
        levels, types = cls._ids(terrain_levels), cls._ids(terrain_types)
        if len(levels) != len(types): raise ValueError('terrain binding lengths')
        return cls([grid[k] for k in keys], [mapping[(r, c)] for r, c in zip(levels, types)])

    def tensors(self, device='cpu'):
        """Padded env-wise metadata; masks delimit valid points/profile/segments.

        All geometry stays env-local. Add env.scene.env_origins only when world
        coordinates are required; never expose GT arrays as sensor observations.
        """
        metas = self.lookup(range(len(self.env_tile_ids))); n = len(metas)
        p, w, q = max(len(m.route_polyline) for m in metas), max(len(m.width_profile) for m in metas), max(len(m.segments) for m in metas)
        result = {
            'kind': torch.tensor([SCENE_KINDS.index(m.kind) for m in metas], device=device),
            'level': torch.tensor([m.level for m in metas], device=device),
            'route_points': torch.zeros(n, p, 2, device=device), 'route_mask': torch.zeros(n, p, dtype=torch.bool, device=device),
            'width_profile': torch.zeros(n, w, 2, device=device), 'width_mask': torch.zeros(n, w, dtype=torch.bool, device=device),
            'segment_type': torch.zeros(n, q, dtype=torch.long, device=device), 'segment_bounds': torch.zeros(n, q, 2, device=device),
            'segment_mask': torch.zeros(n, q, dtype=torch.bool, device=device),
            'landing': torch.zeros(n, 2, 2, device=device), 'landing_mask': torch.tensor([m.landing is not None for m in metas], device=device),
            'turn_center': torch.zeros(n, 2, device=device), 'turn_radius': torch.zeros(n, device=device),
            'turn_mask': torch.tensor([m.turn_spot is not None for m in metas], device=device),
            'route_axis_yaw': torch.tensor([project_route(m, m.route_polyline[0])[2] for m in metas], device=device),
            'available_width': torch.tensor([min(x[1] for x in m.width_profile) for m in metas], device=device),
            'route_length': torch.tensor([m.route_length for m in metas], device=device),
            'train_inertia': torch.tensor([m.train_inertia for m in metas], device=device),
            'spawn_xy': torch.tensor([m.turn_spot[0] if m.turn_spot else m.route_polyline[0] for m in metas], device=device),
        }
        for i, m in enumerate(metas):
            result['route_points'][i, :len(m.route_polyline)] = torch.tensor(m.route_polyline, device=device)
            result['route_mask'][i, :len(m.route_polyline)] = True
            result['width_profile'][i, :len(m.width_profile)] = torch.tensor(m.width_profile, device=device)
            result['width_mask'][i, :len(m.width_profile)] = True
            result['segment_type'][i, :len(m.segments)] = torch.tensor([SEGMENT_TYPES.index(k) for k, _, _ in m.segments], device=device)
            result['segment_bounds'][i, :len(m.segments)] = torch.tensor([(a,b) for _, a,b in m.segments], device=device)
            result['segment_mask'][i, :len(m.segments)] = True
            if m.landing: result['landing'][i] = torch.tensor(m.landing, device=device)
            if m.turn_spot:
                result['turn_center'][i] = torch.tensor(m.turn_spot[0], device=device)
                result['turn_radius'][i] = m.turn_spot[1]
        return result
