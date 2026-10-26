"""Narrow-passage sub-terrains (flat floor + box walls) and a generator that records each tile's geometry."""
from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from gd_lab.gast_narrow import geometry
from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator


def tile_geometry(difficulty: float, cfg: "NarrowPassageTerrainCfg") -> dict:
    width = geometry.width_from_difficulty(difficulty, cfg.w_max, cfg.w_min)
    return geometry.build(cfg.kind, width, cfg.sign, geometry.dead_end_x(difficulty))


def narrow_passage_terrain(difficulty: float, cfg: "NarrowPassageTerrainCfg"):
    geo = tile_geometry(difficulty, cfg)
    cx, cy = cfg.size[0] / 2.0, cfg.size[1] / 2.0
    meshes = [make_plane(cfg.size, 0.0, center_zero=False)]
    for x0, x1, y0, y1 in geo["rects"]:
        transform = np.eye(4)
        transform[:3, 3] = (cx + (x0 + x1) / 2, cy + (y0 + y1) / 2, geometry.WALL_H / 2)
        meshes.append(trimesh.creation.box((x1 - x0, y1 - y0, geometry.WALL_H), transform))
    return meshes, np.array((cx, cy, 0.0))


@configclass
class NarrowPassageTerrainCfg(SubTerrainBaseCfg):
    function = narrow_passage_terrain
    kind: str = "corridor"
    sign: int = 1
    w_max: float = geometry.W_MAX
    w_min: float = geometry.W_MIN


class NarrowTerrainGenerator(GapMetadataTerrainGenerator):
    """Gap metadata as before, plus ``cfg.narrow_tile_metadata[row][col]`` = geometry.build(...) of narrow tiles."""

    def __init__(self, cfg, device="cpu"):
        if cfg.use_cache:
            raise ValueError("narrow tiles need use_cache=False (geometry is recorded at generation)")
        cfg.narrow_tile_metadata = [[None for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]
        self._last_difficulty = None
        super().__init__(cfg, device=device)

    def _get_terrain_mesh(self, difficulty, cfg):
        self._last_difficulty = difficulty
        return super()._get_terrain_mesh(difficulty, cfg)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if isinstance(sub_terrain_cfg, NarrowPassageTerrainCfg):
            geo = tile_geometry(self._last_difficulty, sub_terrain_cfg)
            geo.update(row=row, col=col, difficulty=float(self._last_difficulty))
            self.cfg.narrow_tile_metadata[row][col] = geo
        return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
