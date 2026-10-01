"""Concentric ring-wall terrain around a central spawn platform."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def multi_ring_fence_terrain(
    difficulty: float, cfg: MultiRingFenceTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """N concentric rectangular ring walls evenly spaced around the spawn.

    Robot spawns on the central ``platform_width x platform_width`` square.
    ``n_rings`` rectangular ring walls of thickness ``thickness`` and height
    interpolated from ``height_range`` are placed at evenly-spaced half-widths
    between the platform edge and the tile border. Crossing each ring
    geometrically forces the "leading leg clears the wall -> wall sits between
    fore/rear feet -> trailing leg stubs" pattern, regardless of which radial
    direction the robot heads -- a focused trainer for under-body obstacles the
    height scan barely sees.
    """
    height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])

    meshes_list: list[trimesh.Trimesh] = []
    cx, cy = cfg.size[0] / 2.0, cfg.size[1] / 2.0
    plat_half = cfg.platform_width / 2.0
    max_half = min(cfg.size) / 2.0 - cfg.border

    # Spread rings across the full range [plat_half, max_half - thickness]
    # (endpoints inclusive) -- max spacing between rings, with the inner ring
    # at the platform edge and the outer ring at the tile border.
    if cfg.n_rings == 1:
        inner_halves = [(plat_half + max_half - cfg.thickness) / 2.0]
    else:
        inner_halves = np.linspace(plat_half, max_half - cfg.thickness, cfg.n_rings)

    for inner_half in inner_halves:
        outer_half = inner_half + cfg.thickness
        outer_size = (outer_half * 2.0, outer_half * 2.0)
        inner_size = (inner_half * 2.0, inner_half * 2.0)
        ring_center = (cx, cy, height * 0.5)
        meshes_list += make_border(outer_size, inner_size, height, ring_center)

    terrain_h = 1.0
    ground = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], terrain_h),
        trimesh.transformations.translation_matrix((cx, cy, -terrain_h / 2.0)),
    )
    meshes_list.append(ground)

    origin = np.array([cx, cy, 0.0])
    return meshes_list, origin


@configclass
class MultiRingFenceTerrainCfg(SubTerrainBaseCfg):
    """Concentric rectangular ring walls around a central platform.

    Each ring forces a "leading leg clears the wall -> wall sits between
    fore/rear feet -> trailing leg stubs" event as the robot walks outward --
    a focused trainer for under-body obstacles the height scan barely sees.
    """

    function = multi_ring_fence_terrain

    n_rings: int = 4
    """Number of concentric ring walls."""

    thickness: float = 0.06
    """Wall thickness (radial extent) in m. Should be <= stride length so the
    leading leg clears and the trailing leg produces the stub event."""

    height_range: tuple[float, float] = MISSING
    """(min, max) wall height in m, interpolated by difficulty."""

    platform_width: float = 2.0
    """Central spawn platform side length in m. No walls inside this square."""

    border: float = 0.3
    """Padding (m) between the outermost ring and the tile edge so the outer
    ring is fully on-tile and doesn't clip the tile border."""
