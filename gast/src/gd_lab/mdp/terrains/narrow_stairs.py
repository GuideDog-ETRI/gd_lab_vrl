"""Straight staircase flanked by side walls whose gap narrows with difficulty.

2026-09-15: added after real-world observation that the robot repeatedly
bumps into stair railings/walls while climbing or descending in tight
stairwells -- nothing in the existing terrain set combines "stairs" with "a
nearby lateral wall to avoid," so a robot that never drifts sideways enough
to get penalized on the wide-open stair terrains has no reason to hold a
tight lateral track. Corridor width is the difficulty axis: easy = 1.5m
(plenty of clearance) down to hard = 0.9m -- the narrowest width the real
robot is known to fit through (a turnstile gap), so the hard case stays
physically passable, just unforgiving of drift.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _box(size_xyz, center_xyz):
    return trimesh.creation.box(size_xyz, trimesh.transformations.translation_matrix(center_xyz))


def narrow_stairs_terrain(
    difficulty: float, cfg: NarrowStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Platform -> N ascending steps (walled corridor) -> platform.

    Riser height scales with difficulty like the other stair terrains
    (``step_height_range``). Corridor width scales the *opposite* way
    (``corridor_width_range[0]`` at difficulty 0 down to ``[1]`` at
    difficulty 1) -- pass e.g. (1.5, 0.9): narrower is harder, not wider.
    The walls run only alongside the steps themselves, not the flat
    platforms before/after.
    """
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    corridor_width = cfg.corridor_width_range[0] + difficulty * (
        cfg.corridor_width_range[1] - cfg.corridor_width_range[0]
    )

    meshes_list: list[trimesh.Trimesh] = []
    cy = cfg.size[1] / 2.0
    n = cfg.num_steps
    tread = cfg.step_width
    fill_below = 3.0  # solid fill depth below each tread's own top

    run_total = 2 * cfg.platform_width + n * tread
    x0 = (cfg.size[0] - run_total) / 2.0  # center the whole run in the tile

    # entry platform, ground level
    meshes_list.append(
        _box((cfg.platform_width, cfg.size[1], fill_below), (x0 + cfg.platform_width / 2.0, cy, -fill_below / 2.0))
    )

    # ascending steps (full tile width -- the corridor walls are the only
    # thing narrowing the usable path, not the steps themselves)
    wall_x0 = x0 + cfg.platform_width
    cx = wall_x0
    for i in range(n):
        top = (i + 1) * step_height
        meshes_list.append(_box((tread, cfg.size[1], fill_below), (cx + tread / 2.0, cy, top - fill_below / 2.0)))
        cx += tread
    top_final = n * step_height
    wall_len = n * tread

    # exit platform, flush with the top step
    meshes_list.append(
        _box((cfg.platform_width, cfg.size[1], fill_below), (cx + cfg.platform_width / 2.0, cy, top_final - fill_below / 2.0))
    )

    # side walls: span only the stair run in x; tall enough to clear the
    # highest step everywhere along the run, deep enough to clear the lowest.
    wall_bottom_z = -1.0
    wall_top_z = top_final + cfg.wall_height
    wall_cz = (wall_bottom_z + wall_top_z) / 2.0
    wall_full_h = wall_top_z - wall_bottom_z
    wall_cx = wall_x0 + wall_len / 2.0
    for sign in (1.0, -1.0):
        wy = sign * (corridor_width / 2.0 + cfg.wall_thickness / 2.0)
        meshes_list.append(_box((wall_len, cfg.wall_thickness, wall_full_h), (wall_cx, wy, wall_cz)))

    origin = np.array([cfg.size[0] / 2.0, cy, 0.0])
    return meshes_list, origin


@configclass
class NarrowStairsTerrainCfg(SubTerrainBaseCfg):
    """Straight staircase between two side walls; the gap between them narrows with difficulty."""

    function = narrow_stairs_terrain

    num_steps: int = 5
    """Risers in the flight."""

    step_width: float = 0.3
    """Tread depth (m) per riser."""

    step_height_range: tuple[float, float] = MISSING
    """(min, max) riser height in m, interpolated by difficulty."""

    corridor_width_range: tuple[float, float] = MISSING
    """(easy, hard) usable width in m between the two walls, interpolated by
    difficulty -- pass e.g. (1.5, 0.9): *narrows* as difficulty rises, unlike
    every other range in this codebase. 0.9m is the narrowest gap the real
    robot is known to fit through, so the hard case stays passable."""

    wall_height: float = 0.4
    """How far the walls stand above the highest step (m)."""

    wall_thickness: float = 0.05
    """Wall slab thickness (m)."""

    platform_width: float = 2.0
    """Flat, unwalled run (m) before and after the stair flight."""
