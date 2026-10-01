"""Descend a staircase, then -- after only a short (curriculum-shrinking) flat
run -- immediately climb another one.

2026-09-13: added after this exact transition (pyramid stairs down, straight
into an ascending staircase with ~0 recovery distance) caught the deployed
policy's front foot in the hand-built Mujoco test course. Nothing in the
existing 8 terrain types chains two staircases back-to-back like this --
each stair terrain type occupies its own column and a robot only ever visits
tiles within its assigned column (terrain type), so a robot training on
``pyramid_stairs`` never experiences going down one flight straight into
another. This is a dedicated terrain for exactly that transition, with the
recovery distance as the difficulty axis: easy = ~1.5m flat run to resettle
gait/pitch after the descent, hard = none at all.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _box(size_xyz, center_xyz):
    return trimesh.creation.box(size_xyz, trimesh.transformations.translation_matrix(center_xyz))


def stair_transition_terrain(
    difficulty: float, cfg: StairTransitionTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Platform -> descending stairs -> flat recovery run -> ascending stairs -> platform.

    Riser height scales with difficulty like the other stair terrains
    (``step_height_range``). The recovery run does the *opposite*: it
    *shrinks* as difficulty increases (``recovery_range[0]`` at difficulty 0
    down to ``recovery_range[1]`` at difficulty 1, so pass e.g. (1.5, 0.0)),
    since the hard case here is *no* recovery room, not more of it.
    """
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    recovery = cfg.recovery_range[0] + difficulty * (cfg.recovery_range[1] - cfg.recovery_range[0])

    meshes_list: list[trimesh.Trimesh] = []
    cy = cfg.size[1] / 2.0
    n = cfg.num_steps
    tread = cfg.step_width
    fill_below = 3.0  # solid fill depth below each step's own top, well past any lower step

    run_total = 2 * n * tread + recovery
    x0 = (cfg.size[0] - run_total) / 2.0  # center the whole sequence in the tile

    # spawn platform, flush with the descent's starting (highest) level
    top0 = n * step_height
    if x0 > 1e-4:
        meshes_list.append(_box((x0, cfg.size[1], fill_below), (x0 / 2.0, cy, top0 - fill_below / 2.0)))

    # descending stairs: n risers going down to 0
    cx = x0
    for i in range(n):
        top = top0 - (i + 1) * step_height
        meshes_list.append(_box((tread, cfg.size[1], fill_below), (cx + tread / 2.0, cy, top - fill_below / 2.0)))
        cx += tread

    # flat recovery run at the bottom (0 at difficulty 0's widest, shrinking to ~0)
    if recovery > 1e-4:
        meshes_list.append(_box((recovery, cfg.size[1], fill_below), (cx + recovery / 2.0, cy, -fill_below / 2.0)))
        cx += recovery

    # ascending stairs: n risers back up to top0
    for i in range(n):
        top = (i + 1) * step_height
        meshes_list.append(_box((tread, cfg.size[1], fill_below), (cx + tread / 2.0, cy, top - fill_below / 2.0)))
        cx += tread

    # exit platform, flush with the ascent's ending (highest) level
    x_end = cfg.size[0]
    if x_end - cx > 1e-4:
        meshes_list.append(
            _box((x_end - cx, cfg.size[1], fill_below), ((cx + x_end) / 2.0, cy, top0 - fill_below / 2.0))
        )

    origin = np.array([cfg.size[0] / 2.0, cy, 0.0])
    return meshes_list, origin


@configclass
class StairTransitionTerrainCfg(SubTerrainBaseCfg):
    """Descend a staircase then immediately climb another; recovery distance shrinks with difficulty."""

    function = stair_transition_terrain

    num_steps: int = 5
    """Risers per flight (descent and ascent each have this many)."""

    step_width: float = 0.3
    """Tread depth (m) per riser -- matches the other stair terrains' step_width."""

    step_height_range: tuple[float, float] = MISSING
    """(min, max) riser height in m, interpolated by difficulty (same convention
    as the other stair terrains: bigger difficulty -> taller risers)."""

    recovery_range: tuple[float, float] = MISSING
    """(easy, hard) flat run in m at the bottom between the two flights,
    interpolated by difficulty -- pass e.g. (1.5, 0.0): *shrinks* as difficulty
    rises, unlike every other range in this codebase. The hard case here is
    no recovery room, not more of it."""
