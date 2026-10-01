"""Flat platform crossed by several narrow gaps, gap width ramped by difficulty.

A heightfield can't represent this (a heightfield is a single-valued surface
z=f(x,y) -- it cannot have an actual hole), so this is trimesh-based like
ring_fence.py/nosing_stairs.py: real solid-free gaps, not just a dip.

2026-09-13: added because the blind policy (proprioception only, no
height_scan on the actor -- see cenet_detail.pdf) has no way to *see* a gap
before stepping into it. It can only learn to avoid/cross one at all if the
critic (which does see height_scan) has actually observed gap terrain during
training to shape that behaviour via GAE -- which nothing in the existing
8 terrain types provided. Width ramps 0 -> gap_width_range[1] so early
curriculum levels get an easy, barely-there crack and only the hardest levels
approach the width that was failing in the hand-built Mujoco test course.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def gap_field_terrain(difficulty: float, cfg: GapFieldTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """``cfg.num_gaps`` parallel slots (perpendicular to the walking/+x direction),
    evenly spaced along x, each ``cfg.gap_depth`` deep with a solid pit floor
    (a fallen foot lands, it does not clip through to whatever is below this
    tile). Gap width = gap_width_range[0] + difficulty * (range[1]-range[0]).

    The platform is NOT one continuous slab with gap/wall boxes placed on top
    of it (a union mesh doesn't subtract -- that would just add material and
    leave the surface solid straight across, defeating the whole point). It is
    built from separate ground *strips*, one per run between gaps, that
    genuinely do not touch across each gap's x-range.
    """
    width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    meshes_list: list[trimesh.Trimesh] = []
    cy = cfg.size[1] / 2.0
    top_h = 1.0  # slab thickness of the walkable platform strips

    def add_strip(x0: float, x1: float) -> None:
        if x1 - x0 <= 1e-4:
            return
        cx = (x0 + x1) / 2.0
        meshes_list.append(
            trimesh.creation.box(
                (x1 - x0, cfg.size[1], top_h),
                trimesh.transformations.translation_matrix((cx, cy, -top_h / 2.0)),
            )
        )

    if width <= 1e-4:
        # difficulty 0 (or an unset range): no real gap yet, just one slab
        add_strip(0.0, cfg.size[0])
    else:
        usable = cfg.size[0] - 2.0 * cfg.border
        spacing = usable / (cfg.num_gaps + 1)
        gap_centers = [cfg.border + spacing * (i + 1) for i in range(cfg.num_gaps)]

        prev_edge = 0.0
        pit_bottom_z = -cfg.gap_depth
        for gx in gap_centers:
            g0, g1 = gx - width / 2.0, gx + width / 2.0
            add_strip(prev_edge, g0)
            # solid pit floor under this gap so a fallen foot lands
            meshes_list.append(
                trimesh.creation.box(
                    (width + 0.1, cfg.size[1], 0.1),
                    trimesh.transformations.translation_matrix((gx, cy, pit_bottom_z - 0.05)),
                )
            )
            prev_edge = g1
        add_strip(prev_edge, cfg.size[0])

    cx_all = cfg.size[0] / 2.0
    origin = np.array([cx_all, cy, 0.0])
    return meshes_list, origin


@configclass
class GapFieldTerrainCfg(SubTerrainBaseCfg):
    """Flat platform crossed by several narrow gaps, width ramped by difficulty."""

    function = gap_field_terrain

    num_gaps: int = 4
    """How many parallel gaps to cross along one tile."""

    gap_width_range: tuple[float, float] = MISSING
    """(min, max) gap width in m, interpolated by difficulty."""

    gap_depth: float = 0.5
    """Pit depth (m) below the platform surface. Has a solid bottom -- a
    fallen foot lands rather than clipping through to whatever is below."""

    border: float = 1.0
    """Clear flat run (m) before the first gap and after the last."""
