"""Pyramid-stairs terrains with a nosing (overhang lip) at each step edge.

IsaacLab's built-in ``MeshPyramidStairs*`` make plain box steps with a clean
vertical riser. Real stairs have a small *nosing*: the tread overhangs the
riser below it by a few cm. A climbing foot/toe must clear that lip, so
training on it makes the policy robust to catching on edges.

These two functions copy the built-in pyramid-stairs generators and append,
per step ring, a closed rectangular overhang frame at the tread's top edge.
Normal pyramid (centre high -> descent): overhang OUTWARD. Inverted (centre
pit -> ascent): overhang INWARD. Defaults: 3 cm depth x 3 cm thickness.
"""

from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
)
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def _box(size_xyz, center_xyz):
    return trimesh.creation.box(size_xyz, trimesh.transformations.translation_matrix(center_xyz))


def _add_nosing(meshes_list, cx, cy, z_top, full_x, full_y, r_x, r_y, sign, nose_depth, nose_th):
    """Append a closed rectangular overhang frame (4 lips) at the tread's edge.

    ``r_x``/``r_y`` are distances from the terrain centre to the face the lip
    attaches to; ``full_x``/``full_y`` are the step's outer box dimensions.
    ``sign`` = +1 protrudes away from centre (outward / descent), -1 toward
    centre (inward / ascent). Lip top flush with the tread top, hanging down
    ``nose_th``, depth ``nose_depth`` past the face; sides extended past
    corners so it closes.
    """
    zc = z_top - nose_th / 2.0
    d = nose_depth / 2.0
    lx = full_x + 2.0 * nose_depth
    ly = full_y + 2.0 * nose_depth
    meshes_list.append(_box((lx, nose_depth, nose_th), (cx, cy + r_y + sign * d, zc)))
    meshes_list.append(_box((lx, nose_depth, nose_th), (cx, cy - r_y - sign * d, zc)))
    meshes_list.append(_box((nose_depth, ly, nose_th), (cx + r_x + sign * d, cy, zc)))
    meshes_list.append(_box((nose_depth, ly, nose_th), (cx - r_x - sign * d, cy, zc)))


def pyramid_stairs_nosing_terrain(
    difficulty: float, cfg: MeshPyramidStairsNosingTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Pyramid stairs (centre = high platform) with an outward nosing per step."""
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    nose_d, nose_t = cfg.nose_depth, cfg.nose_thickness

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    meshes_list = list()

    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        box_height = (k + 2) * step_height
        box_dims = (box_size[0], cfg.step_width, box_height)
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        if cfg.holes:
            box_dims_rl = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims_rl = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims_rl, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims_rl, trimesh.transformations.translation_matrix(box_pos))
        meshes_list += [box_top, box_bottom, box_right, box_left]
        z_top = box_z + box_height / 2.0
        _add_nosing(
            meshes_list, terrain_center[0], terrain_center[1], z_top, box_size[0], box_size[1],
            terrain_size[0] / 2.0 - k * cfg.step_width, terrain_size[1] / 2.0 - k * cfg.step_width,
            +1.0, nose_d, nose_t,
        )

    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
    _add_nosing(
        meshes_list, terrain_center[0], terrain_center[1], (num_steps + 1) * step_height,
        box_dims[0], box_dims[1], box_dims[0] / 2.0, box_dims[1] / 2.0,
        +1.0, nose_d, nose_t,
    )
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin


def inverted_pyramid_stairs_nosing_terrain(
    difficulty: float, cfg: MeshInvertedPyramidStairsNosingTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Inverted pyramid stairs (centre = pit) with an inward nosing per step."""
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    nose_d, nose_t = cfg.nose_depth, cfg.nose_thickness

    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height

    meshes_list = list()

    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        box_height = total_height - (k + 1) * step_height
        box_dims = (box_size[0], cfg.step_width, box_height)
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        if cfg.holes:
            box_dims_rl = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims_rl = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims_rl, trimesh.transformations.translation_matrix(box_pos))
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims_rl, trimesh.transformations.translation_matrix(box_pos))
        meshes_list += [box_top, box_bottom, box_right, box_left]
        z_top = box_z + box_height / 2.0
        _add_nosing(
            meshes_list, terrain_center[0], terrain_center[1], z_top, box_size[0], box_size[1],
            terrain_size[0] / 2.0 - (k + 1) * cfg.step_width, terrain_size[1] / 2.0 - (k + 1) * cfg.step_width,
            -1.0, nose_d, nose_t,
        )

    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin


@configclass
class MeshPyramidStairsNosingTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Pyramid stairs with an outward nosing lip per step."""

    function = pyramid_stairs_nosing_terrain
    nose_depth: float = 0.03
    nose_thickness: float = 0.03


@configclass
class MeshInvertedPyramidStairsNosingTerrainCfg(MeshInvertedPyramidStairsTerrainCfg):
    """Inverted pyramid stairs with an inward nosing lip per step."""

    function = inverted_pyramid_stairs_nosing_terrain
    nose_depth: float = 0.03
    nose_thickness: float = 0.03
