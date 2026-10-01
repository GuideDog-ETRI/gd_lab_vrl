"""Simulator-independent geometry, also exercised by CPU mesh tests."""

import numpy as np
import trimesh


def build_boarding_mesh(difficulty, size, width_range, offset_range, center_offset, depth):
    """Central top is exactly z=0; landing steps have independent random signs.

    Use NumPy's seeded RNG, as IsaacLab's terrain generator does. Slots span
    the entire tile width. Floors are below BOTH adjacent deck surfaces.
    """
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must be in [0, 1]")
    if not 0 < width_range[0] <= width_range[1] or not 0 <= offset_range[0] <= offset_range[1]:
        raise ValueError("invalid gap width/height ranges")
    if center_offset <= width_range[1] / 2 or center_offset + width_range[1] / 2 >= size[0] / 2:
        raise ValueError("gaps must leave a central platform and two outer landings")
    if min(size) <= 0 or depth <= offset_range[1]:
        raise ValueError("invalid tile size or pit depth")
    width = width_range[0] + difficulty * (width_range[1] - width_range[0])
    offset = offset_range[0] + difficulty * (offset_range[1] - offset_range[0])
    left_z, right_z = np.random.choice([-1.0, 1.0], size=2) * offset
    cx, cy = size[0] / 2, size[1] / 2
    left, right = cx - center_offset, cx + center_offset
    meshes = []

    def slab(x0, x1, top, thickness):
        transform = trimesh.transformations.translation_matrix(((x0 + x1) / 2, cy, top - thickness / 2))
        meshes.append(trimesh.creation.box((x1 - x0, size[1], thickness), transform))

    slab(0, left - width / 2, left_z, 1.0)
    slab(left + width / 2, right - width / 2, 0.0, 1.0)
    slab(right + width / 2, size[0], right_z, 1.0)
    for center, landing_z in ((left, left_z), (right, right_z)):
        slab(center - width / 2, center + width / 2, min(0.0, landing_z) - depth, 0.1)
    return meshes, np.array([cx, cy, 0.0])
