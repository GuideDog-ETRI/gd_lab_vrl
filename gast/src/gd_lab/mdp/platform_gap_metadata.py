"""Exact per-tile gap geometry recovered from the generated mesh (numpy/trimesh only).

Called at the insertion point of ``TerrainGenerator._add_sub_terrain``, where IsaacLab has
already centred the mesh (``_get_terrain_mesh`` subtracts ``size / 2``), so every x here is
relative to the tile centre == the env origin. z is relative to the tile origin (z = 0).
"""

import numpy as np
import trimesh


def boarding_tile_metadata(mesh: trimesh.Trimesh, center_offset: float, seed: int | None):
    """Recover both pit-floor bounds from the exact local mesh before placement."""
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.face_normals)
    centers = np.asarray(mesh.triangles_center)
    upward = normals[:, 2] > 0.99
    slots = []
    floors = []
    for sign in (-1, 1):
        near_pit = upward & (np.abs(centers[:, 0] - sign * center_offset) < 0.35)
        if not near_pit.any():
            raise ValueError("gap tile mesh has no upward pit faces")
        floor_z = float(centers[near_pit, 2].min())
        floor_faces = near_pit & np.isclose(centers[:, 2], floor_z, atol=1e-5)
        x = triangles[floor_faces, :, 0]
        if len(x) == 0:
            raise ValueError("gap tile mesh has no pit floor")
        slots.append((float(x.min()), float(x.max())))
        floors.append(floor_z)
    left, right = slots
    if not left[1] < right[0]:
        raise ValueError("gap slot order is invalid")
    deck_faces = upward & (centers[:, 2] > max(floors) + 0.2)
    regions = (
        deck_faces & (centers[:, 0] < left[0]),
        deck_faces & (centers[:, 0] > left[1]) & (centers[:, 0] < right[0]),
        deck_faces & (centers[:, 0] > right[1]),
    )
    if not all(region.any() for region in regions):
        raise ValueError("gap tile mesh is missing a deck")
    decks = [float(centers[region, 2].max()) for region in regions]
    return {
        "slots": slots,
        "decks": decks,
        "lower_decks": [min(decks[0], decks[1]), min(decks[1], decks[2])],
        "floors": floors,
        "seed": seed,
    }
