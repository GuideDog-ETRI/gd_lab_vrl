"""B': the downward Depth+IR terrain input does not see the narrow-passage walls (user decision 11-01: passages are
sensed by the LiDAR only; the belly D4xx sees about 0.44 m around the feet, GAST_NARROW_V1_DESIGN.md section 1).

The walls are part of the terrain mesh, so RaycastIrTerrain sees their tops. On narrow tiles this class captures as
if the tile were an empty flat floor: the scan points are put on the floor (env origin z) and the camera visibility
test casts against that plane instead of the terrain mesh (over a flat floor nothing but the legs occludes a floor
point). Measured 11-01 (A'-type policy on narrow tiles): with the walls the height channel already shows only the
floor (wall tops are out of the belly cameras' view), but the wall shadows remove 11 of ~79 visible cells -- a wall
cue the LiDAR-only rule takes away. Other tiles unchanged.
"""
from __future__ import annotations

import torch

from gd_lab.gast.rayir import RaycastIrTerrain
from gd_lab.gast_narrow import geometry
from gd_lab.gast_narrow.state import narrow_state


def passage_mask(env) -> torch.Tensor:
    """Narrow tiles without the cliff decks (C', v2.6): a deck's env origin is its top, 1.0 m above the tile floor,
    and capturing it as a flat floor would hide the drop (and fail the one-height check below). Equal to narrow_mask
    on a terrain without cliff columns (B')."""
    state = narrow_state(env)
    return state.narrow & (state.kind != geometry.KINDS["cliff"])


def _plane_cast(z0: float):
    """Distance-only cast against the horizontal plane z = z0 (the visibility test wants the hit depth to match the
    point depth, as a rendered depth image would)."""
    def cast(starts, directions, max_dist):
        dz = directions[..., 2]
        t = (z0 - starts[..., 2]) / torch.where(dz.abs() > 1e-9, dz, torch.full_like(dz, -1e-9))
        return torch.where((t > 0) & (t <= max_dist), t, torch.full_like(t, float("inf")))
    return cast


class NarrowFloorRaycastIrTerrain(RaycastIrTerrain):
    def _capture(self, env, strength):
        fresh = super()._capture(env, strength)
        narrow = passage_mask(env)
        if not bool(narrow.any()):
            return fresh
        z = env.scene.env_origins[narrow, 2]
        if float(z.max() - z.min()) > 1e-3:  # narrow tiles put the origin on the floor plane at the terrain base
            raise ValueError(f"narrow floors are not at one height: {float(z.min())}..{float(z.max())}")
        data = env.scene["height_scanner"].data
        hits, cast = data.ray_hits_w, self.cast
        floor = hits.clone()
        floor[..., 2] = torch.where(narrow[:, None], z[0], hits[..., 2])
        try:
            data.ray_hits_w, self.cast = floor, _plane_cast(float(z[0]))
            flat = super()._capture(env, strength)
        finally:
            data.ray_hits_w, self.cast = hits, cast
        return torch.where(narrow[:, None], flat, fresh)
