"""Rendering-free camera visibility of the 11x17 height-scan cells.

Mirrors gd_lab.core.camera_geometry.camera_visible_points (same pinhole model,
frame bounds, depth clip and four-neighbour depth agreement), but obtains each
neighbouring pixel's depth by casting a ray through it against the static
terrain mesh instead of reading a rendered depth image. Legs use capsule
proxies; optional enhanced occlusion adds a trunk box and foot spheres, with
conservative depth-edge rejection. These proxies are not exact rendered meshes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from gd_lab.core.camera_geometry import camera_rotation_matrix

# (starts[M,3], unit directions[M,3], max_dist) -> hit distance[M] (inf on miss)
CastFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def contract_intrinsic(contract, device) -> torch.Tensor:
    """3x3 K of the canonical student image (square pixels, centred principal point)."""
    return torch.tensor(
        [[contract.fx, 0.0, contract.width / 2], [0.0, contract.fy, contract.height / 2], [0.0, 0.0, 1.0]],
        device=device,
    )


def ray_hits_capsules(starts, directions, lengths, seg_a, seg_b, radius) -> torch.Tensor:
    """M rays [start, start + length*dir] vs M,S capsules (a->b, radius[S]); True if any touches.

    Closest points of two segments: clamp the ray parameter, project onto the
    capsule axis, then re-project onto the ray (exact up to one clamp pass).
    """
    u = (directions * lengths[:, None])[:, None]  # M,1,3
    v = seg_b - seg_a  # M,S,3
    w0 = starts[:, None] - seg_a
    uu = (u * u).sum(-1).clamp_min(1e-12)
    uv = (u * v).sum(-1)
    vv = (v * v).sum(-1).clamp_min(1e-12)
    uw = (u * w0).sum(-1)
    vw = (v * w0).sum(-1)
    s = ((uv * vw - vv * uw) / (uu * vv - uv * uv).clamp_min(1e-12)).clamp(0, 1)
    t = ((uv * s + vw) / vv).clamp(0, 1)
    s = ((uv * t - uw) / uu).clamp(0, 1)
    gap = w0 + s[..., None] * u - t[..., None] * v
    return (gap.norm(dim=-1) < radius).any(-1)


def ray_hits_box(starts, directions, lengths, center, rotation, half_size):
    """Finite ray vs oriented box; rotation maps local coordinates to world."""
    origin = ((starts-center)[:, None] @ rotation).squeeze(1)
    direction = (directions[:, None] @ rotation).squeeze(1)
    parallel = direction.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(direction), direction)
    a, b = (-half_size-origin)/safe, (half_size-origin)/safe
    near = torch.minimum(a,b).masked_fill(parallel, -torch.inf).amax(-1)
    far = torch.maximum(a,b).masked_fill(parallel, torch.inf).amin(-1)
    outside = (parallel & (origin.abs() > half_size)).any(-1)
    return ~outside & (far >= near.clamp_min(0)) & (far > 1e-5) & (near < lengths-1e-4)


def ray_hits_spheres(starts, directions, lengths, centers, radius):
    delta = centers-starts[:,None]
    along = (delta*directions[:,None]).sum(-1).clamp_min(0)
    along = torch.minimum(along, lengths[:,None])
    closest = starts[:,None]+along[...,None]*directions[:,None]
    return ((closest-centers).square().sum(-1) <= radius**2).any(-1)


def raycast_visible_points(points_w, camera_pos, camera_quat_ros, intrinsic, image_hw, depth_clip,
                           cast: CastFn, occluders=None, body_occluders=None,
                           conservative=False) -> torch.Tensor:
    """N,P world points visible in ONE camera: in frame, within depth clip, unoccluded.

    ``occluders`` = (seg_a[N,S,3], seg_b[N,S,3], radius[S]) robot capsules, optional.
    """
    h, w = image_hw
    rotation = camera_rotation_matrix(camera_quat_ros)
    camera_points = (points_w - camera_pos[:, None]) @ rotation
    z = camera_points[..., 2]
    k = intrinsic if intrinsic.dim() == 3 else intrinsic.expand(points_w.shape[0], 3, 3)
    homogeneous = camera_points @ k.transpose(1, 2)
    uv = homogeneous[..., :2] / z.clamp_min(1e-6)[..., None]
    x0, y0 = torch.nan_to_num(uv, nan=-1.0, posinf=-1.0, neginf=-1.0).floor().long().unbind(-1)
    lo, hi = depth_clip
    visible = ((x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
               & torch.isfinite(points_w).all(-1) & (z >= lo) & (z < hi))
    candidates = visible.nonzero(as_tuple=True)
    if not candidates[0].numel():
        return visible
    env_ids = candidates[0]
    target_z = z[candidates]
    tolerance = 0.015 + 0.005 * target_z.abs()
    k_inv = torch.linalg.inv(k[env_ids])
    world_from_camera = rotation[env_ids]
    agree = torch.ones_like(target_z, dtype=torch.bool)
    minimum = torch.full_like(target_z, torch.inf)
    maximum = torch.full_like(target_z, -torch.inf)
    # Emulate the rendered test: the four neighbouring pixels' depth must agree.
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        pixel = torch.stack(((x0[candidates] + dx).float(), (y0[candidates] + dy).float(),
                             torch.ones_like(target_z)), -1)
        ray_camera = (k_inv @ pixel[..., None])[..., 0]
        length = ray_camera.norm(dim=-1)
        direction = (world_from_camera @ (ray_camera / length[:, None])[..., None])[..., 0]
        hit = cast(camera_pos[env_ids].contiguous(), direction.contiguous(), float(hi) * 2.0)
        depth = hit / length  # optical depth: ray_camera has unit z
        minimum = torch.minimum(minimum, depth)
        maximum = torch.maximum(maximum, depth)
        agree &= torch.isfinite(depth) & (depth >= lo) & (depth < hi) & ((depth - target_z).abs() <= tolerance)
        if occluders is not None:
            seg_a, seg_b, radius = occluders
            agree &= ~ray_hits_capsules(camera_pos[env_ids], direction, target_z * length,
                                        seg_a[env_ids], seg_b[env_ids], radius)
        if body_occluders is not None:
            center, body_rotation, half_size, feet, foot_radius = body_occluders
            # Match the optical near clip: cameras may lie inside the trunk's
            # coarse box. Do not spuriously block rays before the near plane.
            near_length = lo*length
            starts = camera_pos[env_ids] + direction*near_length[:,None]
            lengths = (target_z*length-near_length).clamp_min(0)
            agree &= ~ray_hits_box(starts, direction, lengths, center[env_ids],
                                  body_rotation[env_ids], half_size)
            agree &= ~ray_hits_spheres(starts, direction, lengths, feet[env_ids], foot_radius)
    if conservative:
        agree &= (maximum-minimum) <= tolerance
    visible[candidates] = agree
    return visible


def warp_mesh_cast(mesh) -> CastFn:
    """Distance-only raycast against one IsaacLab warp mesh."""
    from isaaclab.utils.warp import raycast_mesh

    def cast(starts, directions, max_dist):
        _, distance, _, _ = raycast_mesh(starts[None], directions[None], mesh, max_dist=max_dist,
                                         return_distance=True)
        return distance[0]

    return cast
