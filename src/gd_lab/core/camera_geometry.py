"""Torch-only calibrated projection and conservative depth visibility checks."""

import torch
import torch.nn.functional as F


def camera_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Camera-to-world matrix from normalized wxyz, arbitrary leading dimensions."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(*quat.shape[:-1], 3, 3)


def calibrated_resample(image, source_k, target_k, output_shape, mode="nearest"):
    """NHWC -> NHWC; inverse-map canonical pixels into square-pixel render.

    Target and source K use the same pixel-center convention. Reject FoVs
    outside the rendered canvas instead of silently extending border pixels.
    """
    n, h, w, _ = image.shape
    oh, ow = output_shape
    yy, xx = torch.meshgrid(torch.arange(oh, device=image.device), torch.arange(ow, device=image.device), indexing="ij")
    sx = (xx[None] - target_k[:, 0, 2, None, None]) / target_k[:, 0, 0, None, None]
    sy = (yy[None] - target_k[:, 1, 2, None, None]) / target_k[:, 1, 1, None, None]
    sx = sx * source_k[:, 0, 0, None, None] + source_k[:, 0, 2, None, None]
    sy = sy * source_k[:, 1, 1, None, None] + source_k[:, 1, 2, None, None]
    if ((sx < 0) | (sx > w - 1) | (sy < 0) | (sy > h - 1)).any():
        raise ValueError("Canonical camera FoV exceeds the overscan canvas")
    grid = torch.stack((2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1), -1)
    return F.grid_sample(image.float().permute(0, 3, 1, 2), grid.expand(n, -1, -1, -1),
                         mode=mode, align_corners=True).permute(0, 2, 3, 1)


def camera_visible_points(points_w, camera_pos, camera_quat_ros, intrinsic, depth, depth_clip=(0.15, 5.0)):
    """N,P world points visible in ONE camera, including robot/terrain occlusion.

    Require all four adjacent depth pixels to agree with the target depth.
    This conservatively rejects silhouettes/depth edges. Invalid/missing
    returns never prove visibility. Caller unions masks over cameras.
    """
    n, h, w = depth.shape
    rotation = camera_rotation_matrix(camera_quat_ros)
    camera_points = (points_w - camera_pos[:, None]) @ rotation.transpose(1, 2)
    z = camera_points[..., 2]
    homogeneous = camera_points @ intrinsic.transpose(1, 2)
    uv = homogeneous[..., :2] / z.clamp_min(1e-6)[..., None]
    uv_safe = torch.nan_to_num(uv, nan=-1.0, posinf=-1.0, neginf=-1.0)
    x0, y0 = uv_safe.floor().long().unbind(-1)
    in_frame = (x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
    lo, hi = depth_clip
    visible = in_frame & torch.isfinite(camera_points).all(-1) & (z >= lo) & (z < hi)
    flat = depth.reshape(n, -1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        index = (y0 + dy).clamp(0, h - 1) * w + (x0 + dx).clamp(0, w - 1)
        sampled = flat.gather(1, index)
        visible &= torch.isfinite(sampled) & (sampled >= lo) & (sampled < hi)
        visible &= (sampled - z).abs() <= (0.015 + 0.005 * z.abs())
    return visible


def depth_pixels_world(depth, positions, quaternions_ros, intrinsic):
    """N,C,H,W optical depth -> N,C,H,W,3 world points."""
    n, c, h, w = depth.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij")
    x = (xx - intrinsic[..., 0, 2, None, None]) / intrinsic[..., 0, 0, None, None]
    y = (yy - intrinsic[..., 1, 2, None, None]) / intrinsic[..., 1, 1, None, None]
    rays = torch.stack((x.expand(n, c, h, w), y.expand(n, c, h, w), torch.ones_like(depth)), -1)
    points = (rays * depth[..., None]).reshape(n, c, h*w, 3)
    points = points @ camera_rotation_matrix(quaternions_ros).transpose(-1, -2) + positions[:, :, None]
    return points.reshape(n, c, h, w, 3)
