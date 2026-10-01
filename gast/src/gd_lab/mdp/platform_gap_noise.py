"""Platform-gap-only depth ghost and false-return augmentation.

The teacher remains clean.  This module mutates only the student's depth
channel, and only on environments/ pixels that project into the two configured
boarding slots.  IR is deliberately untouched so the student can learn to use
its complementary cue when depth lies.
"""
from dataclasses import dataclass

import torch

from gd_lab.core.camera_geometry import depth_pixels_world


def _sample(bounds, shape, device):
    lo, hi = bounds
    return lo + torch.rand(shape, device=device) * (hi - lo)


@dataclass
class PlatformGapDepthGhostCfg:
    probability: float = 0.25
    lifetime_steps: tuple[int, int] = (2, 6)
    center_offset: float = 1.5
    max_gap_width: float = 0.24
    margin: float = 0.20
    tile_half_width: float = 1.5
    min_depth: float = 0.15
    max_depth: float = 5.0
    false_hole_depth: tuple[float, float] = (2.5, 5.0)
    false_hole_min_offset: float = 0.15
    false_return_offset: tuple[float, float] = (0.15, 0.65)
    patch_fraction: tuple[float, float] = (0.04, 0.16)


class PlatformGapDepthGhost:
    """Stateful gap-local depth corruption for a camera refresh tick."""

    def __init__(self, cfg: PlatformGapDepthGhostCfg | None = None):
        self.cfg = cfg or PlatformGapDepthGhostCfg()
        self.remaining = None
        self.center = None
        self.mode = None
        self.radius = None
        self.hole_depth = None
        self.return_offset = None

    def reset(self, env_ids=None):
        if self.remaining is None:
            return
        if env_ids is None:
            self.remaining.zero_()
        else:
            self.remaining[env_ids] = 0

    def _ensure_state(self, shape, device):
        n, cameras = shape[:2]
        if self.remaining is None or self.remaining.shape != (n, cameras):
            self.remaining = torch.zeros(n, cameras, device=device, dtype=torch.long)
            self.center = torch.zeros(n, cameras, 2, device=device)
            self.mode = torch.zeros(n, cameras, device=device, dtype=torch.long)
            self.radius = torch.ones(n, cameras, device=device)
            self.hole_depth = torch.full((n, cameras), self.cfg.max_depth, device=device)
            self.return_offset = torch.zeros(n, cameras, device=device)

    def __call__(self, frames, snapshot, origins, platform_gap_envs):
        """Return frames with depth corruption; inputs are ``N,C,2,H,W``.

        ``snapshot`` is the canonical tuple from ``canonical_camera_snapshot``:
        clean depth plus camera poses and calibrated intrinsics.  ``origins``
        are terrain-local environment origins and ``platform_gap_envs`` is a
        boolean family mask.
        """
        if frames.ndim != 5 or frames.shape[2] < 2:
            raise ValueError("frames must have shape [N,C,2,H,W]")
        _, clean_depth, positions, rotations, intrinsics = snapshot
        n, cameras, _, h, w = frames.shape
        if clean_depth.shape != (n, cameras, h, w):
            raise ValueError("snapshot depth and student frame shapes differ")
        self._ensure_state(frames.shape, frames.device)
        cfg = self.cfg
        active = platform_gap_envs.reshape(n).bool().to(frames.device)
        if not active.any():
            return frames

        points = depth_pixels_world(clean_depth.clamp(cfg.min_depth, cfg.max_depth), positions,
                                    rotations, intrinsics)
        local = points - origins[:, None, None, None, :]
        gap_x = (local[..., 0].abs() - cfg.center_offset).abs()
        eligible = (gap_x <= cfg.max_gap_width / 2 + cfg.margin)
        eligible &= local[..., 1].abs() <= cfg.tile_half_width
        eligible &= local[..., 2] >= -1.1
        eligible &= local[..., 2] <= 0.35
        eligible &= active[:, None, None, None]
        # The far sentinel represents no return, not a measured surface.
        eligible &= torch.isfinite(clean_depth) & (clean_depth >= cfg.min_depth) & (clean_depth < cfg.max_depth)

        has_eligible = eligible.flatten(2).any(-1)
        fresh = (self.remaining <= 0) & active[:, None] & has_eligible
        start = fresh & (torch.rand_like(self.remaining.float()) < cfg.probability)
        lo, hi = cfg.lifetime_steps
        sampled_life = torch.randint(lo, hi + 1, self.remaining.shape, device=frames.device)
        self.remaining = torch.where(start, sampled_life, self.remaining)
        self.mode = torch.where(start, torch.randint(0, 2, self.mode.shape, device=frames.device), self.mode)

        # Pick each new patch centre from that camera's eligible pixels. Existing
        # camera patches keep all parameters until their lifetime expires.
        random_score = torch.rand_like(clean_depth).masked_fill(~eligible, -1.0)
        centre_index = random_score.flatten(2).argmax(-1)
        sampled_x = (centre_index % w).to(frames.dtype)
        sampled_y = torch.div(centre_index, w, rounding_mode="floor").to(frames.dtype)
        self.center[..., 0] = torch.where(start, sampled_x, self.center[..., 0])
        self.center[..., 1] = torch.where(start, sampled_y, self.center[..., 1])
        fraction = _sample(cfg.patch_fraction, self.remaining.shape, frames.device)
        sampled_radius = (fraction.sqrt() * min(h, w) / 2).clamp_min(1.0)
        self.radius = torch.where(start, sampled_radius, self.radius)
        self.hole_depth = torch.where(
            start, _sample(cfg.false_hole_depth, self.remaining.shape, frames.device), self.hole_depth
        )
        self.return_offset = torch.where(
            start, _sample(cfg.false_return_offset, self.remaining.shape, frames.device), self.return_offset
        )

        active_patch = (self.remaining > 0) & active[:, None]
        ys = torch.arange(h, device=frames.device).view(1, 1, h, 1)
        xs = torch.arange(w, device=frames.device).view(1, 1, 1, w)
        patch = (xs - self.center[..., 0, None, None]).abs() <= self.radius[..., None, None]
        patch = patch & ((ys - self.center[..., 1, None, None]).abs() <= self.radius[..., None, None])
        mask = eligible & patch & active_patch[:, :, None, None]

        out = frames.clone()
        input_depth = frames[:, :, 0]
        # Direction is relative to the clean surface, not preceding generic noise.
        near = (clean_depth - self.return_offset[..., None, None]).clamp(cfg.min_depth, cfg.max_depth)
        far = torch.maximum(clean_depth + cfg.false_hole_min_offset, self.hole_depth[..., None, None])
        far = far.clamp(cfg.min_depth, cfg.max_depth)
        replacement = torch.where(self.mode[..., None, None] == 0, far, near)
        replacement_norm = ((replacement - cfg.min_depth) / (cfg.max_depth - cfg.min_depth)).clamp(0, 1)
        out[:, :, 0] = torch.where(mask, replacement_norm, input_depth)
        self.remaining = (self.remaining - 1).clamp_min(0)
        return out
