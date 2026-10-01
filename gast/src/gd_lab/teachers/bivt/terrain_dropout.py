"""Terrain-group blackouts, so the policy learns to walk blind when the camera
stream disappears.

Both terms keep the VRL layout [height(187), validity(187)]. During a blackout an
environment's whole group is zero, i.e. every cell invalid, which is exactly
what a missing camera stream looks like to the terrain encoder.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase

from gd_lab.mdp.camera_observations import CameraVisibleTerrain

from .blackout import BlackoutSchedule

SCAN_CELLS = 187


def full_visible_height(env) -> torch.Tensor:
    """Every scan cell visible; same height offset/clip/scale as CameraVisibleTerrain."""
    scan = env.scene["height_scanner"].data
    points = scan.ray_hits_w
    if points.shape[1] != SCAN_CELLS:
        raise ValueError("VRL v2 requires the configured 11x17 height scan")
    height = (scan.pos_w[:, 2, None] - points[..., 2] - 0.5).clamp(-1, 1) * 5
    valid = torch.isfinite(height)
    return torch.cat((torch.where(valid, height, 0), valid.float()), -1)


def _schedule(cfg, env) -> BlackoutSchedule:
    params = cfg.params
    return BlackoutSchedule(
        env.num_envs, env.device, params.get("start_prob", 0.0),
        tuple(params.get("duration_steps", (50, 300))), params.get("episode_prob", 0.0),
    )


class FullVisibleTerrainDropout(ManagerTermBase):
    """Camera-free stage 2-A terrain group with blackouts."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.blackout = _schedule(cfg, env)
        self.blackout.reset()

    def reset(self, env_ids=None):
        self.blackout.reset(env_ids)

    def __call__(self, env, start_prob=0.0, duration_steps=(50, 300), episode_prob=0.0) -> torch.Tensor:
        terrain = full_visible_height(env)
        return torch.where(self.blackout.mask(env.common_step_counter)[:, None], 0.0, terrain)


class CameraVisibleTerrainDropout(CameraVisibleTerrain):
    """Camera-visible stage 2-B terrain group with the same blackouts."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.blackout = _schedule(cfg, env)
        self.blackout.reset()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.blackout.reset(env_ids)

    def __call__(self, env, start_prob=0.0, duration_steps=(50, 300), episode_prob=0.0) -> torch.Tensor:
        terrain = super().__call__(env)
        return torch.where(self.blackout.mask(env.common_step_counter)[:, None], 0.0, terrain)
