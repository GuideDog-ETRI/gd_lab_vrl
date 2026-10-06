"""Common v2 student-distillation pieces for RVLD/GAVD (scripts/distill_student.py); GAST has the same in gd_lab.gast.

- near-gap rows: a privileged height-scan ray of a gap-family tile hits more than 0.3 m below the tile origin
  (the same label as GAST's semantic gap map, gd_lab.gast.observations.clean_terrain), i.e. a gap lies inside
  the 1.6 x 1.0 m body grid;
- row-weighted means so near-gap samples weigh ``gap_loss_weight`` (1 reproduces the plain means exactly).
"""

from __future__ import annotations

import torch

GAP_DEPTH_BELOW_ORIGIN = 0.3


def near_gap_from_scan(ray_z: torch.Tensor, origin_z: torch.Tensor, gap_family: torch.Tensor) -> torch.Tensor:
    """[N] bool from ray hit heights [N, R], tile origin heights [N] and the gap-family mask [N]."""
    deep = torch.isfinite(ray_z) & (ray_z < origin_z[:, None] - GAP_DEPTH_BELOW_ORIGIN)
    return gap_family & deep.any(-1)


def near_gap_envs(env) -> torch.Tensor:
    """Privileged near-gap flag of every env at the current step (teacher-side label, never a student input)."""
    from gd_lab.mdp.terrain_families import terrain_family_gate

    ray_z = env.scene["height_scanner"].data.ray_hits_w[..., 2]
    family = terrain_family_gate(env, ("platform_gap", "gap_field")) == 0
    return near_gap_from_scan(ray_z, env.scene.env_origins[:, 2], family)


def row_weights(near_gap: torch.Tensor, gap_loss_weight: float) -> torch.Tensor:
    if gap_loss_weight < 1:
        raise ValueError("gap_loss_weight must be >= 1 (1 = no near-gap emphasis)")
    return 1 + (float(gap_loss_weight) - 1) * near_gap.float()


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-row values [B] averaged with row weights; equals values.mean() when every weight is 1."""
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def row_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """[B] mean squared error per row (weighted_mean(row_mse(a, b), 1) == F.mse_loss(a, b))."""
    return (prediction - target).pow(2).reshape(prediction.shape[0], -1).mean(-1)
