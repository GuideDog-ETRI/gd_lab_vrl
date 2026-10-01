"""Pure tensor kernels for gap-only shaping and one-shot crossing detection."""

import torch


def boarding_support_height(ray_z: torch.Tensor, fallback: torch.Tensor, max_step: float = 0.16):
    """Mean of deck returns only; exclude the deep pit, handle all-missing rays."""
    valid = torch.isfinite(ray_z)
    top = torch.where(valid, ray_z, -torch.inf).amax(dim=1)
    deck = valid & (ray_z >= top[:, None] - max_step - 0.04)
    safe = torch.where(deck, ray_z, 0.0)
    mean = safe.sum(dim=1) / deck.sum(dim=1).clamp_min(1)
    return torch.where(deck.any(dim=1), mean, fallback)


def boarding_drop_cost(foot_z: torch.Tensor, support: torch.Tensor, allowance: float = 0.22):
    """Bounded cost: a legal 16 cm down-step is not a dropped foot."""
    return ((support[:, None] - foot_z - allowance) / 0.25).clamp(0, 1).mean(dim=1)


def boarding_crossing_candidates(feet_local: torch.Tensor, boundary: float, half_width: float):
    """[N, 2] all four feet beyond the left/right gap, inside the own tile.

    boundary includes the maximum gap half-width and foot-radius margin;
    this conservative completion criterion is valid at every difficulty.
    """
    in_tile = (feet_local[..., 1].abs() < half_width - 0.15).all(dim=1)
    return torch.stack(
        ((feet_local[..., 0] < -boundary).all(dim=1), (feet_local[..., 0] > boundary).all(dim=1)), dim=1
    ) & in_tile[:, None]
