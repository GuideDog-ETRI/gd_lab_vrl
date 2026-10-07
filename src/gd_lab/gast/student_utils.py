"""Small tensor helpers for GAST student training."""

from __future__ import annotations

import torch


def finite_batch_rows(value: torch.Tensor) -> torch.Tensor:
    """Return one finite flag per batch row for rank-1 or higher tensors."""
    if value.ndim == 0:
        raise ValueError("expected a batch dimension")
    return torch.isfinite(value.reshape(value.shape[0], -1)).all(dim=1)
