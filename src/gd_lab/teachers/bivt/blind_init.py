"""Warm-start a VRL actor-critic from a blind DreamWaQ checkpoint.

The VRL actor's first layer is the blind one widened by the terrain latent. The
new input columns start at zero, so at iteration 0 the VRL policy acts exactly
like the blind policy; the terrain encoder keeps its fresh initialization.
"""

from __future__ import annotations

import torch

ACTOR_INPUT_WEIGHT = "actor.0.weight"


def blind_to_vrl_state_dict(blind: dict[str, torch.Tensor], vrl: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return ``vrl`` with every blind tensor copied in and the latent columns zeroed."""
    missing = sorted(set(blind) - set(vrl))
    if missing:
        raise ValueError(f"blind checkpoint has tensors the VRL policy lacks: {missing}")
    merged = dict(vrl)
    for name, value in blind.items():
        target = vrl[name]
        if name == ACTOR_INPUT_WEIGHT:
            rows, cols = value.shape
            if target.shape[0] != rows or target.shape[1] <= cols:
                raise ValueError(f"{name}: cannot widen {tuple(value.shape)} to {tuple(target.shape)}")
            widened = torch.zeros_like(target)
            widened[:, :cols] = value.to(target)
            merged[name] = widened
        elif value.shape != target.shape:
            raise ValueError(f"{name}: blind {tuple(value.shape)} != VRL {tuple(target.shape)}")
        else:
            merged[name] = value.to(target)
    return merged


def load_blind_checkpoint(policy: torch.nn.Module, path: str) -> dict:
    """Load a blind ``model_*.pt`` into ``policy``; return the checkpoint's gd_lab infos."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(blind_to_vrl_state_dict(checkpoint["model_state_dict"], policy.state_dict()))
    return (checkpoint.get("infos") or {}).get("gd_lab", {})
