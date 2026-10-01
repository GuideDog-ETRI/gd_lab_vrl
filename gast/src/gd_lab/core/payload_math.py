"""Mass properties of a rigidly attached point payload. Pure torch, no simulator."""

from __future__ import annotations

import torch


def combine_point_payload(
    base_mass: torch.Tensor,
    base_com: torch.Tensor,
    base_inertia: torch.Tensor,
    payload_mass: torch.Tensor,
    payload_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined mass, CoM and inertia about the combined CoM.

    Positions and inertia tensors use the body's link frame. Masses have shape
    ``(...,)``, positions ``(..., 3)`` and inertias ``(..., 3, 3)``; batch dimensions
    may broadcast. The payload is a point mass (no intrinsic inertia or collider).
    ``base_inertia`` is about the original CoM, not about the link origin.
    """
    mass = base_mass + payload_mass
    displacement = payload_position - base_com
    com = base_com + (payload_mass / mass).unsqueeze(-1) * displacement

    # Parallel-axis theorem for both constituents, about their combined CoM.
    reduced_mass = base_mass * payload_mass / mass
    identity = torch.eye(3, dtype=base_inertia.dtype, device=base_inertia.device)
    shift = displacement.square().sum(dim=-1)[..., None, None] * identity
    shift = shift - displacement.unsqueeze(-1) * displacement.unsqueeze(-2)
    inertia = base_inertia + reduced_mass[..., None, None] * shift
    return mass, com, inertia
