"""Loss-specific absolute row-weight floors for UPDATE7.

The existing gap_weight still controls the original losses.
These options select a subset/objective and never multiply gap_weight.
"""
import math

import torch


def validate_weight(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 1:
        raise ValueError(f"{name} must be finite and >= 1")
    return value


def ahead_gap_rows(clean):
    """[N] bool: a known clean gap cell in the ahead corridor (rows 2:9, columns 9:17), visible or not."""
    if clean.ndim != 2 or clean.shape[1] != 4 * 187:
        raise ValueError("clean must be [N,748]")
    labels = clean.reshape(-1, 4, 11, 17)
    corridor_ahead = torch.zeros(11, 17, dtype=torch.bool, device=clean.device)
    corridor_ahead[2:9, 9:] = True
    return ((labels[:, 2] > .5) & labels[:, 3].bool() & corridor_ahead[None]).flatten(1).any(1)


def hidden_ahead_gap_rows(clean, teacher_terrain):
    if clean.ndim != 2 or clean.shape[1] != 4 * 187:
        raise ValueError("clean must be [N,748]")
    if teacher_terrain.shape != (len(clean), 2 * 187):
        raise ValueError("teacher_terrain must be [N,374]")
    labels = clean.reshape(-1, 4, 11, 17)
    terrain = teacher_terrain.reshape(-1, 2, 11, 17)
    gap = labels[:, 2] > .5
    known = labels[:, 3].bool()
    visible = terrain[:, 1] > .5
    corridor_ahead = torch.zeros(
        11, 17, dtype=torch.bool, device=clean.device
    )
    corridor_ahead[2:9, 9:] = True
    return (
        gap & known & ~visible & corridor_ahead[None]
    ).flatten(1).any(1)


def row_weight_floor(base_weight, selected_rows, requested_weight):
    requested_weight = validate_weight(requested_weight, "requested_weight")
    if base_weight.ndim != 1 or selected_rows.shape != base_weight.shape:
        raise ValueError("weights and row selection must have shape [N]")
    if not base_weight.is_floating_point():
        raise ValueError("base_weight must be floating-point")
    if (
        not torch.isfinite(base_weight).all()
        or (base_weight < 1).any()
    ):
        raise ValueError("base weights must be finite and >= 1")
    requested = base_weight.new_full(base_weight.shape, requested_weight)
    return torch.where(
        selected_rows.bool(),
        torch.maximum(base_weight, requested),
        base_weight,
    )


def objective_row_weights(
    base_weight, near_gap, clean, teacher_terrain,
    hidden_gap_latent_weight=1.0, gap_action_weight=1.0,
    ahead_gap_latent_weight=1.0,
):
    hidden = hidden_ahead_gap_rows(clean, teacher_terrain)
    latent_weight = row_weight_floor(
        base_weight, hidden, hidden_gap_latent_weight
    )
    # P1: the same floor rule for any ahead-corridor gap row, visible or hidden; 1 = unchanged.
    latent_weight = row_weight_floor(
        latent_weight, ahead_gap_rows(clean), ahead_gap_latent_weight
    )
    action_weight = row_weight_floor(
        base_weight, near_gap, gap_action_weight
    )
    return latent_weight, action_weight, hidden
