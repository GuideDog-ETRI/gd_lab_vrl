"""Helpers for persisting and restoring curriculum terrain levels on resume."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def capture_family_level_means(terrain, family_columns: dict[str, list[int]]) -> dict[str, float]:
    """Return the current mean terrain level for each named sub-terrain family."""
    levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if levels is None or terrain_types is None:
        return {}
    levels = levels.float()
    means: dict[str, float] = {}
    for family, columns in family_columns.items():
        column_ids = torch.as_tensor(columns, device=terrain_types.device, dtype=terrain_types.dtype)
        mask = torch.isin(terrain_types, column_ids)
        if bool(mask.any()):
            means[family] = float(levels[mask].mean().item())
    return means


def restore_family_start_levels(terrain, env_origins, family_columns, means):
    """Set each family's environments to floor(saved mean) and align their origins."""
    levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    origins = getattr(terrain, "terrain_origins", None)
    if levels is None or terrain_types is None or origins is None:
        return {}
    max_level = int(origins.shape[0]) - 1
    restored = {}
    terrain_env_origins = getattr(terrain, "env_origins", None)
    for family, columns in family_columns.items():
        if family not in means:
            continue
        target = min(max(math.floor(float(means[family])), 0), max_level)
        column_ids = torch.as_tensor(columns, device=terrain_types.device, dtype=terrain_types.dtype)
        mask = torch.isin(terrain_types, column_ids)
        if not bool(mask.any()):
            continue
        levels[mask] = target
        selected_origins = origins[target, terrain_types[mask]]
        env_origins[mask] = selected_origins
        if terrain_env_origins is not None and terrain_env_origins.data_ptr() != env_origins.data_ptr():
            terrain_env_origins[mask] = selected_origins
        restored[family] = target
    return restored


def read_family_level_means_from_tensorboard(checkpoint_path, checkpoint_iteration, families):
    """Find latest family curriculum scalars at/before a legacy checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    candidate_dirs = [checkpoint_path.parent, *list(checkpoint_path.parents)[1:3]]
    event_files = sorted({p for directory in candidate_dirs for p in directory.glob("events.out.tfevents.*")})
    if not event_files:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        found = {}
        for event_file in event_files:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
            for family in families:
                tag = f"Curriculum/terrain_levels_per_family/{family}"
                if tag not in scalar_tags:
                    continue
                eligible = [event for event in accumulator.Scalars(tag) if event.step <= checkpoint_iteration]
                if eligible and (family not in found or eligible[-1].step > found[family][0]):
                    found[family] = (eligible[-1].step, float(eligible[-1].value))
        return {family: value for family, (_, value) in found.items()}
    except Exception:
        return {}
