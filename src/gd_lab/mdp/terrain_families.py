"""Per-environment gates and weight scales keyed by terrain sub-terrain family.

Lets one RewTerm carry a different effective weight per terrain family - a
penalty tuned on flat ground is simply the wrong price on a stair.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_CACHE_ATTR = "_gd_family_cols"


def family_column_masks(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """``{family_name: column indices}`` for the scene terrain, cached on the env.

    Replicates ``TerrainGenerator._generate_curriculum_terrains`` exactly: a
    ``round(proportion * num_cols)`` approximation drifts whenever proportions
    are not clean ``1/num_cols`` multiples, mis-targeting every gate below.
    Empty on a terrain with no generator (play / plane).
    """
    cache = getattr(env, _CACHE_ATTR, None)
    if cache is not None:
        return cache
    gen = getattr(env.scene.terrain.cfg, "terrain_generator", None)
    if gen is None:
        return {}
    names = list(gen.sub_terrains.keys())
    total = sum(sc.proportion for sc in gen.sub_terrains.values())
    cum, acc = [], 0.0
    for sub in gen.sub_terrains.values():
        acc += sub.proportion / total
        cum.append(acc)
    cols: dict[str, list[int]] = {}
    for index in range(gen.num_cols):
        threshold = index / gen.num_cols + 0.001
        first = next(i for i, c in enumerate(cum) if threshold < c)
        cols.setdefault(names[first], []).append(index)
    cache = {name: torch.tensor(c, dtype=torch.long, device=env.device) for name, c in cols.items()}
    setattr(env, _CACHE_ATTR, cache)
    return cache


def _column_index(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Per-env sub-terrain column index, or ``None`` when terrain families are unavailable."""
    terrain_types = getattr(env.scene.terrain, "terrain_types", None)
    if terrain_types is None or not family_column_masks(env):
        return None
    return terrain_types


def terrain_family_gate(env: ManagerBasedRLEnv, families: Sequence[str]) -> torch.Tensor:
    """0.0 for envs currently on any of ``families``, 1.0 elsewhere (and on a
    terrain with no families, so play is unaffected)."""
    gate = torch.ones(env.num_envs, device=env.device)
    if not families:
        return gate
    terrain_types = _column_index(env)
    if terrain_types is None:
        return gate
    cache = family_column_masks(env)
    for family in families:
        cols = cache.get(family)
        if cols is not None:
            gate[torch.isin(terrain_types, cols)] = 0.0
    return gate


def terrain_family_scale(env: ManagerBasedRLEnv, family_scales: dict[str, float] | None) -> torch.Tensor:
    """Per-env weight multiplier: ``scale`` on a listed family, 1.0 everywhere else."""
    scale = torch.ones(env.num_envs, device=env.device)
    if not family_scales:
        return scale
    terrain_types = _column_index(env)
    if terrain_types is None:
        return scale
    cache = family_column_masks(env)
    for family, value in family_scales.items():
        cols = cache.get(family)
        if cols is not None:
            scale[torch.isin(terrain_types, cols)] = float(value)
    return scale
