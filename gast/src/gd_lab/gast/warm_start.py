"""Warm-start a GAST teacher from a BIVT-Ray teacher checkpoint, with per-group PPO learning rates.

The two teachers share the DreamWaQ VRL backbone (``DreamwaqVrlActorCritic``): actor, critic, observation
normalizers, action std and CENet have identical shapes and are copied. Only the terrain encoder differs
(BIVT-Ray: single-frame ``HeightScanCNN``; GAST: ``TemporalTerrainEncoder`` + reconstruction decoder), so
the GAST encoder/decoder keep their fresh initialization -- except the encoder's last linear layer, which
starts at zero: the latent is then exactly 0, the input the BIVT-Ray actor already saw whenever no terrain
cell was visible (its gate), instead of a random code. The encoder still learns (non-zero inputs reach it).

PPO optimizer groups: everything except the terrain encoder/decoder at ``policy_lr`` (default 1e-4, as the
BIVT-Ray Clean gap run), the new terrain encoder/decoder at ``terrain_lr`` (default 1e-3). The schedule must
be ``fixed`` so rsl_rl never rewrites the group rates. CENet keeps its own optimizer (state copied).
"""

from __future__ import annotations

import hashlib

import torch

TERRAIN_PREFIXES = ("terrain_encoder.", "terrain_decoder.")
GROUP_KEY = "gd_lab_group"


def is_terrain(name: str) -> bool:
    return name.startswith(TERRAIN_PREFIXES)


def bivt_to_gast_state_dict(bivt: dict, gast: dict) -> tuple[dict, dict]:
    """Return (merged GAST state dict, report). Every non-terrain GAST tensor must come from BIVT-Ray."""
    merged, copied, fresh = dict(gast), [], []
    for name, target in gast.items():
        if is_terrain(name):
            fresh.append(name)
            continue
        if name not in bivt:
            raise ValueError(f"BIVT-Ray checkpoint lacks {name}; cannot warm-start the GAST backbone")
        value = bivt[name]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"{name}: BIVT-Ray {tuple(value.shape)} != GAST {tuple(target.shape)}")
        merged[name] = value.to(dtype=target.dtype)
        copied.append(name)
    dropped = sorted(name for name in bivt if name not in gast or is_terrain(name))
    unexpected = [name for name in dropped if not is_terrain(name)]
    if unexpected:
        raise ValueError(f"BIVT-Ray tensors with no GAST counterpart: {unexpected}")
    return merged, {"copied": len(copied), "fresh_terrain": len(fresh), "dropped_bivt_terrain": len(dropped)}


def zero_latent_head(policy: torch.nn.Module) -> None:
    head = policy.terrain_encoder.head[0]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()


def build_grouped_optimizer(policy: torch.nn.Module, policy_lr: float, terrain_lr: float,
                            template: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Same optimizer class/hyper-parameters as ``template``, two learning-rate groups."""
    backbone, terrain = [], []
    for name, parameter in policy.named_parameters():
        if name.startswith("cenet."):
            continue  # CENet has its own optimizer
        (terrain if is_terrain(name) else backbone).append(parameter)
    if not backbone or not terrain:
        raise ValueError("expected both backbone and terrain parameters")
    defaults = {k: v for k, v in template.defaults.items() if k != "lr"}
    groups = [{"params": backbone, "lr": float(policy_lr), GROUP_KEY: "policy"},
              {"params": terrain, "lr": float(terrain_lr), GROUP_KEY: "terrain"}]
    return type(template)(groups, **defaults)


def group_learning_rates(optimizer: torch.optim.Optimizer) -> dict:
    return {group.get(GROUP_KEY, f"group{i}"): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def is_grouped(optimizer: torch.optim.Optimizer) -> bool:
    return any(GROUP_KEY in group for group in optimizer.param_groups)


def warm_start_from_bivt(alg, checkpoint_path: str, policy_lr: float, terrain_lr: float) -> dict:
    """Load BIVT-Ray weights into ``alg.policy``, rebuild ``alg.optimizer`` with two groups; return a manifest."""
    if getattr(alg, "schedule", None) != "fixed":
        raise ValueError(f"per-group learning rates need agent.algorithm.schedule=fixed, got {alg.schedule!r}")
    with open(checkpoint_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = alg.policy
    merged, report = bivt_to_gast_state_dict(checkpoint["model_state_dict"], policy.state_dict())
    policy.load_state_dict(merged)
    zero_latent_head(policy)
    extra = (checkpoint.get("infos") or {}).get("gd_lab") or {}
    cenet_state = extra.get("cenet_optimizer_state_dict")
    if cenet_state is not None:
        policy.cenet.optimizer.load_state_dict(cenet_state)
    alg.optimizer = build_grouped_optimizer(policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)  # what rsl_rl logs; the terrain group keeps its own rate
    gap = extra.get("gap_finetune") or {}
    return {"source": checkpoint_path, "source_sha256": digest, "source_iteration": int(checkpoint.get("iter", -1)),
            "source_task": gap.get("task"), "source_camera_profile": gap.get("camera_profile"),
            "cenet_optimizer_restored": cenet_state is not None, "latent_head_zeroed": True,
            "learning_rates": group_learning_rates(alg.optimizer), **report,
            "terrain_level_means_by_family": extra.get("terrain_level_means_by_family")}
