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
# The only BIVT-Ray teachers a GAST Clean-gap teacher may start from (checked, never assumed).
EXPECTED_SOURCE_TASK = "Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0"
# A BIVT-Ray teacher trained with the v2 gap/stair terms (same observations) is an equally valid source.
ACCEPTED_SOURCE_TASKS = (EXPECTED_SOURCE_TASK, "Gd-VrlGapFinetuneCleanV2Raycast-Rbq10-Dreamwaq-v0",
                         "Gd-VrlGapFinetuneCleanV21Raycast-Rbq10-Dreamwaq-v0")
EXPECTED_SOURCE_CAMERA = "vendor_new"
EXPECTED_OBSERVATION_VERSION = "bivt_ray_occlusion_v2"


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


def validate_source(checkpoint: dict, digest: str, expected_sha256: str | None = None) -> None:
    """Refuse a source teacher whose identity or observation/camera contract is not the expected one.

    Shapes alone cannot tell a legacy-camera or other-observation BIVT-Ray teacher from the right one.
    A missing record is a refusal too: nothing is inferred.
    """
    if expected_sha256 is not None and digest != expected_sha256.strip().lower():
        raise ValueError(f"BIVT-Ray teacher sha256 {digest} != expected {expected_sha256}")
    extra = (checkpoint.get("infos") or {}).get("gd_lab") or {}
    gap = extra.get("gap_finetune") or {}
    recorded = {"task": gap.get("task"), "camera_profile": gap.get("camera_profile"),
                "observation_version": (extra.get("observation_context") or {}).get("version")}
    expected = {"task": ACCEPTED_SOURCE_TASKS, "camera_profile": (EXPECTED_SOURCE_CAMERA,),
                "observation_version": (EXPECTED_OBSERVATION_VERSION,)}
    wrong = {k: recorded[k] for k in expected if recorded[k] not in expected[k]}
    if wrong:
        raise ValueError(f"BIVT-Ray teacher contract mismatch or missing record: {wrong}, expected {expected}")


def check_resumed_rates(optimizer: torch.optim.Optimizer, policy_lr: float, terrain_lr: float) -> dict:
    """After a resume: the restored group rates must be the requested ones (fixed-rate contract)."""
    actual = group_learning_rates(optimizer)
    requested = {"policy": float(policy_lr), "terrain": float(terrain_lr)}
    if set(actual) != set(requested) or any(abs(actual[k] - v) > 1e-12 * max(1.0, abs(v)) for k, v in requested.items()):
        raise ValueError(f"restored learning rates {actual} differ from the requested {requested}")
    return actual


def resumed_provenance(checkpoint_path: str) -> dict:
    """The original warm-start record carried by a GAST checkpoint (kept across resumes)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    record = ((checkpoint.get("infos") or {}).get("gd_lab") or {}).get("gast_warm_start")
    if not record:
        raise ValueError(f"{checkpoint_path} carries no gast_warm_start record; not a warm-started GAST run")
    origin = record.get("origin") or {k: v for k, v in record.items() if k != "resumed_from"}
    return {"origin": origin, "resumed_from": {"checkpoint": checkpoint_path, "iteration": int(checkpoint.get("iter", -1))}}


def warm_start_from_bivt(alg, checkpoint_path: str, policy_lr: float, terrain_lr: float,
                         expected_sha256: str | None = None) -> dict:
    """Load BIVT-Ray weights into ``alg.policy``, rebuild ``alg.optimizer`` with two groups; return a manifest."""
    if getattr(alg, "schedule", None) != "fixed":
        raise ValueError(f"per-group learning rates need agent.algorithm.schedule=fixed, got {alg.schedule!r}")
    with open(checkpoint_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_source(checkpoint, digest, expected_sha256)
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
    return {"source": checkpoint_path, "source_sha256": digest, "source_sha256_expected": expected_sha256, "source_iteration": int(checkpoint.get("iter", -1)),
            "source_task": gap.get("task"), "source_camera_profile": gap.get("camera_profile"),
            "cenet_optimizer_restored": cenet_state is not None, "latent_head_zeroed": True,
            "learning_rates": group_learning_rates(alg.optimizer), **report,
            "terrain_level_means_by_family": extra.get("terrain_level_means_by_family")}
