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


# UPDATE134B: continue a trained GAST v2.1 teacher as the Ray+IR teacher (3-channel history frames).
EXPECTED_GAST_SOURCE_TASK = "Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0"
FIRST_CONV = "terrain_encoder.cnn.0.weight"


def gast_to_rayir_state_dict(source: dict, target: dict) -> tuple[dict, dict]:
    """Every target tensor from the GAST teacher; the first Conv gains zero IR input columns.

    Strict both ways: no target-only and no source-only keys, no other shape change.
    """
    missing = sorted(set(target) - set(source))
    extra = sorted(set(source) - set(target))
    if missing or extra:
        raise ValueError(f"GAST -> Ray+IR key mismatch: target-only {missing}, source-only {extra}")
    merged, widened = {}, []
    for name, value in target.items():
        src = source[name].to(dtype=value.dtype)
        if name == FIRST_CONV and tuple(src.shape) != tuple(value.shape):
            out_ch, in_ch, kh, kw = src.shape
            if tuple(value.shape[:1] + value.shape[2:]) != (out_ch, kh, kw) or value.shape[1] <= in_ch:
                raise ValueError(f"{name}: cannot widen {tuple(src.shape)} to {tuple(value.shape)}")
            grown = torch.zeros_like(value)
            grown[:, :in_ch] = src
            merged[name] = grown
            widened.append(f"{name}: {tuple(src.shape)} -> {tuple(value.shape)} (new input channels zero)")
            continue
        if tuple(src.shape) != tuple(value.shape):
            raise ValueError(f"{name}: GAST {tuple(src.shape)} != Ray+IR {tuple(value.shape)}")
        merged[name] = src
    return merged, {"copied": len(merged), "widened": widened}


def warm_start_from_gast(alg, checkpoint_path: str, policy_lr: float, terrain_lr: float,
                         expected_sha256: str | None = None) -> dict:
    """Load a GAST v2.1 teacher into the Ray+IR policy; new PPO optimizer with two groups (iteration 0)."""
    if getattr(alg, "schedule", None) != "fixed":
        raise ValueError(f"per-group learning rates need agent.algorithm.schedule=fixed, got {alg.schedule!r}")
    with open(checkpoint_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.strip().lower():
        raise ValueError(f"GAST teacher sha256 {digest} != expected {expected_sha256}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (checkpoint.get("infos") or {}).get("gd_lab") or {}
    record = extra.get("gast_warm_start") or {}
    # A resumed GAST run (e.g. 26123, resumed from 8400) keeps the task under its warm-start origin.
    source_task = record.get("task") or (record.get("origin") or {}).get("task")
    if source_task != EXPECTED_GAST_SOURCE_TASK:
        raise ValueError(f"GAST source task {source_task!r} != {EXPECTED_GAST_SOURCE_TASK!r}")
    policy = alg.policy
    merged, report = gast_to_rayir_state_dict(checkpoint["model_state_dict"], policy.state_dict())
    policy.load_state_dict(merged, strict=True)
    cenet_state = extra.get("cenet_optimizer_state_dict")
    if cenet_state is not None:
        policy.cenet.optimizer.load_state_dict(cenet_state)
    alg.optimizer = build_grouped_optimizer(policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    return {"source": checkpoint_path, "source_sha256": digest, "source_sha256_expected": expected_sha256,
            "source_iteration": int(checkpoint.get("iter", -1)), "source_task": source_task,
            "source_kind": "gast_v21", "cenet_optimizer_restored": cenet_state is not None,
            "latent_head_zeroed": False, "ppo_optimizer": "new",
            "learning_rates": group_learning_rates(alg.optimizer), **report,
            "terrain_level_means_by_family": extra.get("terrain_level_means_by_family")}


# UPDATE135: a blind DreamWaQ teacher (DWB-38000) as the backbone of a fresh Ray+IR GAST teacher.
def warm_start_from_dwb(alg, checkpoint_path: str, policy_lr: float, terrain_lr: float,
                        expected_sha256: str | None = None) -> dict:
    """Copy the blind actor/critic/normalizers/std/CENet (actor terrain-latent columns zero); fresh terrain
    encoder/decoder with a zero latent head; new PPO and CENet optimizers (the blind loader restores none)."""
    from gd_lab.teachers.bivt.blind_init import blind_to_vrl_state_dict

    if getattr(alg, "schedule", None) != "fixed":
        raise ValueError(f"per-group learning rates need agent.algorithm.schedule=fixed, got {alg.schedule!r}")
    with open(checkpoint_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if expected_sha256 is None or digest != expected_sha256.strip().lower():
        raise ValueError(f"blind teacher sha256 {digest} != expected {expected_sha256}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    policy = alg.policy
    target = policy.state_dict()
    target_only = sorted(name for name in set(target) - set(source) if not is_terrain(name))
    if target_only:
        raise ValueError(f"Ray+IR tensors the blind teacher cannot fill: {target_only}")
    policy.load_state_dict(blind_to_vrl_state_dict(source, target), strict=True)
    zero_latent_head(policy)
    alg.optimizer = build_grouped_optimizer(policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    return {"source": checkpoint_path, "source_sha256": digest, "source_sha256_expected": expected_sha256,
            "source_iteration": int(checkpoint.get("iter", -1)), "source_kind": "blind_dreamwaq",
            "copied": len(source), "fresh_terrain": sum(is_terrain(name) for name in target),
            "cenet_optimizer_restored": False, "latent_head_zeroed": True, "ppo_optimizer": "new",
            "learning_rates": group_learning_rates(alg.optimizer), "terrain_level_means_by_family": None}


def warm_start_from_rayir(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    """UPDATE149: exact native Ray+IR weights, new PPO, restored CENet/terrain means.

    Intentionally a new 0-based 2000-update run, not additive optimizer resume.
    No widening or latent reset is permitted. A2769 and C2769 use this same path.
    """
    if alg.schedule != 'fixed':
        raise ValueError('Ray+IR fine-tune requires fixed schedule')
    with open(checkpoint_path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError('Ray+IR checkpoint sha256 mismatch')
    c = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    extra = (c.get('infos') or {}).get('gd_lab') or {}
    task = (extra.get('gast_warm_start') or {}).get('task')
    if task != expected_task or task not in ('Gd-GastRayIrV21-Rbq10-Dreamwaq-v0',
                                            'Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0'):
        raise ValueError(f'Ray+IR source task {task!r} does not match {expected_task!r}')
    means = extra.get('terrain_level_means_by_family')
    if not means or extra.get('cenet_optimizer_state_dict') is None:
        raise ValueError('Ray+IR fine-tune requires saved terrain levels and CENet optimizer')
    alg.policy.load_state_dict(c['model_state_dict'], strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra['cenet_optimizer_state_dict'])
    alg.optimizer = build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    return dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c['iter']),
                source_task=task, source_kind='native_rayir', ppo_optimizer='new',
                cenet_optimizer_restored=True, latent_head_zeroed=False,
                learning_rates=group_learning_rates(alg.optimizer), terrain_level_means_by_family=means)
