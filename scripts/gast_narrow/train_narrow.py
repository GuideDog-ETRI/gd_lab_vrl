#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for GAST-Narrow v1 (warm start: backgap V2 update 6566).

1. Registers Gd-GastNarrowV1-Rbq10-Dreamwaq-v0 (string entry points; no Isaac import here).
2. --warm_start_rayir accepts the 6566 checkpoint (task Gd-GastRayIrV21Gap50BackV2): every 6566 tensor is copied;
   the new LiDAR encoder is fresh; the actor/critic first layers gain zero input columns (policy = 6566 at start).
   New PPO optimizer (LiDAR encoder in the terrain group), CENet optimizer and retention terrain levels restored,
   narrow families start at level 0 (1.20 m).
3. Online Top-5 uses the narrow score (gd_lab.gast_narrow.narrow_top5).
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gd_lab.core import registry  # noqa: E402

TASK_ID = "Gd-GastNarrowV1-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastNarrowV1", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_narrow.tasks:GastNarrowV1TeacherCfg",
                              agent_cfg="gd_lab.gast_narrow.model:GastNarrowRunnerCfg") == TASK_ID
# The 6566 task id must be registered too: its config class is the parent (imported, not run).
registry.register_task(task="GastRayIrV21Gap50BackV2", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_backgap.tasks:GastRayIrV21Gap50BackV2TeacherCfg",
                       agent_cfg="gd_lab.gast.teacher:GastRayIrRunnerCfg")

import gd_lab.gast.warm_start as warm_start  # noqa: E402

# The LiDAR encoder joins the "terrain" learning-rate group (check_resumed_rates allows exactly two groups).
warm_start.TERRAIN_PREFIXES = warm_start.TERRAIN_PREFIXES + ("lidar_encoder.",)
WIDENED = ("actor.0.weight", "critic.0.weight")


def _from_6566(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    import hashlib
    import torch
    from gd_lab.gast_narrow.tasks import NARROW_FAMILIES
    if expected_task != TASK_ID:
        raise ValueError(f"train_narrow.py runs only {TASK_ID}, got {expected_task}")
    if alg.schedule != "fixed":
        raise ValueError("narrow fine-tune requires fixed schedule")
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("6566 checkpoint sha256 mismatch")
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")
    if source != SOURCE_TASK_ID:
        raise ValueError(f"source task {source!r} != {SOURCE_TASK_ID!r}")
    means = dict(extra.get("terrain_level_means_by_family") or {})
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("6566 checkpoint lacks terrain levels or CENet optimizer")
    src, target = c["model_state_dict"], alg.policy.state_dict()
    source_only = sorted(set(src) - set(target))
    if source_only:
        raise ValueError(f"6566 tensors with no narrow counterpart: {source_only}")
    merged, fresh, widened = {}, [], []
    for name, value in target.items():
        if name.startswith("lidar_encoder."):
            merged[name] = value
            fresh.append(name)
            continue
        if name not in src:
            raise ValueError(f"6566 checkpoint lacks {name}")
        s = src[name].to(dtype=value.dtype)
        if name in WIDENED:
            if s.shape[0] != value.shape[0] or s.shape[1] >= value.shape[1]:
                raise ValueError(f"{name}: cannot widen {tuple(s.shape)} to {tuple(value.shape)}")
            grown = torch.zeros_like(value)
            grown[:, :s.shape[1]] = s
            merged[name] = grown
            widened.append(f"{name}: {tuple(s.shape)} -> {tuple(value.shape)} (new columns zero)")
            continue
        if tuple(s.shape) != tuple(value.shape):
            raise ValueError(f"{name}: 6566 {tuple(s.shape)} != narrow {tuple(value.shape)}")
        merged[name] = s
    alg.policy.load_state_dict(merged, strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    for family in NARROW_FAMILIES:
        means.setdefault(family, 0.0)
    return dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                source_kind="backgap_v2_6566", target_task=expected_task, ppo_optimizer="new",
                cenet_optimizer_restored=True, copied=len(merged) - len(fresh), fresh_lidar=len(fresh),
                widened=widened, learning_rates=warm_start.group_learning_rates(alg.optimizer),
                terrain_level_means_by_family=means)


warm_start.warm_start_from_rayir = _from_6566

from gd_lab.gast_narrow import narrow_top5  # noqa: E402

narrow_top5.install()
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
