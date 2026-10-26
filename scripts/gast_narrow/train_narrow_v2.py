#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for GAST-Narrow v2 (warm start: GAST-Narrow v1 update 1748).

1. Registers Gd-GastNarrowV2-Rbq10-Dreamwaq-v0 (string entry points; no Isaac import here).
2. --warm_start_rayir accepts the v1 1748 checkpoint (task Gd-GastNarrowV1): every tensor is copied, the LiDAR
   encoder included (same [2, 30, 48] map); the actor first layer gains 1 zero input column (margin), the critic 3
   (privileged margin / blocked / distance to the block line), so the policy equals 1748 at start. New PPO
   optimizer (LiDAR encoder in the terrain group), CENet optimizer and terrain levels restored (dead end: level 0).
3. Online Top-5 uses the v2 narrow score (gd_lab.gast_narrow.narrow_top5_v2).
"""
import os
import runpy
import sys
from pathlib import Path

import torch

# Each rank on its own GPU before Isaac starts (as gast_nav/train.py).
if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gd_lab.core import registry  # noqa: E402

TASK_ID = "Gd-GastNarrowV2-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV1-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastNarrowV2", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_narrow.tasks_v2:GastNarrowV2TeacherCfg",
                              agent_cfg="gd_lab.gast_narrow.model:GastNarrowV2RunnerCfg") == TASK_ID
# The source tasks must be registered too: their config classes are the parents (imported, not run).
registry.register_task(task="GastRayIrV21Gap50BackV2", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_backgap.tasks:GastRayIrV21Gap50BackV2TeacherCfg",
                       agent_cfg="gd_lab.gast.teacher:GastRayIrRunnerCfg")
registry.register_task(task="GastNarrowV1", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks:GastNarrowV1TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.model:GastNarrowRunnerCfg")

import gd_lab.gast.warm_start as warm_start  # noqa: E402

# The LiDAR encoder joins the "terrain" learning-rate group (check_resumed_rates allows exactly two groups).
warm_start.TERRAIN_PREFIXES = warm_start.TERRAIN_PREFIXES + ("lidar_encoder.",)
WIDENED = ("actor.0.weight", "critic.0.weight")


def _from_1748(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    import hashlib
    import torch
    from gd_lab.gast_narrow.tasks_v2 import NARROW_FAMILIES
    if expected_task != TASK_ID:
        raise ValueError(f"train_narrow_v2.py runs only {TASK_ID}, got {expected_task}")
    if alg.schedule != "fixed":
        raise ValueError("narrow fine-tune requires fixed schedule")
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("1748 checkpoint sha256 mismatch")
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")
    if source != SOURCE_TASK_ID:
        raise ValueError(f"source task {source!r} != {SOURCE_TASK_ID!r}")
    means = dict(extra.get("terrain_level_means_by_family") or {})
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("1748 checkpoint lacks terrain levels or CENet optimizer")
    src, target = c["model_state_dict"], alg.policy.state_dict()
    source_only = sorted(set(src) - set(target))
    if source_only:
        raise ValueError(f"1748 tensors with no v2 counterpart: {source_only}")
    merged, fresh, widened = {}, [], []
    for name, value in target.items():
        if name not in src:
            raise ValueError(f"1748 checkpoint lacks {name}")
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
            raise ValueError(f"{name}: 1748 {tuple(s.shape)} != v2 {tuple(value.shape)}")
        merged[name] = s
    alg.policy.load_state_dict(merged, strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    for family in NARROW_FAMILIES:
        means.setdefault(family, 0.0)
    return dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                source_kind="narrow_v1_1748", narrow_version="v2", target_task=expected_task, ppo_optimizer="new",
                cenet_optimizer_restored=True, copied=len(merged) - len(fresh), fresh=len(fresh),
                widened=widened, learning_rates=warm_start.group_learning_rates(alg.optimizer),
                terrain_level_means_by_family=means)


warm_start.warm_start_from_rayir = _from_1748

from gd_lab.gast_narrow import narrow_top5_v2  # noqa: E402

narrow_top5_v2.install()
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
