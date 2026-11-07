#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for the backward-gap task.

1. Registers Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0 (gym entry points are strings; no Isaac import here).
2. Lets --warm_start_rayir accept the 3906 source task (Gap50) for this new task id; every other check
   (sha256, saved terrain levels, CENet optimizer, fixed schedule) is the original code path.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gd_lab.core import registry  # noqa: E402

TASK_ID = "Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastRayIrV21Gap50Back", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_backgap.tasks:GastRayIrV21Gap50BackTeacherCfg",
                              agent_cfg="gd_lab.gast.teacher:GastRayIrRunnerCfg") == TASK_ID
TASK_ID_V2 = "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastRayIrV21Gap50BackV2", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_backgap.tasks:GastRayIrV21Gap50BackV2TeacherCfg",
                              agent_cfg="gd_lab.gast.teacher:GastRayIrRunnerCfg") == TASK_ID_V2
TARGETS = (TASK_ID, TASK_ID_V2)

import gd_lab.gast.warm_start as warm_start  # noqa: E402

_original = warm_start.warm_start_from_rayir


def _from_3906(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    if expected_task not in TARGETS:
        raise ValueError(f"train_backgap.py runs only {TARGETS}, got {expected_task}")
    import hashlib
    import torch
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")
    if source not in TARGETS:  # 3906 (Gap50 task): the original code path, unchanged
        report = _original(alg, checkpoint_path, policy_lr, terrain_lr, expected_sha256=expected_sha256,
                           expected_task=SOURCE_TASK_ID)
        return dict(report, target_task=expected_task)
    # Continuation of this task (3000 -> 10000 updates): the same steps as warm_start_from_rayir, whose task
    # allow-list only names the two 3906 tasks. Weights, CENet optimizer and terrain levels restored; new PPO.
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("backgap checkpoint sha256 mismatch")
    if alg.schedule != "fixed":
        raise ValueError("backgap continuation requires fixed schedule")
    means = extra.get("terrain_level_means_by_family")
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("backgap continuation requires saved terrain levels and CENet optimizer")
    alg.policy.load_state_dict(c["model_state_dict"], strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    report = dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                  source_kind="backgap_continuation", ppo_optimizer="new", cenet_optimizer_restored=True,
                  latent_head_zeroed=False, learning_rates=warm_start.group_learning_rates(alg.optimizer),
                  terrain_level_means_by_family=means)
    return dict(report, target_task=expected_task)


warm_start.warm_start_from_rayir = _from_3906
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
