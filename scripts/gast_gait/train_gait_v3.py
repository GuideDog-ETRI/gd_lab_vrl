#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for the gait quality v3 gap task (docs/design/GAIT_V3_DESIGN.md).

1. Registers Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0 (gym entry points are strings; no Isaac import here).
2. --warm_start_rayir accepts the 3906 source task (Gap50, the original code path) or a continuation of a backgap /
   gait v3 checkpoint (same steps as scripts/gast_backgap/train_backgap.py: weights, CENet optimizer and terrain
   levels restored, new PPO). Every other check (sha256, fixed schedule) is the original code path.
3. Multi-node set-up of scripts/gast_narrow/train_lineage_a.py (rank GPU before Isaac, long process-group timeout).
Launch through scripts/gast_narrow/train_narrow_v23.sh with NARROW_TRAIN_PY=scripts/gast_gait/train_gait_v3.py
(scripts/gast_gait/train_gait_v3_3gpu.sh sets everything).
"""
import faulthandler
import hashlib
import os
import runpy
import signal
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gd_lab.core import registry  # noqa: E402

if os.environ.get("GD_LAB_NORMALIZER_DEFER", "0") == "1":
    raise SystemExit("gait v3 needs GD_LAB_NORMALIZER_DEFER=0 (no anchor runner to flush the normalizer)")
if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
faulthandler.register(signal.SIGUSR1, all_threads=True)

_init_pg = torch.distributed.init_process_group


def _init_pg_long(*args, **kwargs):
    import datetime
    kwargs.setdefault("timeout", datetime.timedelta(minutes=float(os.environ.get("NARROW_PG_TIMEOUT_MIN", "60"))))
    return _init_pg(*args, **kwargs)


torch.distributed.init_process_group = _init_pg_long

TASK_ID = "Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastGaitV3Gap", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_gait.tasks:GastGaitV3GapTeacherCfg",
                              agent_cfg="gd_lab.gast.teacher:GastRayIrRunnerCfg") == TASK_ID
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0"  # 3906
CONTINUATION_TASKS = ("Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0", "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0",
                      TASK_ID)

import gd_lab.gast.warm_start as warm_start  # noqa: E402

_original = warm_start.warm_start_from_rayir


def _from_3906_or_continuation(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    if expected_task != TASK_ID:
        raise ValueError(f"train_gait_v3.py runs only {TASK_ID}, got {expected_task}")
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")
    if source not in CONTINUATION_TASKS:  # 3906 (Gap50 task): the original code path, unchanged
        report = _original(alg, checkpoint_path, policy_lr, terrain_lr, expected_sha256=expected_sha256,
                           expected_task=SOURCE_TASK_ID)
        return dict(report, target_task=expected_task)
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("gait v3 source checkpoint sha256 mismatch")
    if alg.schedule != "fixed":
        raise ValueError("gait v3 continuation requires fixed schedule")
    means = extra.get("terrain_level_means_by_family")
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("gait v3 continuation requires saved terrain levels and CENet optimizer")
    alg.policy.load_state_dict(c["model_state_dict"], strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    report = dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                  source_kind="gait_v3_continuation", ppo_optimizer="new", cenet_optimizer_restored=True,
                  latent_head_zeroed=False, learning_rates=warm_start.group_learning_rates(alg.optimizer),
                  terrain_level_means_by_family=means)
    return dict(report, target_task=expected_task)


warm_start.warm_start_from_rayir = _from_3906_or_continuation
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
