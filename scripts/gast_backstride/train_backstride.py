#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for the backward gap stride task (gd_lab.gast_backstride).

Continuation only: the source is an A' (Gd-GastGaitV3Gap) or backstride checkpoint -- weights, CENet optimizer and
terrain levels restored, new PPO (the gait v3 continuation path of scripts/gast_gait/train_gait_v3.py).
Launch through scripts/gast_narrow/train_narrow_v23.sh (scripts/gast_backstride/train_backstride_3gpu.sh).
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

if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
faulthandler.register(signal.SIGUSR1, all_threads=True)

_init_pg = torch.distributed.init_process_group


def _init_pg_long(*args, **kwargs):
    import datetime
    kwargs.setdefault("timeout", datetime.timedelta(minutes=60))
    return _init_pg(*args, **kwargs)


torch.distributed.init_process_group = _init_pg_long

TASK_ID = "Gd-GastGaitV3BackStride-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastGaitV3BackStride", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_backstride.tasks:GastGaitV3BackStrideTeacherCfg",
                              agent_cfg="gd_lab.gast_backstride.tasks:GastGaitV3BackStrideRunnerCfg") == TASK_ID
CONTINUATION_TASKS = ("Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0", TASK_ID)

import gd_lab.gast.warm_start as warm_start  # noqa: E402


def _continuation(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    if expected_task != TASK_ID:
        raise ValueError(f"train_backstride.py runs only {TASK_ID}, got {expected_task}")
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")
    if source not in CONTINUATION_TASKS:
        raise ValueError(f"backstride starts from an A' / backstride checkpoint, got source task {source}")
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("backstride source checkpoint sha256 mismatch")
    if alg.schedule != "fixed":
        raise ValueError("backstride continuation requires fixed schedule")
    means = extra.get("terrain_level_means_by_family")
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("backstride continuation requires saved terrain levels and CENet optimizer")
    alg.policy.load_state_dict(c["model_state_dict"], strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    report = dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                  source_kind="backstride_continuation", ppo_optimizer="new", cenet_optimizer_restored=True,
                  latent_head_zeroed=False, learning_rates=warm_start.group_learning_rates(alg.optimizer),
                  terrain_level_means_by_family=means)
    return dict(report, target_task=expected_task)


warm_start.warm_start_from_rayir = _continuation
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
