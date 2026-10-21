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

import gd_lab.gast.warm_start as warm_start  # noqa: E402

_original = warm_start.warm_start_from_rayir


def _from_3906(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    if expected_task != TASK_ID:
        raise ValueError(f"train_backgap.py runs only {TASK_ID}, got {expected_task}")
    report = _original(alg, checkpoint_path, policy_lr, terrain_lr, expected_sha256=expected_sha256,
                       expected_task=SOURCE_TASK_ID)
    return dict(report, target_task=TASK_ID)


warm_start.warm_start_from_rayir = _from_3906
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
