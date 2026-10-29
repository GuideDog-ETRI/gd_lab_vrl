#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for GAST-Narrow v2.3 (warm start: GAST-Narrow v2.2 update 371).

1. Registers Gd-GastNarrowV23-Rbq10-Dreamwaq-v0 (v2.2 + 44-column terrain, safe speed cap, shin clearance, leg touch).
2. --warm_start_rayir accepts the v2.2 371 checkpoint (task Gd-GastNarrowV22): every tensor is copied unchanged (same
   shapes; only the column counts change); new PPO optimizer, CENet optimizer and terrain levels (by family) restored.
3. Online Top-5 uses the v2 narrow score (gd_lab.gast_narrow.narrow_top5_v2).
The 6566 anchor checkpoint comes from NARROW_ANCHOR_CKPT (launcher).
"""
import faulthandler
import os
import runpy
import signal
import sys
from pathlib import Path

import torch

# Each rank on its own GPU before Isaac starts (as gast_nav/train.py).
if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# `kill -USR1 <pid>` prints every thread's Python stack to stderr (diagnosing distributed hangs).
faulthandler.register(signal.SIGUSR1, all_threads=True)
# NARROW_STACK_DUMP_S=t: these ranks (NARROW_STACK_DUMP_RANKS, default 0) print all Python stacks every t seconds
# (Kit takes SIGUSR1 and py-spy has no ptrace on the compute nodes).
_dump_ranks = os.environ.get("NARROW_STACK_DUMP_RANKS", "0").split(",")
if os.environ.get("NARROW_STACK_DUMP_S") and ("all" in _dump_ranks or os.environ.get("RANK", "0") in _dump_ranks):
    faulthandler.dump_traceback_later(float(os.environ["NARROW_STACK_DUMP_S"]), repeat=True)

# Process-group timeout 10 min -> NARROW_PG_TIMEOUT_MIN (60): on 128 nodes some ranks build their 2048-env scene
# more than 10 min after others, and the first collective (NCCL unique-id exchange) timed out (2026-10-28).
_init_pg = torch.distributed.init_process_group


def _init_pg_long(*args, **kwargs):
    import datetime
    kwargs.setdefault("timeout", datetime.timedelta(minutes=float(os.environ.get("NARROW_PG_TIMEOUT_MIN", "60"))))
    return _init_pg(*args, **kwargs)


torch.distributed.init_process_group = _init_pg_long

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gd_lab.core import registry  # noqa: E402

TASK_ID = "Gd-GastNarrowV23-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV22-Rbq10-Dreamwaq-v0"
assert registry.register_task(task="GastNarrowV23", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_narrow.tasks_v23:GastNarrowV23TeacherCfg",
                              agent_cfg="gd_lab.gast_narrow.tasks_v23:GastNarrowV23RunnerCfg") == TASK_ID
registry.register_task(task="GastNarrowV22", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks_v22:GastNarrowV22TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.tasks_v22:GastNarrowV22RunnerCfg")
registry.register_task(task="GastNarrowV21", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks_v21:GastNarrowV21TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.tasks_v21:GastNarrowV21RunnerCfg")
registry.register_task(task="GastNarrowV2", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks_v2:GastNarrowV2TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.model:GastNarrowV2RunnerCfg")
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
def _from_371(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    import hashlib
    import torch
    from gd_lab.gast_narrow.tasks_v2 import NARROW_FAMILIES
    if expected_task != TASK_ID:
        raise ValueError(f"train_narrow_v23.py runs only {TASK_ID}, got {expected_task}")
    # adaptive (KL) is allowed only on request: rsl-rl then sets every param group to one lr, so the terrain group
    # loses its own rate (NARROW_SCHEDULE=adaptive, PPO setting comparison 2026-10-29).
    if alg.schedule != "fixed" and not (alg.schedule == "adaptive" and os.environ.get("NARROW_SCHEDULE") == "adaptive"):
        raise ValueError("narrow fine-tune requires fixed schedule")
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("371 checkpoint sha256 mismatch")
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")  # the task the 371 run trained
    # 371 (v2.2) starts v2.3; a v2.3 checkpoint continues it (moving a run between node counts, 2026-10-28).
    if source not in (SOURCE_TASK_ID, TASK_ID):
        raise ValueError(f"source task {source!r} is neither {SOURCE_TASK_ID!r} nor {TASK_ID!r}")
    means = dict(extra.get("terrain_level_means_by_family") or {})
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("371 checkpoint lacks terrain levels or CENet optimizer")
    src, target = c["model_state_dict"], alg.policy.state_dict()
    source_only = sorted(set(src) - set(target))
    if source_only:
        raise ValueError(f"371 tensors with no v2.3 counterpart: {source_only}")
    merged = {}
    for name, value in target.items():
        if name not in src:
            raise ValueError(f"371 checkpoint lacks {name}")
        s = src[name].to(dtype=value.dtype)
        if tuple(s.shape) != tuple(value.shape):
            raise ValueError(f"{name}: 371 {tuple(s.shape)} != v2.3 {tuple(value.shape)}")
        merged[name] = s
    alg.policy.load_state_dict(merged, strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    ppo_optimizer = "new"
    if source == TASK_ID and c.get("optimizer_state_dict"):
        alg.optimizer.load_state_dict(c["optimizer_state_dict"])
        ppo_optimizer = "restored"
    for family in NARROW_FAMILIES:
        means.setdefault(family, 0.0)
    return dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                source_kind="narrow_v23_continue" if source == TASK_ID else "narrow_v22_371", narrow_version="v2.3",
                target_task=expected_task, ppo_optimizer=ppo_optimizer,
                cenet_optimizer_restored=True, copied=len(merged), fresh=0,
                anchor=os.environ.get("NARROW_ANCHOR_CKPT"), anchor_weight=os.environ.get("NARROW_ANCHOR_WEIGHT", "1.0"),
                learning_rates=warm_start.group_learning_rates(alg.optimizer), terrain_level_means_by_family=means)


warm_start.warm_start_from_rayir = _from_371

from gd_lab.gast_narrow import narrow_top5_v2  # noqa: E402

narrow_top5_v2.install()
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
