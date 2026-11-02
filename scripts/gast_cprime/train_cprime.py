#!/usr/bin/env python3
"""Run scripts/gast/train_teacher.py unchanged for stage C' (docs/design/GAST_NARROW_V26_CLIFF_DESIGN.md).

Same as scripts/gast_bprime/train_bprime.py (copied, the original untouched) with:
1. Registers Gd-GastNarrowCPrime-Rbq10-Dreamwaq-v0 (gd_lab.gast_cprime.tasks: B' + cliff decks + support_end input).
2. --warm_start_rayir accepts a B' checkpoint (start: actor.0.weight widened by the one zero support_end column, new
   PPO optimizer) or a C' checkpoint (continue, PPO restored).
3. Online Top-5: the v2.5 score with the cliff gates (configs/gast_narrow/online_top5_lineage16_cprime.json).
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

TASK_ID = "Gd-GastNarrowCPrime-Rbq10-Dreamwaq-v0"
BPRIME_TASK_ID = "Gd-GastNarrowBPrime-Rbq10-Dreamwaq-v0"
GAIT_TASK_ID = "Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0"
BACKGAP_TASK_ID = "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0"
WIDENED = ("actor.0.weight",)
assert registry.register_task(task="GastNarrowCPrime", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_cprime.tasks:GastNarrowCPrimeTeacherCfg",
                              agent_cfg="gd_lab.gast_cprime.tasks:GastNarrowCPrimeRunnerCfg") == TASK_ID
assert registry.register_task(task="GastNarrowBPrime", robot="Rbq10", method="Dreamwaq",
                              env_cfg="gd_lab.gast_bprime.tasks:GastNarrowBPrimeTeacherCfg",
                              agent_cfg="gd_lab.gast_bprime.tasks:GastNarrowBPrimeRunnerCfg") == BPRIME_TASK_ID
registry.register_task(task="GastNarrowV25", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks_v25:GastNarrowV25TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.tasks_v25:GastNarrowV25RunnerCfg")
registry.register_task(task="GastGaitV3Gap", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_gait.tasks:GastGaitV3GapTeacherCfg",
                       agent_cfg="gd_lab.gast.teacher:GastRayIrRunnerCfg")
registry.register_task(task="GastNarrowV24", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks_v24:GastNarrowV24TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.tasks_v24:GastNarrowV24RunnerCfg")
registry.register_task(task="GastNarrowV23", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks_v23:GastNarrowV23TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.tasks_v23:GastNarrowV23RunnerCfg")
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
def _from_bprime(alg, checkpoint_path, policy_lr, terrain_lr, *, expected_sha256, expected_task):
    import hashlib
    import torch
    from gd_lab.gast_narrow.tasks_v2 import NARROW_FAMILIES
    if expected_task != TASK_ID:
        raise ValueError(f"train_cprime.py runs only {TASK_ID}, got {expected_task}")
    # adaptive (KL) is allowed only on request: rsl-rl then sets every param group to one lr, so the terrain group
    # loses its own rate (NARROW_SCHEDULE=adaptive, PPO setting comparison 2026-10-29).
    if alg.schedule != "fixed" and not (alg.schedule == "adaptive" and os.environ.get("NARROW_SCHEDULE") == "adaptive"):
        raise ValueError("narrow fine-tune requires fixed schedule")
    with open(checkpoint_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.lower():
        raise ValueError("source checkpoint sha256 mismatch")
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = (c.get("infos") or {}).get("gd_lab") or {}
    source = (extra.get("gast_warm_start") or {}).get("task")  # the task the source run trained
    # A B' checkpoint starts C' (one zero actor column); a C' checkpoint continues it. A' / backgap sources go through B'.
    if source not in (BPRIME_TASK_ID, TASK_ID):
        raise ValueError(f"source task {source!r} is not B' or C'")
    from_backgap = False
    means = dict(extra.get("terrain_level_means_by_family") or {})
    if not means or extra.get("cenet_optimizer_state_dict") is None:
        raise ValueError("source checkpoint lacks terrain levels or CENet optimizer")
    src, target = c["model_state_dict"], alg.policy.state_dict()
    source_only = sorted(set(src) - set(target))
    if source_only:
        raise ValueError(f"source tensors with no C' counterpart: {source_only}")
    merged, fresh, widened = {}, [], []
    for name, value in target.items():
        if from_backgap and name.startswith("lidar_encoder."):
            merged[name] = value
            fresh.append(name)
            continue
        if name not in src:
            raise ValueError(f"source checkpoint lacks {name}")
        s = src[name].to(dtype=value.dtype)
        # Zero column: from a B' checkpoint the one support_end column (actor only); the critic is unchanged.
        widen_here = name in WIDENED and value.shape[1] == s.shape[1] + 1
        if widen_here:
            if s.shape[0] != value.shape[0] or s.shape[1] >= value.shape[1]:
                raise ValueError(f"{name}: cannot widen {tuple(s.shape)} to {tuple(value.shape)}")
            grown = torch.zeros_like(value)
            grown[:, :s.shape[1]] = s
            merged[name] = grown
            widened.append(f"{name}: {tuple(s.shape)} -> {tuple(value.shape)} (new columns zero)")
            continue
        if tuple(s.shape) != tuple(value.shape):
            raise ValueError(f"{name}: source {tuple(s.shape)} != C' {tuple(value.shape)}")
        merged[name] = s
    alg.policy.load_state_dict(merged, strict=True)
    alg.policy.cenet.optimizer.load_state_dict(extra["cenet_optimizer_state_dict"])
    alg.optimizer = warm_start.build_grouped_optimizer(alg.policy, policy_lr, terrain_lr, alg.optimizer)
    alg.learning_rate = float(policy_lr)
    ppo_optimizer = "new"
    # A C' checkpoint has the C' shapes: continue with its PPO moments.
    if source == TASK_ID and c.get("optimizer_state_dict") and not widened:
        alg.optimizer.load_state_dict(c["optimizer_state_dict"])
        ppo_optimizer = "restored"
    from gd_lab.gast_narrow.tasks_v24 import TIGHT_FAMILY, TURN_FAMILIES
    from gd_lab.gast_narrow.tasks_v25 import CLIFF_COLUMNS, CLIFF_FAMILY, TURN_ON
    present = (*NARROW_FAMILIES, *((*TURN_FAMILIES, TIGHT_FAMILY[0]) if TURN_ON else ()),
               *((CLIFF_FAMILY[0],) if CLIFF_COLUMNS > 0 else ()))
    for family in present:
        means.setdefault(family, 0.0)  # the cliff family starts at level 0 (restore skips families missing here)
    if source == BPRIME_TASK_ID and not widened:
        raise ValueError("a B' source must widen actor.0.weight by the support_end column")
    kind = {TASK_ID: "cprime_continue", BPRIME_TASK_ID: "cprime_from_bprime"}[source]
    return dict(source=checkpoint_path, source_sha256=digest, source_iteration=int(c["iter"]), source_task=source,
                source_kind=kind, narrow_version="cprime",
                target_task=expected_task, ppo_optimizer=ppo_optimizer,
                cenet_optimizer_restored=True, copied=len(merged) - len(fresh), fresh=len(fresh), widened=widened,
                anchor=os.environ.get("NARROW_ANCHOR_CKPT"), anchor_weight=os.environ.get("NARROW_ANCHOR_WEIGHT", "1.0"),
                learning_rates=warm_start.group_learning_rates(alg.optimizer), terrain_level_means_by_family=means)


warm_start.warm_start_from_rayir = _from_bprime

from gd_lab.gast_narrow import narrow_top5_v25  # noqa: E402

narrow_top5_v25.install()
sys.path.insert(0, str(ROOT / "scripts/gast"))  # cli_args.py next to train_teacher.py
sys.argv[0] = str(ROOT / "scripts/gast/train_teacher.py")
runpy.run_path(sys.argv[0], run_name="__main__")
