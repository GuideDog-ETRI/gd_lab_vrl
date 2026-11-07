#!/usr/bin/env python3
"""Lineage stage A (16-node plan): scripts/gast_backgap/train_backgap.py unchanged, with the multi-node set-up of
train_narrow_v23.py (rank GPU before Isaac, long process-group timeout, optional stack dumps).

Launch through scripts/gast_narrow/train_narrow_v23.sh with NARROW_TRAIN_PY=scripts/gast_narrow/train_lineage_a.py,
NARROW_TASK=Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0 and NARROW_SOURCE=3906 (see docs/hangang/WORKORDER_LINEAGE16_20261030.md).
"""
import faulthandler
import os
import runpy
import signal
import sys
from pathlib import Path

import torch

# The deferred normalizer update is flushed only by the narrow anchor runner; the backward-gap runner has none.
if os.environ.get("GD_LAB_NORMALIZER_DEFER", "0") == "1":
    raise SystemExit("lineage stage A needs GD_LAB_NORMALIZER_DEFER=0 (no anchor runner to flush the normalizer)")

if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

faulthandler.register(signal.SIGUSR1, all_threads=True)
_dump_ranks = os.environ.get("NARROW_STACK_DUMP_RANKS", "0").split(",")
if os.environ.get("NARROW_STACK_DUMP_S") and ("all" in _dump_ranks or os.environ.get("RANK", "0") in _dump_ranks):
    faulthandler.dump_traceback_later(float(os.environ["NARROW_STACK_DUMP_S"]), repeat=True)

_init_pg = torch.distributed.init_process_group


def _init_pg_long(*args, **kwargs):
    import datetime
    kwargs.setdefault("timeout", datetime.timedelta(minutes=float(os.environ.get("NARROW_PG_TIMEOUT_MIN", "60"))))
    return _init_pg(*args, **kwargs)


torch.distributed.init_process_group = _init_pg_long

ROOT = Path(__file__).resolve().parents[2]
sys.argv[0] = str(ROOT / "scripts/gast_backgap/train_backgap.py")
runpy.run_path(sys.argv[0], run_name="__main__")
