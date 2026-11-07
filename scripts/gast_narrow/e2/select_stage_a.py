#!/usr/bin/env python3
"""Pick STAGE_A from an E2 run of the lineage stage A candidates (docs/hangang/WORKORDER_STAGE_B_V25_20261031.md).

Rule (nominal condition): a candidate qualifies when its backward gap crossing, forward gap crossing and stairs pass
rates are each >= the baseline's (6566) minus --tol points; among the qualifying ones the highest backward gap
crossing wins (ties: forward gap, then stairs). Writes <out>/STAGE_A.txt ("<name> <absolute path> <sha256>") and
<out>/stage_a_selection.md (the table), and prints both. Exit 2 when no candidate qualifies (the file is not written).
  python3 scripts/gast_narrow/e2/select_stage_a.py logs/e2/<run> [--baseline b6566] [--tol 1.0] [--condition nominal]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from summarize_e2 import load_trials  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--baseline", default="b6566")
    ap.add_argument("--tol", type=float, default=1.0, help="points below the baseline still accepted")
    ap.add_argument("--condition", default="nominal")
    a = ap.parse_args()
    out = Path(a.out)
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    res = load_trials(out).get(a.condition, {})
    if a.baseline not in res:
        sys.exit(f"baseline {a.baseline} has no {a.condition} trial in {out}")
    rows = {}
    for model, cells in res.items():
        back = [m.get("cross") for c, m in cells.values() if c["suite"] == "gap" and c["direction"] == "backward"]
        fwd = [m.get("cross") for c, m in cells.values() if c["suite"] == "gap" and c["direction"] == "forward"]
        stairs = [m.get("pass_") for c, m in cells.values() if c["suite"] == "stairs"]
        rows[model] = (mean(back), mean(fwd), mean(stairs))
    base = rows[a.baseline]
    tol = a.tol / 100.0
    lines = [f"# STAGE_A selection ({out.name}, {a.condition}, baseline {a.baseline}, tolerance {a.tol} pt)", "",
             "| 모델 | 후진 갭 건넘 | 전진 갭 건넘 | 계단 통과 | 통과 |", "|---|---|---|---|---|"]
    ok = []
    for model, vals in sorted(rows.items(), key=lambda kv: [-(v or 0) for v in kv[1]]):
        passed = model != a.baseline and all(v is not None and b is not None and v >= b - tol for v, b in zip(vals, base))
        if passed:
            ok.append((vals, model))
        cells = " | ".join("-" if v is None else f"{100 * v:.1f}" for v in vals)
        lines.append(f"| {model} | {cells} | {'O' if passed else ('기준' if model == a.baseline else 'X')} |")
    text = "\n".join(lines) + "\n"
    (out / "stage_a_selection.md").write_text(text)
    print(text)
    if not ok:
        print("no candidate qualifies: STAGE_A.txt not written")
        sys.exit(2)
    _, best = max(ok)
    path = Path(cfg["models"][best])
    path = path if path.is_absolute() else ROOT / path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "STAGE_A.txt").write_text(f"{best} {path} {digest}\n")
    print(f"STAGE_A = {best} {path} {digest}")


if __name__ == "__main__":
    main()
