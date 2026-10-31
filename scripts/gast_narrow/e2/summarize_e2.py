#!/usr/bin/env python3
"""E2 summary: trials/*/result.json (+ e1/*/result.json) -> summary.json and summary.md (Korean tables).

  python3 scripts/gast_narrow/e2/summarize_e2.py <e2 out dir>

Definitions (each env = its first episode):
  gap      cross    = a slot crossed, not bypassed, no base contact
           clean    = clean attempts / decided attempts (clean + non-clean + retreat + reverse), per attempt
           fall     = base contact termination
  stairs / terrain  pass = reached the far end of the flight (outward: Chebyshev r >= 3.6 m, inward: r <= 1.0 m)
                    before any base contact; fall; track = mean |v_xy - cmd| (m/s) while moving
  narrow   passable rows: pass = exit passed without base contact; touch-free = wall touch < 0.1 s
           blocked rows: hold = 1.5 s still before the line, no breach; breach = front crossed the line
           (v24: cells tagged /back drive the passage tail first)
  turn     (v24) turnable cells: success = target angle reached with the centre drift <= 0.10 m throughout, no base
           contact (the training pass event); dirty = reached with a larger drift (ends the episode); drift_fail =
           drift > 0.20 m before reaching; stall = time-out without reaching; drift = mean / p95 of the max centre drift; time = mean seconds to
           the target. Not-turnable cells: hold = < 15 deg turned and 1.5 s still (the hold event), no breach / fall;
           breach = turned past 15 deg; turned = mean angle turned (deg)
  E1       gap clean/reached, stairs passed/reached (the E1 harness definitions)
Episodes that ended within the first 0.05 s are dropped (stale contact at the start; counted in summary.json).
Retention metrics (gap cross/clean, stairs pass, terrain pass; fall as 1 - fall) are compared with the baseline;
more than max_drop_pt below it is flagged.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


INVALID_START_S = 0.05


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def nose_metrics(rows):
    """Gap nose-up (front up, deg; e2_trial nose_up_*): mean / 90th percentile of the episode max, share > 20 deg,
    mean seconds above 15 deg. Empty for result.json files written before the metric existed."""
    ups = sorted(r["nose_up_max_deg"] for r in rows if "nose_up_max_deg" in r)
    if not ups:
        return {}
    return dict(nose=mean(ups), nose_p90=ups[min(len(ups) - 1, int(0.9 * len(ups)))],
                nose20=sum(u > 20.0 for u in ups) / len(ups), nose_t15=mean([r["nose_up_t15"] for r in rows]))


DEG_KEYS, SEC_KEYS = ("nose", "nose_p90", "turned"), ("nose_t15", "time")


def fmt(key, v):
    if v is None:
        return "-"
    if key in ("track", "clearance", "drift", "drift_p95", "e_min"):
        return f"{v:.3f}"
    if key in DEG_KEYS:
        return f"{v:.1f}°"
    if key == "leg_force_p90":
        return f"{v:.0f} N"
    if key == "leg_impulse":
        return f"{v:.1f}"
    if key in SEC_KEYS:
        return f"{v:.2f} s"
    return pct(v)


def turn_metrics(cell, rows):
    """Turn cell (v24): the cell's tile decides turnable (e2_trial: actual radius >= 0.46 + m)."""
    n = len(rows)
    fall = sum(r["base_contact"] for r in rows) / n
    drifts = sorted(r["drift_max"] for r in rows)
    forces = sorted(r["leg_force_max"] for r in rows if "leg_force_max" in r)
    out = dict(n=n, turnable=cell["turnable"], fall=fall, drift=mean(drifts),
               drift_p95=drifts[min(n - 1, int(0.95 * n))],
               drift_fail=sum(r["drift_done"] and not r["reached"] for r in rows) / n,  # left the drift circle
               touch_free=sum(r["touch_time"] < 0.1 for r in rows) / n,
               leg_force_p90=forces[min(len(forces) - 1, int(0.9 * len(forces)))] if forces else None,
               leg_impulse=mean([r["leg_impulse"] for r in rows if "leg_impulse" in r]) if any("leg_impulse" in r for r in rows) else None)
    if cell["turnable"]:
        ok = [r for r in rows if r["narrow_passed"] and not r["base_contact"]]
        # v2.5 pass = reached, drift <= 0.30, leg peak < 150 N; clean = pass + drift <= 0.10 + no touch (v2.4 files:
        # the pass itself was the clean turn, "dirty" = reached with drift > 0.10).
        clean = [r for r in ok if r.get("turn_clean", r["drift_max"] <= 0.10 and r["touch_time"] < 0.1)]
        out.update(success=len(ok) / n, clean=len(clean) / n,
                   dirty=sum(r["reached"] and not r["narrow_passed"] for r in rows) / n,
                   stall=sum(r["timeout"] and not r["reached"] for r in rows) / n,
                   time=mean([r["reach_t"] for r in ok if r["reach_t"] >= 0]))
    else:
        out.update(hold=sum(r["narrow_held"] and not r["narrow_breached"] and not r["base_contact"] for r in rows) / n,
                   breach=sum(r["narrow_breached"] for r in rows) / n, turned=mean([r["turned_deg"] for r in rows]))
    return out


def cell_metrics(cell, rows):
    s = cell["suite"]
    n = len(rows)
    fall = sum(r["base_contact"] for r in rows) / n
    if s == "gap":
        cross = sum(r["gap_achieved"] and not r["gap_bypassed"] and not r["base_contact"] for r in rows) / n
        att = [sum(x) for x in zip(*[r["gap_attempts"] for r in rows])]  # started, clean, nonclean, retreat, reverse, end
        decided = sum(att[1:5])
        return dict(n=n, cross=cross, clean=att[1] / decided if decided else None, attempts=decided, fall=fall,
                    foot_drop=sum(r.get("sum_platform_gap_foot_drop", 0.0) != 0.0 for r in rows) / n,
                    **nose_metrics(rows))
    if s in ("stairs", "terrain"):
        ok = sum(r["passed_r"] for r in rows) / n
        track = mean([r["err_v"] / r["moving_t"] for r in rows if r["moving_t"] > 0.5])
        return dict(n=n, pass_=ok, fall=fall, track=track,
                    body=sum(r.get("sum_undesired_contacts", 0.0) != 0.0 for r in rows) / n,
                    pitch=mean([r["pitch_max_deg"] for r in rows]))
    if s == "turn":
        return turn_metrics(cell, rows)
    if s == "cliff":  # v25: hold inside the zone, a point over the edge (breach), a fall off the deck
        fell = sum(r.get("cliff_fell", False) or r["base_contact"] for r in rows) / n
        return dict(n=n, hold=sum(r["narrow_held"] and not r["narrow_breached"] and not r.get("cliff_fell", False)
                                  and not r["base_contact"] for r in rows) / n,
                    breach=sum(r["narrow_breached"] for r in rows) / n, fall=fell,
                    e_min=mean([r["e_min"] for r in rows if "e_min" in r]) if rows else None,
                    **nose_metrics(rows))
    passable = [r for r in rows if not r["blocked"]]
    blocked = [r for r in rows if r["blocked"]]
    clear = sorted(r["min_clearance"] for r in passable)
    return dict(n=n, n_pass=len(passable), n_block=len(blocked),
                pass_=(sum(r["narrow_passed"] and not r["base_contact"] for r in passable) / len(passable)) if passable else None,
                touch_free=(sum(r["touch_time"] < 0.1 for r in passable) / len(passable)) if passable else None,
                clearance=clear[len(clear) // 2] if clear else None,
                hold=(sum(r["narrow_held"] and not r["narrow_breached"] and not r["base_contact"] for r in blocked)
                      / len(blocked)) if blocked else None,
                breach=(sum(r["narrow_breached"] for r in blocked) / len(blocked)) if blocked else None, fall=fall)


def load_trials(out: Path):
    res = defaultdict(dict)  # res[condition][model] = {cell key: (cell, metrics)}
    for f in sorted(out.glob("trials/*__*/result.json")):
        model, cond = f.parent.name.split("__", 1)
        cond = cond.split("__only-")[0]  # partial-suite trial directories (run_e2 --only) merge into the condition
        r = json.loads(f.read_text())
        by = defaultdict(list)
        for row in r["rows"]:
            # Ended within the first 0.05 s: the reset carried a stale contact (spawn artefact), not a policy result.
            if row["duration"] <= INVALID_START_S:
                continue
            by[row["cell"]].append(row)
        res[cond].setdefault(model, {}).update({c["key"]: (c, cell_metrics(c, by[c["index"]])) for c in r["cells"]})
    return res


def e1_summary(out: Path):
    res = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # res[(label, arm)][model] = [ok, reached, contact/fd]
    for f in sorted(out.glob("e1/*/result.json")):
        name = f.parent.name
        model, rest = name.split("_s", 1)
        _, arm, label = rest.split("_", 2)
        a = json.loads(f.read_text())["approaches"]
        reached = [x for x in a if x["reached"]]
        cell = res[(label, arm)][model]
        if label == "gap":
            cell[0] += sum(bool(x["clean"]) for x in reached)
            cell[2] += sum(bool(x.get("foot_drop")) for x in a)
        else:
            cell[0] += sum(bool(x["passed"]) for x in reached)
            cell[2] += sum(bool(x.get("body_contact")) for x in a)
        cell[1] += len(reached)
    return res


def main():
    out = Path(sys.argv[1])
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    base, drop = cfg["report"]["baseline"], float(cfg["report"]["max_drop_pt"]) / 100
    res = load_trials(out)
    models = list(cfg["models"])
    lines = [f"# E2 결과: {cfg['name']}", "", f"- 기준 모델: {base}, 기존 능력 지표가 기준보다 {100 * drop:.0f}%p 넘게 낮으면 ⚠",
             f"- seed {cfg['seed']}, 에피소드 {cfg['seconds']:.0f} s, env마다 첫 에피소드 1회", ""]
    flags = defaultdict(list)
    summary = {"conditions": {}, "flags": flags}

    def table(title, cond, suite, cols, pick=None):
        per = res.get(cond, {})
        keys = []
        for m in models:
            for k, (c, _) in per.get(m, {}).items():
                if c["suite"] == suite and k not in keys and (pick is None or pick(c)):
                    keys.append(k)
        if not keys:
            return
        present = [m for m in models if m in per]
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| 셀 | 지표 | " + " | ".join(present) + " |")
        lines.append("|---|---|" + "---|" * len(present))
        agg = defaultdict(lambda: defaultdict(list))
        for k in keys:
            for label, key, retention in cols:
                vals = []
                for m in present:
                    v = per[m].get(k, (None, {}))[1].get(key)
                    vals.append(v)
                    if v is not None:
                        agg[(label, key, retention)][m].append(v)
                if all(v is None for v in vals):
                    continue
                cell = per[present[0]][k][0]
                extra = f" (폭 {cell['actual_width']:.2f})" if "actual_width" in cell and suite in ("narrow",) else ""
                if suite == "turn":
                    extra = f" (r {cell['actual_radius']:.3f}, 여유 {cell['slack']:+.3f}{', 경계' if cell.get('ambiguous') else ''})"
                lines.append(f"| {k}{extra} | {label} | " + " | ".join(
                    fmt(key, v) for v in vals) + " |")
        lines.append(f"| **평균** | | " + " | ".join("" for _ in present) + " |")
        for (label, key, retention), by_m in agg.items():
            avg = {m: mean(v) for m, v in by_m.items()}
            cells_ = []
            for m in present:
                v = avg.get(m)
                txt = fmt(key, v)
                if retention and v is not None and base in avg and avg[base] is not None and m != base:
                    good = v if key != "fall" else 1 - v
                    ref = avg[base] if key != "fall" else 1 - avg[base]
                    if good < ref - drop:
                        txt += " ⚠"
                        flags[m].append(f"{cond}/{suite}/{label} {pct(v)} vs {pct(avg[base])}")
                cells_.append(txt)
            lines.append(f"| 평균 | {label} | " + " | ".join(cells_) + " |")
            summary["conditions"].setdefault(cond, {}).setdefault(suite, {})[label] = avg
        lines.append("")

    for cond in cfg["conditions"]:
        if cond not in res:
            continue
        c = cfg["conditions"][cond]
        rear = f", 후방 Mid-360 {c['rear']}" if "rear" in c else ""
        lines.append(f"## 조건 {cond} (라이다 {c['lidar']}{rear}, 탑재물 {c.get('payload_kg')} kg, 밀기 {c.get('push')}, "
                     f"계단 외란 {c.get('stair_disturbance')})")
        lines.append("")
        table("제자리 회전: 회전 가능", cond, "turn", [("성공", "success", False), ("깨끗한 회전(≤ 0.10 m, 무접촉)", "clean", False),
                                              ("도달·판정 밖", "dirty", False), ("다리 최대력 p90", "leg_force_p90", False),
                                              ("다리 충격량 평균 N·s", "leg_impulse", False),
                                              ("이동 한계 종료", "drift_fail", False), ("미달(stall)", "stall", False),
                                              ("넘어짐", "fall", False), ("몸 중심 최대 이동 m", "drift", False),
                                              ("이동 p95 m", "drift_p95", False), ("소요 시간", "time", False),
                                              ("무접촉", "touch_free", False)], pick=lambda c: c["turnable"])
        table("제자리 회전: 회전 불가 (멈춰야 함)", cond, "turn", [("서기", "hold", False), ("침범(15° 초과)", "breach", False),
                                                         ("회전각 평균", "turned", False),
                                                         ("이동 > 0.2 m 종료", "drift_fail", False), ("넘어짐", "fall", False),
                                                         ("무접촉", "touch_free", False)], pick=lambda c: not c["turnable"])
        table("갭", cond, "gap", [("건넘", "cross", True), ("clean/시도", "clean", True), ("넘어짐", "fall", True),
                                  ("머리 들림 최대 평균", "nose", False), ("머리 들림 최대 p90", "nose_p90", False),
                                  ("20° 초과", "nose20", False), ("15° 초과 시간", "nose_t15", False)])
        table("계단", cond, "stairs", [("통과", "pass_", True), ("넘어짐", "fall", True), ("몸통 접촉", "body", False)])
        table("경사·거친 지면", cond, "terrain", [("통과", "pass_", True), ("넘어짐", "fall", True),
                                             ("속도 오차 m/s", "track", False)])
        table("좁은 통로", cond, "narrow", [("통과", "pass_", False), ("무접촉", "touch_free", False),
                                         ("벽 거리 중앙값 m", "clearance", False), ("서기", "hold", False),
                                         ("침범", "breach", False)])
        table("막힌 통로", cond, "dead_end", [("서기", "hold", False), ("침범", "breach", False)])
        table("낭떠러지 (멈춰야 함)", cond, "cliff", [("서기", "hold", False), ("끝선 침범", "breach", False), ("추락", "fall", False),
                                              ("끝선 최소 여유 m", "e_min", False), ("머리 들림 최대 평균", "nose", False),
                                              ("15° 초과 시간", "nose_t15", False)])
    e1 = e1_summary(out)
    if e1:
        lines += ["## E1 (LiDAR 지도 0, m 접어 넣음)", "", "| 지형 | arm | " + " | ".join(models) + " |",
                  "|---|---|" + "---|" * len(models)]
        for (label, arm), by_m in sorted(e1.items()):
            vals = []
            for m in models:
                ok, reached, bad = by_m.get(m, [0, 0, 0])
                vals.append(f"{ok}/{reached} = {pct(ok / reached) if reached else '-'} ({bad})" if reached else "-")
            lines.append(f"| {label} | {arm} | " + " | ".join(vals) + " |")
        lines.append("")
        summary["e1"] = {f"{k[0]}/{k[1]}": dict(v) for k, v in e1.items()}
    lines += ["## 기준 대비 하락 (⚠)", ""]
    for m in models:
        if flags.get(m):
            lines.append(f"- **{m}**: " + "; ".join(flags[m]))
    if not any(flags.values()):
        lines.append("- 없음")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
