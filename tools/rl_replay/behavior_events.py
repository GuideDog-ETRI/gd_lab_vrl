"""Bad-behavior events in a rollout recording and the reward terms behind them.

Reads a recording written by RolloutLog (record_rollout.py or train_student_live.py --replay_out), finds the
behaviors we want to remove, and ranks the reward terms by how much they push the policy toward or away from
each behavior. Pure numpy, no simulator.

  python3 tools/rl_replay/behavior_events.py logs/rl_replay/<rec>.json [--md report.md] [--json events.json]

Events (definitions follow gd_rbq10_deploy_vrl/evaluation gap_feet.py / hind_metrics.py and the v2.1 terms):
  foot_drop     a foot more than 3 cm below the supporting deck (the gap intrusion threshold)
  high_lift     a foot more than 20 cm above the deck under it (the v2.1 over-lift threshold)
  hind_hop      both hind feet off the ground for >= 50 ms (5 policy steps)
  fall          episode termination, or base tilt > 60 deg
Heights come from the recorded height-scanner hits (ground truth, 11x17 around the base). The deck under a foot
is the highest scan hit within 25 cm of it (spans the 2-26 cm training gaps); the foot's own standing offset is calibrated per env from steps
where that foot is in contact on level ground.

Attribution. For every reward term k the per-term advantage share is
    a_k(t) = r_k(t) + gamma * lam * (1 - done_t) * a_k(t + 1)
i.e. GAE with the critic left out, split by term (sum over k + the critic part = A_t). At the decision step
0.2 s before an event starts it says which terms make that stretch better or worse than usual. Reported per
event type: the mean a_k at those steps minus its mean over steps far from any event of that type.
  negative  -> the term penalizes what led to the behavior (PPO pushes away from it)
  positive  -> the term rewards it (a term that encourages the bad behavior)
The window is short and the recording is a few seconds, so this is a hint where to look, not a measurement
of the reward's effect on training.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

LEAD = 20          # decision step: 0.2 s before onset at 100 Hz
GUARD = 50         # baseline excludes +-0.5 s around events
DROP_M, LIFT_M, TILT_DEG, HOP_STEPS, DECK_R, MERGE = 0.03, 0.20, 60.0, 5, 0.25, 2
EVENTS = ("foot_drop", "high_lift", "hind_hop", "fall")
NAMES_KO = {"foot_drop": "발 빠짐", "high_lift": "높이 차기", "hind_hop": "뒷발 동시 뜀", "fall": "넘어짐"}


def tilt_deg(quat_wxyz):
    w, x, y, z = quat_wxyz.T
    up_z = 1 - 2 * (x * x + y * y)  # body z axis projected on world z
    return np.degrees(np.arccos(np.clip(up_z, -1, 1)))


def runs(mask):
    """(start, end_exclusive) of consecutive True runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    merged = []
    for r in out:  # one event even if a scan sample flickers for a step or two
        if merged and r[0] - merged[-1][1] <= MERGE:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return merged


def detect(env, meta):
    """{event: [(start, end), ...]} for one env."""
    T = len(env["reward"])
    names = meta["body_names"]
    feet = [names.index(n) for n in meta["foot_names"]]
    bodies = np.array(env["bodies"], dtype=float)            # [T, B, 3]
    foot = bodies[:, feet]                                    # [T, 4, 3]
    scan = np.array([[[np.nan if c is None else c for c in p] for p in s] for s in env["scan"]], dtype=float)  # [T,187,3]
    contact = np.array(env["foot_contact"], dtype=bool)       # [T, 4]
    deck = np.full(foot.shape[:2], np.nan)
    below = np.full(foot.shape[:2], np.nan)
    for t in range(T):
        pts = scan[t]
        ok = np.isfinite(pts).all(1)
        if not ok.any():
            continue
        pts = pts[ok]
        d = np.hypot(pts[None, :, 0] - foot[t, :, None, 0], pts[None, :, 1] - foot[t, :, None, 1])  # [4, P]
        near = d < DECK_R
        z = np.where(near, pts[None, :, 2], -np.inf)
        deck[t] = np.where(near.any(1), z.max(1), np.nan)
        below[t] = pts[d.argmin(1), 2]                         # ground right under the foot
    level = np.abs(deck - below) < 0.01
    offset = np.array([np.nanmedian(np.where(contact[:, f] & level[:, f], foot[:, f, 2] - deck[:, f], np.nan))
                       if (contact[:, f] & level[:, f]).any() else 0.02 for f in range(4)])
    sole = foot[..., 2] - offset[None]                        # foot bottom estimate
    events = {}
    events["foot_drop"] = runs(((sole < deck - DROP_M) & np.isfinite(deck)).any(1))
    events["high_lift"] = runs(((sole > deck + LIFT_M) & np.isfinite(deck)).any(1))
    hind = [i for i, n in enumerate(meta["foot_names"]) if n.upper().startswith(("R", "H"))]
    air = ~contact[:, hind].any(1) if len(hind) == 2 else np.zeros(T, bool)
    events["hind_hop"] = [r for r in runs(air) if r[1] - r[0] >= HOP_STEPS]
    done = np.array(env["done"], dtype=bool)
    tilt = tilt_deg(np.array(env["base_quat"], dtype=float))
    events["fall"] = runs(done | (tilt > TILT_DEG))
    return events


def advantage_share(env, meta):
    """[T, K] per-term GAE share without the critic."""
    terms = np.array(env["terms"], dtype=float)
    done = np.array(env["done"], dtype=float)
    k = meta["gamma"] * meta["lam"]
    share = np.zeros_like(terms)
    nxt = np.zeros(terms.shape[1])
    for t in range(len(terms) - 1, -1, -1):
        nxt = terms[t] + k * (1 - done[t]) * nxt
        share[t] = nxt
    return share


def analyze(rec):
    meta = rec["meta"]
    K = len(meta["term_names"])
    found = {e: [] for e in EVENTS}
    decision = {e: [] for e in EVENTS}
    baseline = {e: [] for e in EVENTS}
    for idx, env in enumerate(rec["envs"]):
        ev = detect(env, meta)
        share = advantage_share(env, meta)
        T = len(share)
        for name, spans in ev.items():
            busy = np.zeros(T, bool)
            for s, e in spans:
                busy[max(0, s - GUARD):min(T, e + GUARD)] = True
                t = max(0, s - LEAD)
                decision[name].append(share[t])
                found[name].append({"env": idx, "start": s, "end": e, "decision_step": t,
                                    "advantage": env["advantage"][t] if env.get("advantage") else None})
            baseline[name].append(share[~busy])
    report = {"meta": {"task": meta.get("task"), "checkpoint": meta.get("checkpoint"), "driver": meta.get("driver"),
                       "num_envs": meta.get("num_envs"), "steps": meta.get("steps"), "dt": meta.get("dt")},
              "definitions": {"foot_drop_m": DROP_M, "high_lift_m": LIFT_M, "tilt_deg": TILT_DEG,
                              "hind_hop_steps": HOP_STEPS, "lead_steps": LEAD, "guard_steps": GUARD},
              "events": {}}
    for name in EVENTS:
        base = np.concatenate(baseline[name]) if baseline[name] else np.zeros((0, K))
        entry = {"count": len(found[name]), "occurrences": found[name], "terms": []}
        if found[name] and len(base):
            diff = np.mean(decision[name], 0) - base.mean(0)
            order = np.argsort(diff)
            entry["terms"] = [{"term": meta["term_names"][i], "share_minus_baseline": round(float(diff[i]), 5)}
                              for i in order]
        report["events"][name] = entry
    return report


def markdown(report, label):
    lines = [f"### {label}: 나쁜 행동과 보상 항목", "",
             "결정 시점(사건 0.2초 전)의 항목별 advantage 몫 − 평소 값. 음수 = 그 행동을 벌줌, 양수 = 부추김.", "",
             "| 행동 | 횟수 | 가장 벌주는 항목 (음수) | 부추기는 항목 (양수) |", "|---|---|---|---|"]
    for name in EVENTS:
        e = report["events"][name]
        neg = [t for t in e["terms"] if t["share_minus_baseline"] < 0][:3]
        pos = [t for t in reversed(e["terms"]) if t["share_minus_baseline"] > 0][:3]
        fmt = lambda ts: ", ".join(f"{t['term']} ({t['share_minus_baseline']:+.3f})" for t in ts) or "—"
        lines.append(f"| {NAMES_KO[name]} | {e['count']} | {fmt(neg) if e['count'] else '—'} | {fmt(pos) if e['count'] else '—'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording")
    ap.add_argument("--label", default=None)
    ap.add_argument("--md", default=None, help="write the markdown table here")
    ap.add_argument("--json", default=None, help="write events + attribution here (default <recording>.events.json)")
    a = ap.parse_args()
    rec = json.loads(Path(a.recording).read_text())
    report = analyze(rec)
    out = Path(a.json) if a.json else Path(a.recording).with_suffix(".events.json")
    out.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=1))
    md = markdown(report, a.label or Path(a.recording).stem)
    if a.md:
        Path(a.md).write_text(md)
    print(md)
    print(f"events: {out}")


if __name__ == "__main__":
    main()
