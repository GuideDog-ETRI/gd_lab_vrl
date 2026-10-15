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
  negative  -> that term's discounted reward is lower than usual around the behavior
  positive  -> higher than usual
It is the critic-free discounted share (not the GAE, no critic residual) and a diagnostic, not a causal attribution
of the action: a penalty that is less negative than usual also shows as positive, and shares near the end of the
recording/episode are censored. Action events (liftoff/touchdown) compare contact before vs after the same step.
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
BAD_EVENTS = ("foot_drop", "high_lift", "hind_hop", "fall")
# Actions the policy takes every stride: a foot group leaving or touching the ground (contact transitions).
ACTION_EVENTS = ("liftoff_front", "touchdown_front", "liftoff_hind", "touchdown_hind")
EVENTS = BAD_EVENTS + ACTION_EVENTS
NAMES_KO = {"foot_drop": "발 빠짐", "high_lift": "높이 차기", "hind_hop": "뒷발 동시 뜀", "fall": "넘어짐",
            "liftoff_front": "앞발 들기", "touchdown_front": "앞발 딛기", "liftoff_hind": "뒷발 들기", "touchdown_hind": "뒷발 딛기"}
ACTION_LEAD = 5    # action events: analysis offset 0.05 s before the step whose action changed the contact
DEBOUNCE, FALL_DONE_TILT_DEG = 2, 30.0


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
    front, hind = foot_groups(meta["foot_names"])
    air = ~contact[:, hind].any(1)
    events["hind_hop"] = [r for r in runs(air) if r[1] - r[0] >= HOP_STEPS]
    done = np.array(env["done"], dtype=bool)
    tilt = tilt_deg(np.array(env["base_quat"], dtype=float))
    # Proxy: a fall is a large tilt, or a termination while already tilted (a time-out at level stance is not a fall).
    events["fall"] = runs((tilt > TILT_DEG) | (done & (tilt > FALL_DONE_TILT_DEG)))
    # Foot liftoff / touchdown caused by the action of step i: contact before step i vs after step i (same row, so a
    # reset never links two episodes). Debounced: a change that flips back within DEBOUNCE steps is contact chatter.
    after = np.array(env["foot_contact_after"], dtype=float) if env.get("foot_contact_after") is not None else None
    for group, idx in (("front", front), ("hind", hind)):
        for kind in ("liftoff", "touchdown"):
            steps = set()
            for f in idx:
                c0 = contact[:, f]
                if after is not None and np.isfinite(after[:, f]).all():
                    c1 = after[:, f] > 0.5
                else:  # older recordings: next row, never across a reset
                    c1 = np.r_[c0[1:], c0[-1]]
                    c1 = np.where(done, c0, c1)
                edge = (c0 & ~c1) if kind == "liftoff" else (~c0 & c1)
                for i in np.flatnonzero(edge):
                    hold = c0[i + 1:i + 1 + DEBOUNCE]       # the next rows of the same episode must keep the new state
                    if len(hold) == DEBOUNCE and not done[i:i + DEBOUNCE].any() and not (hold == c1[i]).all():
                        continue  # did not persist: contact chatter
                    steps.add(int(i))
            events[f"{kind}_{group}"] = [(i, i + 1) for i in sorted(steps)]  # group event, deduplicated across feet
    return events


def foot_groups(names):
    """(front, hind) foot indices; only the RBQ10 names FL/FR/RL/RR (or HL/HR) are accepted."""
    keys = [n.split("_")[0].upper() for n in names]
    front = [i for i, k in enumerate(keys) if k in ("FL", "FR")]
    hind = [i for i, k in enumerate(keys) if k in ("RL", "RR", "HL", "HR")]
    if len(front) != 2 or len(hind) != 2:
        raise ValueError(f"unexpected foot names {names}")
    return front, hind


def dominance(share, done):
    """[T, K] deviation of each term's discounted share from its own mean over the same episode, and argmax |dev|.

    This is a diagnostic of which term's discounted reward is unusually high (+) or low (-) at a step; it is not the
    GAE (no critic residual) and not a causal attribution of the action. Shares near the end of the recording or
    episode are cut short (censored), so they shrink toward 0."""
    done = np.asarray(done, dtype=bool)
    episode = np.concatenate(([0], np.cumsum(done[:-1])))
    dev = np.empty_like(share)
    for k in np.unique(episode):
        rows = episode == k
        dev[rows] = share[rows] - share[rows].mean(0, keepdims=True)
    return dev, np.abs(dev).argmax(1)


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
        done = np.array(env["done"], dtype=bool)
        episode = np.concatenate(([0], np.cumsum(done[:-1])))  # episode id of each row
        for name, spans in ev.items():
            busy = np.zeros(T, bool)
            lead = ACTION_LEAD if name in ACTION_EVENTS else LEAD
            for s, e in spans:
                guard = GUARD if name in BAD_EVENTS else lead + 2   # keep the decision window out of the baseline
                busy[max(0, s - guard):min(T, e + (GUARD if name in BAD_EVENTS else 2))] = True
                t = s - lead
                if t < 0 or episode[t] != episode[s]:
                    continue  # the analysis step would fall before this episode started: censored
                decision[name].append(share[t])
                found[name].append({"env": idx, "start": s, "end": e, "decision_step": t,
                                    "advantage": env["advantage"][t] if env.get("advantage") else None})
            baseline[name].append(share[~busy])
    report = {"meta": {"task": meta.get("task"), "checkpoint": meta.get("checkpoint"), "driver": meta.get("driver"),
                       "num_envs": meta.get("num_envs"), "steps": meta.get("steps"), "dt": meta.get("dt")},
              "definitions": {"foot_drop_m": DROP_M, "high_lift_m": LIFT_M, "tilt_deg": TILT_DEG,
                              "hind_hop_steps": HOP_STEPS, "lead_steps": LEAD, "guard_steps": GUARD,
                              "action_lead_steps": ACTION_LEAD},
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


def annotate(env, meta):
    """Per-env fields the viewer shows: term_adv [T,K] (from unrounded terms when given), events {name: [[s, e), ...]}."""
    share = advantage_share(env, meta)
    return {"term_adv": np.round(share, 6).tolist(),
            "events": {k: [list(map(int, r)) for r in v] for k, v in detect(env, meta).items()}}


def markdown(report, label):
    lines = [f"### {label}: 나쁜 행동과 보상 항목", "",
             "분석 시점(사건 0.2초 전)의 항목별 할인 보상 몫(critic 제외, GAE 아님) − 사건과 무관한 구간의 평균. 음수 = 평소보다 작음, 양수 = 평소보다 큼. 인과 귀속이 아닌 진단값.", "",
             "| 행동 | 횟수 | 평소보다 작은 항목 (음수) | 평소보다 큰 항목 (양수) |", "|---|---|---|---|"]
    for name in EVENTS:
        if name == ACTION_EVENTS[0]:
            lines += ["", "분석 시점(접지를 바꾼 행동 0.05초 전)의 같은 값 (보폭마다 반복되는 행동; 같은 스텝의 양발 전이는 1회).", "",
                      "| 행동 | 횟수 | 평소보다 작은 항목 (음수) | 평소보다 큰 항목 (양수) |", "|---|---|---|---|"]
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
