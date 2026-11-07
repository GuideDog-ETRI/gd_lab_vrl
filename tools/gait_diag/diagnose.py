#!/usr/bin/env python3
"""MuJoCo gait targets -> rl_replay reward attribution.

For every target in gait_targets.TARGETS: the value measured in the MuJoCo runs (deploy-repo state.jsonl, 10 Hz)
and in the rl_replay recordings of the same checkpoint (100 Hz, Isaac). For each target the replay misses, the
steps that violate it are compared with the clean steps of the same scene:

- brakes: weighted reward per second at the violating steps minus at the clean steps, per term. Negative = the term
  charges the violation itself. A small total means nothing in the reward stops the behaviour.
- drivers: the term's critic-free discounted share (``term_adv``) in the ``window`` seconds before each violation
  onset minus its mean over clean steps. Positive = the term's future reward is higher on the way into the
  violation, i.e. the term pays for the path that ends there. Correlational (on-policy, one checkpoint).

usage: diagnose.py --mujoco <run_dir>[,<run_dir>...] --replay <recording.json>[,...] [--out report.md]
       run dirs are classified by name: *flat*, *stairs*, *gap*/*backward*.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait_targets import BY_KEY, TARGETS, passes  # noqa: E402

FEET = (13, 14, 15, 16)  # FL FR RL RR in meta body_names
HIPS = (1, 2, 3, 4)      # FL_hip FR_hip RL_hip RR_hip
HIND = (2, 3)
FOOT_RADIUS = 0.02
FLAT_RANGE = 0.03
BAND_UP, BAND_DOWN = 3.0, 6.0
POSITIVE = {"track_lin_vel_xy_exp", "track_ang_vel_z_exp", "platform_gap_crossing", "platform_gap_clean",
            "gap_foot_clean", "platform_gap_monitor"}


def pct(xs, p):
    xs = sorted(x for x in xs if x == x)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def stat(xs, kind):
    return {"mean": mean, "p90": lambda v: pct(v, 0.9), "max": lambda v: max(v) if v else float("nan")}[kind](xs)


# ---------------------------------------------------------------- MuJoCo -------------------------------------------
def mujoco_scene(name):
    n = name.lower()
    if "stairs" in n:
        return "stairs"
    if "flat" in n:
        return "flat"
    return "gap"


def mujoco_metrics(run_dirs):
    out = {k: [] for k in BY_KEY}
    runs = {"fall": 0, "n": 0}
    for d in run_dirs:
        rows = []
        for line in open(os.path.join(d, "state.jsonl")):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not rows:
            continue
        scene = mujoco_scene(os.path.basename(d.rstrip("/")))
        runs["n"] += 1
        if min(r["pos"][2] for r in rows) < 0.3:
            runs["fall"] += 1
        if scene != "flat":
            continue
        moving = [r for r in rows if abs(r["vel"][0]) > 0.15]
        for r in moving:
            x, y, z, w = r["imu_xyzw"]
            out["nose_mean"].append(math.degrees(math.asin(max(-1, min(1, 2 * (x * z - w * y))))))
        out["nose_p90"] = out["nose_max"] = out["nose_mean"]
        static = [st.median(r["foot_fz"][f] for r in rows[:10]) for f in range(4)]
        dt = st.median(b["wall"] - a["wall"] for a, b in zip(rows, rows[1:]) if b["wall"] > a["wall"])
        if not moving:
            continue
        i0, i1 = rows.index(moving[0]), rows.index(moving[-1])
        for f in HIND:
            fz = [rows[i]["foot_fz"][f] for i in range(i0, i1)]
            on = fz[0] > 15
            for i, v in enumerate(fz):
                if not on and v > 15:
                    on = True
                    out["td_force_p90"].append(max(fz[i:i + int(0.08 / dt) + 1]) / max(static[f], 1.0))
                elif on and v < 5:
                    on = False
    res = {}
    for t in TARGETS:
        if t.key == "fall_rate":
            res[t.key] = runs["fall"] / runs["n"] if runs["n"] else float("nan")
        elif t.key in ("nose_mean", "nose_p90", "nose_max", "td_force_p90"):
            res[t.key] = stat(out[t.key], t.stat)
        else:
            res[t.key] = float("nan")  # needs 100 Hz foot data: replay only
    return res


# ---------------------------------------------------------------- replay -------------------------------------------
def nose_deg(q):
    w, x, y, z = q
    return -math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - x * z)))))


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def ground_pitch(scan, pos, q):
    c, s = math.cos(yaw(q)), math.sin(yaw(q))
    f, b, fx, bx = [], [], [], []
    for h in scan:
        if not math.isfinite(h[2]):
            continue
        dx, dy = h[0] - pos[0], h[1] - pos[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        if abs(ly) >= 0.25:
            continue
        if 0.1 < lx < 0.5:
            f.append(h[2]); fx.append(lx)
        elif -0.5 < lx < -0.1:
            b.append(h[2]); bx.append(lx)
    if not f or not b:
        return 0.0
    return math.degrees(math.atan2(st.median(f) - st.median(b), max(0.05, st.median(fx) - st.median(bx))))


def body_dx(q, foot, hip):
    """Foot minus hip along the trunk heading (yaw frame), m."""
    c, s_ = math.cos(yaw(q)), math.sin(yaw(q))
    return c * (foot[0] - hip[0]) + s_ * (foot[1] - hip[1])


def nearest_ground(scan, p):
    best, z = 1e9, float("nan")
    for h in scan:
        d = (h[0] - p[0]) ** 2 + (h[1] - p[1]) ** 2
        if d < best and math.isfinite(h[2]):
            best, z = d, h[2]
    return z


class Steps:
    """Per-step labels of one recording: scene, eligibility and the violation masks of every target."""

    def __init__(self, path):
        d = json.load(open(path))
        m = d["meta"]
        self.name = os.path.basename(path)
        self.names = m["term_names"]
        self.dt = m["step_dt"]
        self.envs = d["envs"]
        self.label()

    def label(self):
        self.rows = []  # (env, t, scene, moving, {target: violated or None}, value per target)
        self.values = {k: [] for k in BY_KEY}
        self.fall_envs = 0
        self.drop_envs = 0
        self.gap_envs = 0
        for ei, e in enumerate(self.envs):
            T = len(e["terms"])
            ev = e.get("events") or {}
            fall_steps = {t for a, b in ev.get("fall", []) for t in range(a, b)}
            drop_on = {a for a, _ in ev.get("foot_drop", [])}
            if ev.get("fall"):
                self.fall_envs += 1
            scenes, viol = [], [dict() for _ in range(T)]
            for t in range(T):
                zz = [h[2] for h in e["scan"][t] if math.isfinite(h[2])]
                rng = (max(zz) - min(zz)) if zz else 1.0
                g = ground_pitch(e["scan"][t], e["base_pos"][t], e["base_quat"][t]) if rng >= FLAT_RANGE else 0.0
                scene = "flat" if rng < FLAT_RANGE else ("stairs" if abs(g) > 5.0 else "gap")
                scenes.append(scene)
                moving = math.hypot(*e["command"][t][:2]) > 0.1
                n = nose_deg(e["base_quat"][t])
                if moving and scene == "flat":
                    self.values["nose_mean"].append(n)
                    for k in ("nose_mean", "nose_p90", "nose_max"):
                        viol[t][k] = n > BY_KEY[k].hi
                if moving and scene == "stairs":
                    out = max(n - (max(0.0, g) + BAND_UP), (min(0.0, g) - BAND_DOWN) - n, 0.0)
                    self.values["nose_band_p90"].append(out)
                    viol[t]["nose_band_p90"] = out > BY_KEY["nose_band_p90"].hi
                if t in fall_steps:
                    viol[t]["fall_rate"] = True
                if t in drop_on:
                    viol[t]["foot_drop_rate"] = True
            if any(s == "gap" for s in scenes):
                self.gap_envs += 1
                if drop_on:
                    self.drop_envs += 1
            self.values["nose_p90"] = self.values["nose_max"] = self.values["nose_mean"]
            # feet: touchdown speed, swing time, lift height (flat, moving)
            for fi, b in enumerate(FEET):
                lift_t, peak, peak_t = None, -1.0, None
                for t in range(1, T):
                    c0, c1 = e["foot_contact"][t - 1][fi], e["foot_contact"][t][fi]
                    p = e["bodies"][t][b]
                    if c0 and not c1:
                        lift_t, peak, peak_t = t, -1.0, None
                    elif not c1 and lift_t is not None:
                        gz = nearest_ground(e["scan"][t], p)
                        h = p[2] - FOOT_RADIUS - gz if gz == gz else float("nan")
                        if h == h and h > peak:
                            peak, peak_t = h, t
                    elif not c0 and c1 and lift_t is not None:
                        flat = scenes[t] == "flat" and math.hypot(*e["command"][t][:2]) > 0.1
                        if flat:
                            q0 = e["bodies"][t - 1][b]
                            sp = math.dist(q0, p) / self.dt
                            sw = (t - lift_t) * self.dt
                            self.values["td_speed_p90"].append(sp)
                            self.values["swing_mean"].append(sw)
                            viol[t]["td_speed_p90"] = viol[t].get("td_speed_p90", False) or (sp > BY_KEY["td_speed_p90"].hi)
                            tg = BY_KEY["swing_mean"]
                            viol[t]["swing_mean"] = viol[t].get("swing_mean", False) or (not (tg.lo <= sw <= tg.hi))
                            if peak_t is not None:
                                self.values["lift_max_p90"].append(peak)
                                tl = BY_KEY["lift_max_p90"]
                                viol[peak_t]["lift_max_p90"] = viol[peak_t].get("lift_max_p90", False) or (not (tl.lo <= peak <= tl.hi))
                        lift_t = None
            # gap leg following (gap tiles, moving): trailing-foot speed / lag on forward crossings, leading hind brace on
            # backward ones (6566 crosses backward with a long brace and a fast trailing flight, 0 falls)
            for fi, b in enumerate(FEET):
                lift_t, peak_sp, peak_t, stance_t = None, 0.0, None, None
                for t in range(1, T):
                    c0, c1 = e["foot_contact"][t - 1][fi], e["foot_contact"][t][fi]
                    vx = e["command"][t][0]
                    if scenes[t] != "gap" or abs(vx) < 0.1:
                        lift_t, stance_t = None, None
                        continue
                    trailing = (fi >= 2) if vx > 0 else (fi < 2)
                    p, q0 = e["bodies"][t][b], e["bodies"][t - 1][b]
                    sp = math.dist(q0, p) / self.dt
                    if c0 and not c1:
                        lift_t, peak_sp, peak_t = t, 0.0, None
                        if stance_t is not None and not trailing and vx < 0:
                            dur = (t - stance_t) * self.dt
                            self.values["back_brace_p90"].append(dur)
                            viol[t]["back_brace_p90"] = viol[t].get("back_brace_p90", False) or (dur < BY_KEY["back_brace_p90"].lo)
                        stance_t = None
                        if trailing and vx > 0:  # trailing-foot targets: forward gap only (G4)
                            lag = -math.copysign(1.0, vx) * body_dx(e["base_quat"][t], p, e["bodies"][t][HIPS[fi]])
                            self.values["gap_trail_lag_p90"].append(lag)
                            viol[t]["gap_trail_lag_p90"] = viol[t].get("gap_trail_lag_p90", False) or (lag > BY_KEY["gap_trail_lag_p90"].hi)
                    elif not c1 and lift_t is not None and trailing and vx > 0:
                        if sp > peak_sp:
                            peak_sp, peak_t = sp, t
                    elif not c0 and c1:
                        stance_t = t
                        if lift_t is not None and trailing and vx > 0:
                            self.values["gap_td_speed_p90"].append(sp)
                            viol[t]["gap_td_speed_p90"] = viol[t].get("gap_td_speed_p90", False) or (sp > BY_KEY["gap_td_speed_p90"].hi)
                            if peak_t is not None:
                                self.values["gap_swing_peak_p90"].append(peak_sp)
                                viol[peak_t]["gap_swing_peak_p90"] = viol[peak_t].get("gap_swing_peak_p90", False) or (peak_sp > BY_KEY["gap_swing_peak_p90"].hi)
                        lift_t = None
            for t in range(T):
                self.rows.append((ei, t, scenes[t], viol[t]))

    def metric(self, key):
        t = BY_KEY[key]
        if key == "fall_rate":
            return self.fall_envs / len(self.envs)
        if key == "td_force_p90":
            return float("nan")  # replay has no foot forces
        return stat(self.values[key], t.stat)

    def attribute(self, key, window_s=0.3):
        """(brakes, drivers): lists of (term, delta) sorted by |delta|; None when the target has no violations."""
        scene = BY_KEY[key].scene
        bad, clean, onsets = [], [], []
        prev = {}
        event = key in ("fall_rate", "foot_drop_rate")  # only violations are labelled
        for ei, t, sc, v in self.rows:
            if key in v and v[key]:
                bad.append((ei, t))
                if not prev.get(ei, False):
                    onsets.append((ei, t))
            elif key in v or (event and scene in ("any", sc)):
                clean.append((ei, t))
            prev[ei] = bool(v.get(key))
        if not bad or not clean:
            return None
        K = len(self.names)
        w = max(1, int(window_s / self.dt))

        def avg(points, field, per_s):
            acc = [0.0] * K
            for ei, t in points:
                row = self.envs[ei][field][t]
                for k in range(K):
                    acc[k] += row[k]
            return [a / len(points) / (self.dt if per_s else 1.0) for a in acc]

        pre = [(ei, s) for ei, t in onsets for s in range(max(0, t - w), t)] or onsets
        rb, rc = avg(bad, "terms", True), avg(clean, "terms", True)
        ab, ac = avg(pre, "term_adv", False), avg(clean, "term_adv", False)
        brakes = sorted(((self.names[k], rb[k] - rc[k]) for k in range(K)), key=lambda x: x[1])
        drivers = sorted(((self.names[k], ab[k] - ac[k]) for k in range(K)), key=lambda x: -x[1])
        return dict(n_bad=len(bad), n_onsets=len(onsets), n_clean=len(clean), brakes=brakes, drivers=drivers,
                    track=rc[self.names.index("track_lin_vel_xy_exp")])


# ---------------------------------------------------------------- report -------------------------------------------
def fmt(v, unit):
    if v != v:
        return "—"
    return f"{v:.2f}" if unit in ("m/s", "x", "s", "") else (f"{v * 100:.1f} cm" if unit == "m" else f"{v:.1f}°")


MIN_DRIVER = 0.01  # term_adv delta below this is noise (one recording, 32 envs)


def drivers(a, top):
    pos = [(n, d) for n, d in a["drivers"] if d > 0]
    floor = max(MIN_DRIVER, 0.2 * pos[0][1]) if pos else MIN_DRIVER
    return [(n, d) for n, d in pos if d >= floor][:top]


def brakes(a, top):
    """Penalty terms that charge the violation (negative delta), and positive terms whose pay drops there."""
    pen = [(n, d) for n, d in a["brakes"] if d < 0 and n not in POSITIVE][:top]
    lost = [(n, d) for n, d in a["brakes"] if d < 0 and n in POSITIVE][:top]
    return pen, lost


def verdict(a, top=3):
    pen, lost = brakes(a, 99)
    brake_sum = sum(d for _, d in pen)
    top_drv = drivers(a, top)
    notes = []
    if brake_sum > -0.05 * max(a["track"], 0.1):
        notes.append(f"제동 부족 (위반 순간 감점 합 {brake_sum:+.3f}/s, 속도 추종 {a['track']:+.2f}/s의 5 % 미만)")
    if lost:
        notes.append("위반 시 줄어드는 양의 보상: " + ", ".join(f"{n} {d:+.3f}" for n, d in lost[:2]))
    pos = [n for n, _ in top_drv if n in POSITIVE]
    if pos:
        notes.append("양의 보상이 위반 쪽으로 유인: " + ", ".join(pos))
    neg = [n for n, _ in top_drv if n not in POSITIVE]
    if neg:
        notes.append("다른 감점을 피하려고 위반을 택함(감점 회피 경로): " + ", ".join(neg))
    return "; ".join(notes) or "뚜렷한 단일 원인 없음"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mujoco", default="", help="comma-separated MuJoCo run dirs (state.jsonl)")
    ap.add_argument("--replay", required=True, help="comma-separated rl_replay recordings")
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    mj = mujoco_metrics([d for d in a.mujoco.split(",") if d]) if a.mujoco else {}
    steps = [Steps(p) for p in a.replay.split(",") if p]
    lines = [f"# gait 목표 진단", "", f"- MuJoCo: {len([d for d in a.mujoco.split(',') if d])} 런",
             f"- replay: {', '.join(s.name for s in steps)}", "",
             "| 목표 | 기준 | MuJoCo | Isaac | 판정 | 유인 항목 (term_adv Δ, 위반 직전) | 제동 항목 (감점 Δ /s, 위반 순간) | 해석 |",
             "|---|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        lim = (f"≥ {t.lo}" if t.lo is not None else "") + (" " if t.lo is not None and t.hi is not None else "") + \
              (f"≤ {t.hi}" if t.hi is not None else "")
        iv = [s.metric(t.key) for s in steps]
        iv = [v for v in iv if v == v]
        isaac = mean(iv) if t.stat in ("mean", "rate") else (max(iv) if iv else float("nan"))
        mjv = mj.get(t.key, float("nan"))
        ok = passes(t, mjv) and passes(t, isaac)
        drv = brk = why = ""
        if not ok:
            att = [s.attribute(t.key, a.window) for s in steps]
            att = [x for x in att if x]
            if att:
                best = max(att, key=lambda x: x["n_bad"])
                drv = ", ".join(f"{n} {d:+.3f}" for n, d in drivers(best, a.top)) or "(Δ ≥ 0.01 없음)"
                brk = ", ".join(f"{n} {d:+.3f}" for n, d in brakes(best, a.top)[0]) or "(없음)"
                why = verdict(best, a.top) + f" (위반 {best['n_bad']} 스텝 / 시작 {best['n_onsets']}회)"
            else:
                why = "replay에 위반 스텝 없음 (MuJoCo에서만 미달: Isaac–MuJoCo 차이)"
        lines.append(f"| {t.label} | {lim} | {fmt(mjv, t.unit)} | {fmt(isaac, t.unit)} | {'통과' if ok else '**미달**'} "
                     f"| {drv} | {brk} | {why} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
