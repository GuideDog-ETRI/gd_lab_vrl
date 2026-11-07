"""Gait targets (docs/design/GAIT_V3_DESIGN.md section 2, docs/design/DOG_GAIT_REFERENCE.md section 4).

One entry per metric: how it is measured in MuJoCo (state.jsonl, 10 Hz) and in an rl_replay recording (100 Hz), the
limit, and the scene it applies to. ``lo`` / ``hi`` are inclusive bounds; None = unbounded.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    key: str
    label: str            # Korean label for the report
    unit: str
    lo: float | None
    hi: float | None
    scene: str            # flat | stairs | gap_fwd | gap_back | any
    stat: str             # mean | p90 | max | rate (fraction of events / runs)


TARGETS = [
    Target("nose_mean", "평지 머리 들림 평균", "deg", None, 2.0, "flat", "mean"),
    Target("nose_p90", "평지 머리 들림 p90", "deg", None, 4.0, "flat", "p90"),
    Target("nose_max", "평지 머리 들림 최대", "deg", None, 8.0, "flat", "max"),
    Target("nose_band_p90", "계단·경사 자세 이탈 p90 (수평~경사 밖)", "deg", None, 3.0, "stairs", "p90"),
    Target("td_speed_p90", "평지 착지 속도 p90", "m/s", None, 0.6, "flat", "p90"),
    Target("td_force_p90", "평지 착지 수직력 p90 (정지력 배수)", "x", None, 2.0, "flat", "p90"),
    Target("lift_max_p90", "평지 스윙 발 높이 최대 p90", "m", 0.05, 0.10, "flat", "p90"),
    Target("swing_mean", "평지 스윙 시간 평균", "s", 0.24, 0.45, "flat", "mean"),
    Target("gap_trail_lag_p90", "전진 갭 뒤따르는 발 이륙 시 뒤처짐 p90", "m", None, 0.20, "gap_fwd", "p90"),
    Target("gap_swing_peak_p90", "전진 갭 뒤따르는 발 스윙 최고 속도 p90", "m/s", None, 2.5, "gap_fwd", "p90"),
    Target("gap_td_speed_p90", "전진 갭 뒤따르는 발 착지 속도 p90", "m/s", None, 1.0, "gap_fwd", "p90"),
    Target("back_brace_p90", "후진 갭 앞선 발(뒷발) 버팀 시간 p90 (길수록 안전, 6566 0.95)", "s", 0.8, None, "gap_back", "p90"),
    Target("fall_rate", "넘어짐 비율", "", None, 0.01, "any", "rate"),
]

BY_KEY = {t.key: t for t in TARGETS}


def passes(t: Target, value: float) -> bool:
    if value != value:  # nan: not measured
        return True
    return (t.lo is None or value >= t.lo) and (t.hi is None or value <= t.hi)
