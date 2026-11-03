"""Stage C' online Top-5: the v2.5 score (narrow_top5_v25) + the gap "not reached" rate and its gate.

Why (t1f): E2 t1d model_299 stood in front of the gaps (not reached 41 % forward / 73 % backward, margin_t1 0-1 %) while
the online gate only saw platform_gap_termination_rate (0.13-0.26): a stop ends in the time-out, not in a termination,
so the run never showed it. gap_unreached_rate = macro over the gap levels of (no crossing credited and no base contact).
Gate max_gap_unreached_rate (folded into terrain_mean_level_pass like the v2 narrow gates).
"""
from __future__ import annotations

from collections import defaultdict

from gd_lab.gast_narrow import narrow_top5_v25 as v25
from gd_lab.rl import online_top5

GAP_FAMILY = "platform_gap"


def gap_unreached_rate(episodes):
    strata = defaultdict(list)
    for row in episodes:
        if row["family"] == GAP_FAMILY:
            strata[row["level"]].append(float(not row["gap_success"] and not row["base_contact"]))
    rates = [sum(rows) / len(rows) for rows in strata.values() if rows]
    return sum(rates) / len(rates) if rates else None


def score_episodes(episodes):
    result = v25.score_episodes(episodes)
    if result is not None:
        result["criteria_metrics"]["gap_unreached_rate"] = gap_unreached_rate(episodes)
    return result


def gates(candidate, reference, criteria=None):
    result = v25.gates(candidate, reference, criteria)
    criteria = criteria or {}
    if "max_gap_unreached_rate" in criteria:
        value = candidate["criteria_metrics"].get("gap_unreached_rate")
        ok = value is None or value <= criteria["max_gap_unreached_rate"]
        result["gap_unreached_pass"] = ok
        result["terrain_mean_level_pass"] = result["terrain_mean_level_pass"] and ok
    return result


def install():
    v25.install()
    online_top5.score_episodes = score_episodes
    online_top5.gates = gates
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("max_gap_unreached_rate", 1.0)
