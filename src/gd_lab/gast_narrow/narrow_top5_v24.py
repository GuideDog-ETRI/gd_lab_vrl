"""Online Top-5 for GAST-Narrow v2.4: the v2 score over the non-turn episodes plus R, the turn success rate.

Score = 0.20 P + 0.15 H + 0.10 N + 0.05 L + 0.10 W + 0.15 G + 0.10 B + 0.05 T + 0.10 R
  R  turn tiles: (turned with drift <= 0.10 m, or stood when the turn was not possible) and no breach,
     macro over turn (family, level) strata.
P/H/N/L/W/G/B/T are v2's, computed without the turn episodes (their pass/hold flags mean turn / stand there).
Gates are v2's (optional hold / breach limits included).
"""
from __future__ import annotations

from collections import defaultdict

from gd_lab.gast_narrow import narrow_top5_v2 as v2
from gd_lab.rl import online_top5

TURN_PREFIX = "narrow_turn"
WEIGHTS = {"P": 0.20, "H": 0.15, "N": 0.10, "L": 0.05, "W": 0.10, "G": 0.15, "B": 0.10, "T": 0.05, "R": 0.10}


def score_episodes(episodes):
    turn = [row for row in episodes if row["family"].startswith(TURN_PREFIX)]
    result = v2.score_episodes([row for row in episodes if not row["family"].startswith(TURN_PREFIX)])
    if result is None:
        return None
    strata = defaultdict(list)
    for row in turn:
        strata[(row["family"], row["level"])].append(row)

    def macro(value, select=lambda row: True):
        rates = [sum(value(row) for row in chosen) / len(chosen)
                 for chosen in ([row for row in rows if select(row)] for rows in strata.values()) if chosen]
        return sum(rates) / len(rates) if rates else None

    success = lambda row: float((row["passage"] or row["hold"]) and not row["breach"])  # noqa: E731
    rotate = lambda row: row["slack"] > 0  # noqa: E731
    components = dict(result["components"])
    r = macro(success)
    components["R"] = 0.0 if r is None else r
    result["components"] = components
    result["score"] = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    result["score_definition"] = WEIGHTS
    result["criteria_metrics"].update(
        turn_success_rate=r, turn_rotate_rate=macro(lambda row: row["passage"], rotate),
        turn_stop_rate=macro(lambda row: row["hold"], lambda row: not rotate(row)),
        turn_breach_rate=macro(lambda row: row["breach"], lambda row: not rotate(row)), turn_samples=len(turn))
    return result


def install():
    v2.install()
    online_top5.score_episodes = score_episodes
