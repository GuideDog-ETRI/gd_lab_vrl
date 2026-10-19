"""Online Top-5 checkpoint ranking for GAST-Nav (design D-9, user decision 2026-10-19).

Score over a rolling window of iterations:

    S = w_P*P + w_T*T + w_C*(1-C) + w_U*(1-U) + w_R*(1-R) + w_L*L

    P  progress ratio      mean clamp(v_path / v_cap, 0, 1) over NAV ticks with go
    T  stop success        stopped ticks / ticks that require a stop (go=0 or stair-end wait)
    C  termination rate    non-timeout episode ends / all episode ends
    U  unknown entry rate  NAV ticks whose body entered an unknown cell / NAV ticks
    R  reverse rotation    mean clamp(CCW accumulated deg / 3, 0, 1) over TURN ticks
    L  LEGACY success      LEGACY episodes ended by time-out / LEGACY episodes ended

A candidate is eligible only when it is in stage A, past top5_start_iteration,
at least top5_min_spacing iterations from every kept entry, and passes the gates
C <= max_termination_rate, U <= unknown_baseline + max_unknown_entry_increase,
L >= legacy_baseline - max_legacy_drop. A baseline < 0 means "measure it":
the mean over the first baseline_iterations iterations (A0 start, policy equal to
3906 because the adapters are zero-initialised). U is relative because the OBS1
near map leaves most cells around the body unknown (frozen 3906 enters unknown on
about half of NAV ticks, local smoke 2026-10-19).

Criteria live in a JSON file that is re-read whenever its mtime/size changes, so
they can be edited from outside while training runs. Invalid edits keep the last
valid values and never stop training.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

DEFAULTS = {
    "w_P": 0.35, "w_T": 0.15, "w_C": 0.20, "w_U": 0.10, "w_R": 0.10, "w_L": 0.10,
    "window": 50,
    "top5_start_iteration": 400,
    "top5_min_spacing": 100,
    "max_termination_rate": 0.09,
    "max_unknown_entry_increase": 0.05,
    "unknown_baseline": -1.0,
    "max_legacy_drop": 0.02,
    "legacy_baseline": -1.0,
    "baseline_iterations": 50,
}
INTEGER_KEYS = ("window", "top5_start_iteration", "top5_min_spacing", "baseline_iterations")


def validate(override: dict) -> dict:
    if not isinstance(override, dict):
        raise ValueError("criteria JSON must be an object")
    unknown = sorted(set(override) - set(DEFAULTS))
    if unknown:
        raise ValueError("unknown keys: " + ", ".join(unknown))
    result = dict(DEFAULTS)
    result.update(override)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key in INTEGER_KEYS:
            if int(value) != value or value < 1 and key != "top5_start_iteration" or value < 0:
                raise ValueError(f"{key} must be a non-negative integer (window/spacing/baseline >= 1)")
            result[key] = int(value)
        elif key in ("legacy_baseline", "unknown_baseline"):
            if value > 1:
                raise ValueError(f"{key} must be <= 1 (negative = measure)")
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    return result


class CriteriaFile:
    """Re-read criteria JSON only when the file's mtime or size changes."""

    def __init__(self, path):
        self.path = Path(path)
        self.criteria = dict(DEFAULTS)
        self.signature = None
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(DEFAULTS, indent=2) + "\n")

    def refresh(self):
        try:
            stat = self.path.stat()
        except OSError as exc:
            return dict(self.criteria), f"cannot read Top-5 criteria ({exc}); keeping last valid values"
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return dict(self.criteria), None
        self.signature = signature
        try:
            self.criteria = validate(json.loads(self.path.read_text()))
            return dict(self.criteria), f"reloaded Top-5 criteria from {self.path}"
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return dict(self.criteria), f"invalid Top-5 criteria ({exc}); keeping last valid values"


def rates(counts, terminations, timeouts):
    """counts: 10 summed counters from NavState.top5_take (all ranks)."""
    c = [float(x) for x in counts]
    div = lambda a, b: a / b if b > 0 else None
    return {
        "P": div(c[0], c[1]), "T": div(c[2], c[3]), "U": div(c[4], c[5]), "R": div(c[6], c[7]),
        "L": div(c[9], c[8]), "C": div(float(terminations), float(terminations + timeouts)),
    }


class OnlineTop5:
    def __init__(self, directory, criteria_path, k=5):
        self.directory = Path(directory)
        self.k = k
        self.criteria_file = CriteriaFile(criteria_path)
        self.history = deque(maxlen=10000)
        self.records = []  # (score, iteration, path)
        self.baseline = None
        self.unknown_baseline = None

    def _window(self, size):
        rows = list(self.history)[-size:]
        out = {}
        for key in ("P", "T", "C", "U", "R", "L"):
            values = [row[key] for row in rows if row[key] is not None]
            out[key] = sum(values) / len(values) if values else None
        return out

    def observe(self, iteration, row, stage):
        """Add one iteration of rates; return a dict describing the decision."""
        criteria, message = self.criteria_file.refresh()
        self.history.append(dict(row, iteration=iteration))
        n = criteria["baseline_iterations"]
        if len(self.history) >= n:
            first = list(self.history)[:n]
            mean = lambda key: (lambda v: sum(v) / len(v) if v else None)([r[key] for r in first if r[key] is not None])
            if self.baseline is None:
                self.baseline = mean("L")
            if self.unknown_baseline is None:
                self.unknown_baseline = mean("U")
        baseline = criteria["legacy_baseline"] if criteria["legacy_baseline"] >= 0 else self.baseline
        unknown_baseline = criteria["unknown_baseline"] if criteria["unknown_baseline"] >= 0 else self.unknown_baseline
        window = self._window(criteria["window"])
        decision = {"iteration": iteration, "stage": stage, "window": window, "legacy_baseline": baseline, "unknown_baseline": unknown_baseline,
                    "criteria_message": message, "eligible": False, "reasons": [], "score": None}
        reasons = decision["reasons"]
        if stage != "A":
            reasons.append("not stage A")
        if iteration < criteria["top5_start_iteration"]:
            reasons.append("before top5_start_iteration")
        if len(self.history) < criteria["window"]:
            reasons.append("window not full")
        if any(window[key] is None for key in window):
            reasons.append("missing rate: " + ",".join(k for k, v in window.items() if v is None))
        if baseline is None or unknown_baseline is None:
            reasons.append("baseline not measured")
        if not reasons:
            if window["C"] > criteria["max_termination_rate"]:
                reasons.append("termination rate")
            if window["U"] > unknown_baseline + criteria["max_unknown_entry_increase"]:
                reasons.append("unknown entry rate")
            if window["L"] < baseline - criteria["max_legacy_drop"]:
                reasons.append("legacy regression")
            if any(abs(iteration - it) < criteria["top5_min_spacing"] for _, it, _ in self.records):
                reasons.append("spacing")
        if not reasons:
            w = window
            score = (criteria["w_P"] * w["P"] + criteria["w_T"] * w["T"] + criteria["w_C"] * (1 - w["C"])
                     + criteria["w_U"] * (1 - w["U"]) + criteria["w_R"] * (1 - w["R"]) + criteria["w_L"] * w["L"])
            decision.update(eligible=True, score=score)
        return decision

    def consider(self, decision, save):
        """Keep the entry if it ranks in the top k. save(path) writes the checkpoint."""
        if not decision["eligible"]:
            return False
        score, iteration = decision["score"], decision["iteration"]
        if len(self.records) >= self.k and score <= min(r[0] for r in self.records):
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"nav_{iteration:06d}.pt"
        save(path)
        self.records.append((float(score), int(iteration), path))
        self.records.sort(reverse=True)
        for _, _, old in self.records[self.k:]:
            Path(old).unlink(missing_ok=True)
        self.records = self.records[: self.k]
        (self.directory / "top5.json").write_text(json.dumps(
            [{"score": s, "iteration": i, "path": str(p)} for s, i, p in self.records], indent=2) + "\n")
        return True

    def state(self):
        return {"records": [(s, i, str(p)) for s, i, p in self.records], "baseline": self.baseline, "unknown_baseline": self.unknown_baseline,
                "history": list(self.history)[-max(DEFAULTS["window"], 200):]}

    def load_state(self, state):
        self.records = [(s, i, Path(p)) for s, i, p in state.get("records", []) if Path(p).is_file()]
        self.baseline = state.get("baseline")
        self.unknown_baseline = state.get("unknown_baseline")
        for row in state.get("history", []):
            self.history.append(row)
