"""Validated live settings for GAST student distillation.

Only optimizer/loss scheduling and checkpoint cadence can be changed while a
run is alive. Simulator topology, camera transport, teacher and model contracts
are intentionally fixed at startup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


LIVE_DEFAULTS: dict[str, int | float] = {
    "learning_rate": 1.0e-3,
    "student_warmup": 1000,
    "student_ramp": 4000,
    "checkpoint_interval": 1000,
    "bptt_steps": 16,
}

IMMUTABLE_SETTINGS = (
    "iterations", "num_envs", "seed", "teacher_checkpoint", "student_arch",
    "hazard_loss_coef",
    "camera_profile", "camera_interval_ms", "camera_delay_ms", "camera_drop_prob",
    "top5_start_iteration", "top5_keep", "top5_smoothing_windows",
    "top5_min_visible_fraction", "top5_min_visible_sample_fraction",
    "top5_min_hazard_supervised_fraction",
)


def _validate(settings: dict[str, Any]) -> dict[str, int | float]:
    unknown = set(settings) - set(LIVE_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported live setting(s): {', '.join(sorted(unknown))}")
    merged = dict(LIVE_DEFAULTS)
    merged.update(settings)
    for key in ("student_warmup", "student_ramp", "checkpoint_interval", "bptt_steps"):
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if merged["student_warmup"] < 0:
        raise ValueError("student_warmup must be nonnegative")
    for key in ("student_ramp", "checkpoint_interval", "bptt_steps"):
        if merged[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("learning_rate",):
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if key == "learning_rate" and value <= 0:
            raise ValueError("learning_rate must be positive")
    return merged


class StudentLiveConfig:
    """Poll a JSON file and retain the last valid settings on malformed edits."""

    def __init__(self, path: str | Path, initial: dict[str, Any]):
        self.path = Path(path)
        self.current = _validate(initial)
        self._signature: tuple[int, int] | None = None
        if not self.path.exists():
            self._write(self.current)

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.path)
        stat = self.path.stat()
        self._signature = (stat.st_mtime_ns, stat.st_size)

    def poll(self) -> tuple[dict[str, int | float], dict[str, int | float], str | None]:
        """Return (settings, changed-values, warning); invalid edits are ignored."""
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self._signature:
                return dict(self.current), {}, None
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("root must be a JSON object")
            proposed = _validate(raw)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            return dict(self.current), {}, f"ignored invalid live config {self.path}: {exc}"

        changed = {key: value for key, value in proposed.items()
                   if value != self.current[key]}
        self.current = proposed
        self._signature = signature
        return dict(self.current), changed, None
