"""Load repeatable experiment arms as Hydra defaults, before explicit CLI overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

_EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "configs" / "experiment"


def training_arm_overrides(arm: str | None, *, play: bool = False) -> list[str]:
    """Resolve TRAIN_ARM without importing IsaacLab or changing the base recipe.

    Play/Gamepad retain their own command and curriculum settings while using
    the selected arm's timing, gains, and checkpoint directory.
    """
    if arm is None:
        return []
    if arm not in {"1", "2", "3", "4"}:
        raise ValueError(f"TRAIN_ARM must be 1, 2, 3, or 4; got {arm!r}")
    path = _EXPERIMENT_DIR / f"arm_{arm}.yaml"
    overrides = yaml.safe_load(path.read_text())
    if not isinstance(overrides, list) or not all(isinstance(item, str) and "=" in item for item in overrides):
        raise ValueError(f"Expected a list of Hydra overrides in {path}")
    if play:
        overrides = [item for item in overrides if not item.startswith(("env.commands.", "env.curriculum."))]
    return overrides
