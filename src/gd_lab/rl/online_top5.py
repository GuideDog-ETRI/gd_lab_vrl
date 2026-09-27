"""Online, rollout-only checkpoint ranking for the Vision RL teacher."""

from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

WEIGHTS = {"G": 0.40, "B": 0.25, "P": 0.15, "T": 0.10, "E": 0.05, "Q": 0.05}
DEFAULT_MIN_PLATFORM_GAP_MEAN_LEVEL = 8.0


def score_episodes(episodes: list[dict]) -> dict | None:
    """Macro-average episode metrics over observed (family, level) strata.

    G is averaged over gap strata only. A batch without a gap episode cannot
    establish the safety gate and is not eligible for ranking.
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["family"], int(episode["level"]))].append(episode)
    gap_groups = {key: rows for key, rows in groups.items() if key[0] == "platform_gap"}
    if not gap_groups:
        return None

    def macro(key: str, strata: dict) -> float:
        return sum(sum(row[key] for row in rows) / len(rows) for rows in strata.values()) / len(strata)

    components = {
        "G": macro("gap_success", gap_groups),
        "B": 1.0 - macro("base_contact", groups),
        "P": macro("progress", groups),
        "T": macro("tracking", groups),
        "E": macro("energy", groups),
        "Q": macro("return", groups),
    }
    score = sum(WEIGHTS[key] * components[key] for key in WEIGHTS)
    counts = {f"{family}:{level}": len(rows) for (family, level), rows in sorted(groups.items())}
    curriculum = {family: sum(level * len(rows) for (name, level), rows in groups.items() if name == family) /
                  sum(len(rows) for (name, _), rows in groups.items() if name == family)
                  for family in sorted({name for name, _ in groups})}
    return {"score": score, "components": components, "sample_counts": counts,
            "curriculum_snapshot": {"mean_level_by_family": curriculum}, "score_is_online_proxy": True}


def gates(candidate: dict, reference: dict | None) -> dict:
    """Reject >2 pp gap regression or >1 pp base-contact worsening."""
    g = candidate["components"]["G"]
    b = candidate["components"]["B"]
    ref_g = reference["components"]["G"] if reference else None
    ref_b = reference["components"]["B"] if reference else None
    return {"gap_success_pass": ref_g is None or g >= ref_g - 0.02,
            "base_contact_pass": ref_b is None or b >= ref_b - 0.01,
            "reference_iteration": reference["iteration"] if reference else None}

def meets_minimum_gap_difficulty(candidate: dict, minimum: float) -> bool:
    """Require the online rollout to represent sufficiently hard gap terrain."""
    level = candidate["curriculum_snapshot"]["mean_level_by_family"].get("platform_gap")
    return level is not None and level >= minimum



def select_top5(entries: list[dict], candidate: dict, spacing: int = 100) -> list[dict]:
    """Choose up to five spaced candidates; better neighbors replace weaker ones."""
    if spacing < 1:
        raise ValueError("spacing must be positive")
    ranked = sorted([*entries, candidate], key=lambda row: (-row["score"], row["iteration"]))
    selected = []
    for row in ranked:
        if all(abs(row["iteration"] - other["iteration"]) >= spacing for other in selected):
            selected.append(row)
        if len(selected) == 5:
            break
    return selected


def rank_and_save_top5(
    rank: int, episodes: list[dict], directory: Path, iteration: int, spacing: int, save_model,
    min_platform_gap_mean_level: float = DEFAULT_MIN_PLATFORM_GAP_MEAN_LEVEL,
) -> dict:
    """Rank a gathered rollout and write its checkpoint on rank zero only."""
    status = {"selected": False, "saved": False, "error": None}
    if rank != 0:
        return status
    try:
        candidate = score_episodes(episodes)
        if candidate is None:
            return status
        directory = Path(directory)
        entries = json.loads((directory / "leaderboard.json").read_text())["entries"] if directory.exists() else []
        candidate["iteration"] = iteration
        candidate["eligibility"] = {"minimum_platform_gap_mean_level": min_platform_gap_mean_level,
                                    "platform_gap_mean_level_pass": meets_minimum_gap_difficulty(
                                        candidate, min_platform_gap_mean_level)}
        if not candidate["eligibility"]["platform_gap_mean_level_pass"]:
            return status
        candidate["gate_results"] = gates(candidate, entries[0] if entries else None)
        if not all(candidate["gate_results"][key] for key in ("gap_success_pass", "base_contact_pass")):
            return status
        selected = select_top5(entries, candidate, spacing)
        status["selected"] = any(row["iteration"] == iteration for row in selected)
        if status["selected"]:
            rotate_top5(directory, selected, iteration, save_model)
            status["saved"] = True
    except Exception as exc:
        status["error"] = f"Online Top-5 ranking failed at iteration {iteration}: {exc}"
    return status


def _exchange_directories(a: Path, b: Path) -> None:
    # Linux renameat2(RENAME_EXCHANGE) switches the whole leaderboard in one
    # atomic namespace operation; readers see either complete generation.
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(a), -100, os.fsencode(b), 2)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def rotate_top5(directory: Path, entries: list[dict], candidate_iteration: int, save_model) -> None:
    """Stage a complete generation and atomically publish it.

    Existing checkpoint files are hard-linked into the staging directory; only
    the new model is serialized. The previous generation is removed afterward.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".best_top5-", dir=directory.parent))
    try:
        old = {}
        if directory.exists():
            metadata = json.loads((directory / "leaderboard.json").read_text())
            old = {row["iteration"]: directory / f'{row["iteration"]}_top{rank}.pt'
                   for rank, row in enumerate(metadata["entries"], 1)}
        for rank, row in enumerate(entries, 1):
            dest = stage / f'{row["iteration"]}_top{rank}.pt'
            if row["iteration"] == candidate_iteration:
                save_model(str(dest))
            else:
                os.link(old[row["iteration"]], dest)
        (stage / "leaderboard.json").write_text(json.dumps({"entries": entries}, indent=2) + "\n")
        if directory.exists():
            _exchange_directories(stage, directory)
            shutil.rmtree(stage)
        else:
            os.replace(stage, directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_episode(row: dict) -> bool:
    return all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in
               ("gap_success", "base_contact", "progress", "tracking", "energy", "return"))
