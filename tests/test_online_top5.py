"""Pure scoring and filesystem tests; no simulator or GPU is required."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src/gd_lab/rl"))
from online_top5 import (
    gates,
    meets_minimum_gap_difficulty,
    rank_and_save_top5,
    rotate_top5,
    score_episodes,  # noqa: E402
    select_top5,
)


def episode(family="platform_gap", level=0, **changes):
    row = {"family": family, "level": level, "gap_success": 1.0, "base_contact": 0.0,
           "progress": 0.5, "tracking": 0.5, "energy": 0.5, "return": 0.5}
    return {**row, **changes}


def candidate(iteration, score):
    return {
        "iteration": iteration, "score": score, "components": {"G": 0.8, "B": 0.9},
        "criteria_metrics": {
            "mean_terrain_level": 9.0, "base_contact_rate": 0.0,
            "platform_gap_termination_rate": 0.0, "stairs_termination_rate": 0.0,
        },
        "curriculum_snapshot": {}, "sample_counts": {}, "gate_results": {}, "score_is_online_proxy": True,
    }


def test_score_macro_averages_family_and_level():
    result = score_episodes([episode(), episode(gap_success=0.0), episode(level=1, gap_success=1.0),
                             episode(family="stairs", level=0, base_contact=1.0)])
    assert result["components"]["G"] == 0.75
    assert result["components"]["B"] == pytest.approx(2 / 3)
    assert result["score"] == sum(weight * value for weight, value in
                                   zip((0.4, 0.25, 0.15, 0.10, 0.05, 0.05), result["components"].values(), strict=True))
    assert result["sample_counts"] == {"platform_gap:0": 2, "platform_gap:1": 1, "stairs:0": 1}
    assert score_episodes([episode(family="stairs")]) is None


def test_gates_exact_thresholds():
    reference = candidate(100, 0.5)
    passing = candidate(200, 0.7)
    passing["components"] = {"G": 0.78, "B": 0.89}
    assert gates(passing, reference)["gap_success_pass"]
    assert gates(passing, reference)["base_contact_pass"]
    passing["components"]["G"] = 0.779
    passing["components"]["B"] = 0.889
    assert not gates(passing, reference)["gap_success_pass"]
    assert not gates(passing, reference)["base_contact_pass"]



def test_absolute_terrain_and_termination_gates():
    rows = [episode(family="platform_gap", level=9, base_contact=0.06),
            episode(family="pyramid_stairs", level=9, base_contact=0.06)]
    scored = score_episodes(rows)
    results = gates(scored, None)
    assert results["terrain_mean_level_pass"]
    assert results["base_contact_rate_pass"]
    assert results["platform_gap_termination_pass"]
    assert results["stairs_termination_pass"]
def test_gap_difficulty_gate_requires_mean_level_eight():
    low = score_episodes([episode(level=7)])
    high = score_episodes([episode(level=8)])
    assert not meets_minimum_gap_difficulty(low, 8.0)
    assert meets_minimum_gap_difficulty(high, 8.0)


def test_ranking_spacing_and_limit():
    entries = [candidate(i, 1 - i / 1000) for i in (100, 200, 300, 400, 500)]
    assert [x["iteration"] for x in select_top5(entries, candidate(250, 0.5))] == [100, 200, 300, 400, 500]
    assert [x["iteration"] for x in select_top5(entries, candidate(250, 1.1))] == [250, 100, 400, 500]
    assert [x["iteration"] for x in select_top5(entries, candidate(600, 1.1))] == [600, 100, 200, 300, 400]


def test_atomic_rotation_and_checkpoint_bytes(tmp_path: Path):
    directory = tmp_path / "best_top5"
    first = [candidate(100, 0.8)]
    rotate_top5(directory, first, 100, lambda path: Path(path).write_bytes(b"checkpoint-100"))
    # Seed a second candidate through normal rotation; both remain loadable bytes.
    second = [candidate(200, 0.9), candidate(100, 0.8)]
    rotate_top5(directory, second, 200, lambda path: Path(path).write_bytes(b"checkpoint-200"))
    assert sorted(p.name for p in directory.glob("*.pt")) == ["100_top2.pt", "200_top1.pt"]
    assert (directory / "100_top2.pt").read_bytes() == b"checkpoint-100"
    assert (directory / "200_top1.pt").read_bytes() == b"checkpoint-200"
    assert [row["iteration"] for row in json.loads((directory / "leaderboard.json").read_text())["entries"]] == [200, 100]


def test_failed_stage_keeps_published_generation(tmp_path: Path):
    directory = tmp_path / "best_top5"
    rotate_top5(directory, [candidate(100, 0.8)], 100, lambda path: Path(path).write_bytes(b"original"))

    def fail(_path):
        raise OSError("save failed")

    with pytest.raises(OSError, match="save failed"):
        rotate_top5(directory, [candidate(200, 0.9), candidate(100, 0.8)], 200, fail)
    assert sorted(p.name for p in directory.iterdir()) == ["100_top1.pt", "leaderboard.json"]
    assert (directory / "100_top1.pt").read_bytes() == b"original"


def test_only_rank_zero_reads_and_writes_top5(tmp_path: Path):
    directory = tmp_path / "best_top5"
    directory.mkdir()
    (directory / "leaderboard.json").write_text("invalid json")
    saves = []

    def save(path):
        saves.append(path)
        Path(path).write_bytes(b"checkpoint")

    status = rank_and_save_top5(1, [episode()], directory, 100, 100, save, 0)
    assert status == {"selected": False, "saved": False, "error": None}
    assert saves == []
    assert (directory / "leaderboard.json").read_text() == "invalid json"
    assert sorted(p.name for p in directory.iterdir()) == ["leaderboard.json"]

    failure = rank_and_save_top5(0, [episode()], directory, 100, 100, save, 0)
    assert failure["error"].startswith("Online Top-5 ranking failed at iteration 100:")
    assert not failure["saved"]
    assert saves == []

    (directory / "leaderboard.json").unlink()
    directory.rmdir()
    success = rank_and_save_top5(0, [episode(family="platform_gap", level=9),
                                     episode(family="pyramid_stairs", level=9)], directory, 100, 100, save, 0)
    assert success == {"selected": True, "saved": True, "error": None}
    assert len(saves) == 1
