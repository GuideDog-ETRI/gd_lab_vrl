"""BAVRL adapter for shared rollout-only Top-5 selection (no extra simulation)."""
import json
from pathlib import Path

from gd_lab.rl.online_top5 import (
    Top5CriteriaReloader, gates, meets_minimum_gap_difficulty,
    rank_and_save_top5, score_episodes,
)


class OnlineQuality:
    def __init__(self, output, criteria_file):
        self.output = Path(output)
        self.directory = self.output / "best_top5"
        self.criteria = Top5CriteriaReloader(criteria_file)

    def update(self, iteration, episodes, save_model):
        criteria, message = self.criteria.refresh()
        candidate = score_episodes(episodes)
        reference = None
        leaderboard = self.directory / "leaderboard.json"
        if leaderboard.exists():
            entries = json.loads(leaderboard.read_text())["entries"]
            reference = entries[0] if entries else None
        failed = []
        if candidate is None:
            failed.append("no_completed_gap_episode")
        else:
            if not meets_minimum_gap_difficulty(candidate, criteria["min_platform_gap_mean_level"]):
                failed.append("platform_gap_mean_level_pass")
            failed.extend(k for k, v in gates(candidate, reference, criteria).items()
                          if k.endswith("pass") and not v)
        status = rank_and_save_top5(0, episodes, self.directory, iteration,
                                   criteria["top5_min_spacing"], save_model, criteria=criteria)
        row = {"iteration": iteration, "episodes": len(episodes), "candidate": candidate,
               "failed_gates": failed, "criteria": criteria, "criteria_message": message,
               "score_is_online_proxy": True, **status}
        with (self.output / "quality_samples.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if status["error"]:
            raise RuntimeError(status["error"])
        return row
