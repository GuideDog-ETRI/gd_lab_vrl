"""Online Top-5 for GAST-Narrow v1: the rl.online_top5 machinery with a narrow-aware episode row and score.

Score = 0.40 S + 0.15 W + 0.15 G + 0.20 B + 0.10 T, macro-averaged over (family, level) strata:
  S  narrow passage rate (exit line passed, no trunk contact)          narrow strata
  W  touch-free rate (robot points within 5 mm of a wall < 0.1 s)       narrow strata
  G  clean gap crossing rate                                            platform_gap strata (retention)
  B  1 - base-contact termination rate                                  all strata
  T  command tracking                                                   retention strata
Gate keys are the ones rank_and_save_top5 requires; ``min_terrain_mean_level`` applies to the narrow mean level.
"""
from __future__ import annotations

import math
import types
from collections import defaultdict

import torch

from gd_lab.rl import online_top5

WEIGHTS = {"S": 0.40, "W": 0.15, "G": 0.15, "B": 0.20, "T": 0.10}
NARROW_PREFIX = "narrow_"


def install_episode_collector(env, family_columns):
    raw = env.unwrapped
    records: list[dict] = []
    original_reset = raw._reset_idx
    column_family = {column: family for family, columns in family_columns.items() for column in columns}

    def record_and_reset(self, ids):
        terrain = self.scene.terrain
        types_ = terrain.terrain_types[ids].detach().cpu().tolist()
        levels = terrain.terrain_levels[ids].detach().cpu().tolist()
        base = self.termination_manager.get_term("base_contact")[ids].detach().cpu().tolist()
        position = self.scene["robot"].data.root_pos_w[ids, :2]
        origins = self.scene.env_origins[ids, :2]
        distance = torch.linalg.vector_norm(position - origins, dim=1).detach().cpu().tolist()
        command = self.command_manager.get_term("base_velocity")
        commanded = command.cmd_dist_integral[ids].detach().cpu().tolist()
        durations = (self.episode_length_buf[ids].float() * self.step_dt).clamp_min(self.step_dt)
        duration = durations.detach().cpu().tolist()
        sums = {name: value[ids].detach().cpu().tolist() for name, value in self.reward_manager._episode_sums.items()}
        crossing = self.reward_manager.get_term_cfg("platform_gap_crossing").func
        achieved = crossing.achieved[ids].any(dim=1).detach().cpu().tolist()
        bypassed = crossing.bypassed[ids].detach().cpu().tolist()
        monitor = self.reward_manager.get_term_cfg("platform_gap_monitor").func
        clean = monitor.tracker.paid[ids].any(dim=1).detach().cpu().tolist()
        narrow = self.reward_manager.get_term_cfg("narrow_monitor").func
        passed = narrow.success[ids].detach().cpu().tolist()
        touch = narrow.touch_time[ids].detach().cpu().tolist()
        clearance = narrow.min_clearance[ids].detach().cpu().tolist()
        max_level = max(1, self.scene.terrain.cfg.terrain_generator.num_rows - 1)
        for index, column in enumerate(types_):
            family = column_family.get(column)
            if family is None:
                continue
            seconds = duration[index]
            track = sum(sums.get(name, [0.0] * len(types_))[index] for name in
                        ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"))
            tracking = min(1.0, max(0.0, track / (1.8 * seconds)))
            penalties = sum(abs(sums.get(name, [0.0] * len(types_))[index]) for name in
                            ("dof_torques_l2", "joint_power", "action_rate_l2", "smoothness"))
            total_return = sum(values[index] for values in sums.values())
            row = {"family": family, "level": int(levels[index]),
                   "gap_success": float(bool(achieved[index]) and not bypassed[index] and not base[index]),
                   "gap_clean_success": float(bool(clean[index]) and not bypassed[index] and not base[index]),
                   "base_contact": float(bool(base[index])),
                   "progress": min(1.0, max(0.0, distance[index] / max(commanded[index], 0.1))),
                   "tracking": tracking, "energy": 1.0 / (1.0 + penalties / seconds),
                   "return": 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -total_return / seconds /
                                                                  (1.0 + levels[index] / max_level)))))}
            if family.startswith(NARROW_PREFIX):
                row.update(passage=float(bool(passed[index]) and not base[index]),
                           touch_free=float(touch[index] < 0.1),
                           min_clearance=float(min(clearance[index], 1.0)))
            if online_top5.validate_episode(row):
                records.append(row)
        return original_reset(ids)

    raw._reset_idx = types.MethodType(record_and_reset, raw)
    return records


def score_episodes(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["family"], int(episode["level"]))].append(episode)
    narrow = {k: v for k, v in groups.items() if k[0].startswith(NARROW_PREFIX)}
    retention = {k: v for k, v in groups.items() if not k[0].startswith(NARROW_PREFIX)}
    gap = {k: v for k, v in groups.items() if k[0] == "platform_gap"}
    if not narrow or not gap:
        return None

    def macro(key, strata):
        return sum(sum(row[key] for row in rows) / len(rows) for rows in strata.values()) / len(strata)

    components = {"S": macro("passage", narrow), "W": macro("touch_free", narrow),
                  "G": macro("gap_clean_success", gap), "B": 1.0 - macro("base_contact", groups),
                  "T": macro("tracking", retention) if retention else 0.0}
    score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    curriculum = {family: sum(level * len(rows) for (name, level), rows in groups.items() if name == family) /
                  sum(len(rows) for (name, _), rows in groups.items() if name == family)
                  for family in sorted({name for name, _ in groups})}
    narrow_rows = [row for rows in narrow.values() for row in rows]
    stair = {k: v for k, v in groups.items() if k[0].startswith(online_top5.STAIR_FAMILY_PREFIX)}
    metrics = {
        "narrow_mean_level": sum(row["level"] for row in narrow_rows) / len(narrow_rows),
        "narrow_passage_rate": components["S"], "narrow_touch_free_rate": components["W"],
        "narrow_min_clearance_mean": macro("min_clearance", narrow),
        "base_contact_rate": macro("base_contact", groups),
        "narrow_termination_rate": macro("base_contact", narrow),
        "platform_gap_termination_rate": macro("base_contact", gap),
        "stairs_termination_rate": macro("base_contact", stair) if stair else None,
        "gap_crossing_rate": macro("gap_success", gap), "gap_clean_crossing_rate": components["G"],
    }
    counts = {f"{family}:{level}": len(rows) for (family, level), rows in sorted(groups.items())}
    return {"score": score, "components": components, "criteria_metrics": metrics, "sample_counts": counts,
            "gap_success_definition": "clean_crossing", "score_definition": WEIGHTS,
            "curriculum_snapshot": {"mean_level_by_family": curriculum}, "score_is_online_proxy": True}


def gates(candidate, reference, criteria=None):
    criteria = dict(online_top5.DEFAULT_TOP5_CRITERIA if criteria is None else criteria)
    c, m = candidate["components"], candidate["criteria_metrics"]
    ref = reference["components"] if reference else None
    stairs = m.get("stairs_termination_rate")
    return {"gap_success_pass": ref is None or c["G"] >= ref["G"] - criteria["max_gap_success_regression"],
            "base_contact_pass": ref is None or c["B"] >= ref["B"] - criteria["max_base_contact_regression"],
            "terrain_mean_level_pass": m["narrow_mean_level"] >= criteria["min_terrain_mean_level"],
            "base_contact_rate_pass": m["base_contact_rate"] <= criteria["max_base_contact_rate"],
            "platform_gap_termination_pass":
                m["platform_gap_termination_rate"] <= criteria["max_platform_gap_termination_rate"],
            "stairs_termination_pass": stairs is None or stairs <= criteria["max_stairs_termination_rate"],
            "reference_iteration": reference["iteration"] if reference else None}


def install():
    """Patch the module globals rank_and_save_top5 / DreamwaqRunner use (this process only)."""
    from gd_lab.rl import runner
    runner.install_episode_collector = install_episode_collector
    online_top5.score_episodes = score_episodes
    online_top5.gates = gates
