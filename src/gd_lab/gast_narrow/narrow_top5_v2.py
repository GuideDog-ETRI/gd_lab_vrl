"""Online Top-5 for GAST-Narrow v2 (same machinery as v1, v2 episode row and score).

Score = 0.25 P + 0.15 H + 0.10 N + 0.05 L + 0.10 W + 0.15 G + 0.15 B + 0.05 T
  P  passable episodes: exit passed, no trunk contact               macro over narrow (family, level) strata
  H  blocked episodes: stood 1.5 s before the line, no breach         macro over narrow strata with blocked rows
  N  tight passable (slack w - (robot + 2 m) < 0.10 m): pass rate     pooled
  L  passable L-bends: pass rate                                      macro over bend strata
  W  touch-free rate, G clean gap crossing, B 1 - base contact, T tracking (as v1)
A component without samples in the window counts 0. Gates are v1's; with min_narrow_hold_rate / max_narrow_breach_rate
in the criteria (lineage 16-node run) the blocked-passage hold and breach rates also gate the narrow pass.
"""
from __future__ import annotations

import math
import types
from collections import defaultdict

import json
import os

import torch

from gd_lab.gast_narrow import narrow_top5 as v1
from gd_lab.rl import online_top5

WEIGHTS = {"P": 0.25, "H": 0.15, "N": 0.10, "L": 0.05, "W": 0.10, "G": 0.15, "B": 0.15, "T": 0.05}
NARROW_PREFIX = v1.NARROW_PREFIX
TIGHT_SLACK = 0.10


def install_episode_collector(env, family_columns):
    raw = env.unwrapped
    records: list[dict] = []
    original_reset = raw._reset_idx
    column_family = {column: family for family, columns in family_columns.items() for column in columns}
    dump = os.environ.get("GD_LAB_TOP5_DUMP")

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
        passed = narrow.passed[ids].detach().cpu().tolist()
        held = narrow.held[ids].detach().cpu().tolist()
        breached = narrow.breached[ids].detach().cpu().tolist()
        touch = narrow.touch_time[ids].detach().cpu().tolist()
        clearance = narrow.min_clearance[ids].detach().cpu().tolist()
        from gd_lab.gast_narrow.mdp_v2 import narrow_episode  # Isaac modules: import after the app starts
        episode = narrow_episode(self)
        blocked = episode.blocked[ids].detach().cpu().tolist()
        slack = episode.slack[ids].detach().cpu().tolist()
        margin = episode.margin[ids].detach().cpu().tolist()
        turn = getattr(self, "_narrow_turn", None)
        turnable = turn.turnable[ids].detach().cpu().tolist() if turn is not None else None
        term_flags = {name: self.termination_manager.get_term(name)[ids].detach().cpu().tolist()
                      for name in self.termination_manager.active_terms} if dump else {}
        max_level = max(1, self.scene.terrain.cfg.terrain_generator.num_rows - 1)
        lengths = self.episode_length_buf[ids].detach().cpu().tolist()
        for index, column in enumerate(types_):
            family = column_family.get(column)
            if family is None or lengths[index] == 0:  # a full reset (start of training): no episode to record
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
                ok = not base[index]
                row.update(blocked=float(bool(blocked[index])),
                           passage=float(bool(passed[index]) and ok),
                           hold=float(bool(held[index]) and not breached[index] and ok),
                           breach=float(bool(breached[index])),
                           slack=float(slack[index]), margin=float(margin[index]),
                           touch_free=float(touch[index] < 0.1),
                           min_clearance=float(min(clearance[index], 1.0)))
            if online_top5.validate_episode(row):
                records.append(row)
                if dump and family.startswith("narrow_turn"):  # GD_LAB_TOP5_DUMP=path: turn rows as JSON lines
                    row_dump = dict(row, seconds=seconds, held=bool(held[index]), breached=bool(breached[index]),
                                    base=bool(base[index]), turnable=bool(turnable[index]) if turnable else None,
                                    terms=[name for name, flags in term_flags.items() if flags[index]],
                                    drift=float(turn.drift[ids][index]) if turn is not None else None,
                                    turned_deg=float(math.degrees(turn.turned[ids][index])) if turn is not None else None,
                                    blocked_v2=bool(blocked[index]),
                                    hold_max=float(turn.hold_max[ids][index]) if turn is not None else None,
                                    diag=turn.diag[ids][index].tolist() if turn is not None else None)
                    with open(dump, "a") as f:
                        f.write(json.dumps(row_dump) + "\n")
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

    def macro(key, strata, select=lambda row: True):
        rates = []
        for rows in strata.values():
            chosen = [row for row in rows if select(row)]
            if chosen:
                rates.append(sum(row[key] for row in chosen) / len(chosen))
        return sum(rates) / len(rates) if rates else None

    def pooled(key, rows):
        return sum(row[key] for row in rows) / len(rows) if rows else None

    narrow_rows = [row for rows in narrow.values() for row in rows]
    passable = lambda row: not row["blocked"]  # noqa: E731
    blocked = lambda row: bool(row["blocked"])  # noqa: E731
    tight = [row for row in narrow_rows if passable(row) and row["slack"] < TIGHT_SLACK]
    bends = {k: v for k, v in narrow.items() if "bend" in k[0]}
    raw = {"P": macro("passage", narrow, passable), "H": macro("hold", narrow, blocked),
           "N": pooled("passage", tight), "L": macro("passage", bends, passable),
           "W": macro("touch_free", narrow), "G": macro("gap_clean_success", gap),
           "B": 1.0 - macro("base_contact", groups), "T": macro("tracking", retention) if retention else None}
    components = {k: (0.0 if v is None else v) for k, v in raw.items()}
    score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    curriculum = {family: sum(level * len(rows) for (name, level), rows in groups.items() if name == family) /
                  sum(len(rows) for (name, _), rows in groups.items() if name == family)
                  for family in sorted({name for name, _ in groups})}
    stair = {k: v for k, v in groups.items() if k[0].startswith(online_top5.STAIR_FAMILY_PREFIX)}
    blocked_rows = [row for row in narrow_rows if blocked(row)]
    metrics = {
        "narrow_mean_level": sum(row["level"] for row in narrow_rows) / len(narrow_rows),
        "narrow_passage_rate": components["P"], "narrow_hold_rate": components["H"],
        "narrow_tight_passage_rate": raw["N"], "narrow_bend_passage_rate": raw["L"],
        "narrow_breach_rate": pooled("breach", blocked_rows),
        "narrow_blocked_share": len(blocked_rows) / len(narrow_rows),
        "dead_end_hold_rate": macro("hold", {k: v for k, v in narrow.items() if k[0] == "narrow_dead_end"}, blocked),
        "narrow_tight_samples": len(tight),
        "narrow_touch_free_rate": components["W"],
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
    """v1 gates; the optional hold / breach limits fold into the narrow pass (terrain_mean_level_pass, as v1 does
    with the narrow mean level): the 128-node v2.3 run let models recovering from a hold collapse into Top-5."""
    result = v1.gates(candidate, reference, criteria)
    criteria = criteria or {}
    m = candidate["criteria_metrics"]
    hold, breach = m.get("narrow_hold_rate"), m.get("narrow_breach_rate")
    if "min_narrow_hold_rate" in criteria:  # no blocked episode in the window: nothing to judge, pass
        result["narrow_hold_pass"] = hold is None or hold >= criteria["min_narrow_hold_rate"]
        result["terrain_mean_level_pass"] = result["terrain_mean_level_pass"] and result["narrow_hold_pass"]
    if "max_narrow_breach_rate" in criteria:
        result["narrow_breach_pass"] = breach is None or breach <= criteria["max_narrow_breach_rate"]
        result["terrain_mean_level_pass"] = result["terrain_mean_level_pass"] and result["narrow_breach_pass"]
    dead_end = m.get("dead_end_hold_rate")  # v2.4e: the dead-end family alone (the macro H hid its drop to 44-78 %)
    if "min_dead_end_hold_rate" in criteria:
        result["dead_end_hold_pass"] = dead_end is None or dead_end >= criteria["min_dead_end_hold_rate"]
        result["terrain_mean_level_pass"] = result["terrain_mean_level_pass"] and result["dead_end_hold_pass"]
    return result


def install():
    """Patch the module globals rank_and_save_top5 / DreamwaqRunner use (this process only)."""
    from gd_lab.rl import runner
    runner.install_episode_collector = install_episode_collector
    online_top5.score_episodes = score_episodes
    online_top5.gates = gates
    # The criteria file may set these (the reloader rejects keys missing from the defaults); off by default.
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("min_narrow_hold_rate", 0.0)
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("max_narrow_breach_rate", 1.0)
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("min_dead_end_hold_rate", 0.0)
