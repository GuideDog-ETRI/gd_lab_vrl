"""Online Top-5 for GAST-Narrow v2.5: v2.4's score over the non-cliff episodes + C, the cliff hold rate.

Score = 0.20 P + 0.15 H + 0.10 N + 0.05 L + 0.05 W + 0.15 G + 0.10 B + 0.05 T + 0.10 R + 0.05 C
  C  cliff tiles: stood 1.5 s inside the zone, no breach / fall; macro over (level) strata.
  R  turn tiles: the v2.5 pass (reached, drift <= 0.30, leg peak force < 150 N) or stood where the turn is not
     possible, no breach (narrow_top5_v24).
P/H/N/L/W are v2's over the passage episodes only (cliff and turn rows excluded: a cliff row would count as a
"passable" passage with passage = 0). Extra metrics: cliff_hold_rate / cliff_breach_rate / cliff_fall_rate (gates
min_cliff_hold_rate, max_cliff_breach_rate), turn_clean_rate, turn_leg_force_p90, stand_nose_up_t10_mean (gate
max_stand_nose_up_t10, seconds above 10 deg in episodes that stood), dead_end_hold_rate (v2.4e gate).
The collector is v2's with the v2.5 fields (turn clean / leg force, nose-up record, cliff fall / edge margin).
"""
from __future__ import annotations

import json
import math
import types
from collections import defaultdict

import torch

from gd_lab.gast_narrow import narrow_top5_v2 as v2
from gd_lab.gast_narrow import narrow_top5_v24 as v24
from gd_lab.rl import online_top5

CLIFF_FAMILY = "narrow_cliff"
WEIGHTS = {"P": 0.20, "H": 0.15, "N": 0.10, "L": 0.05, "W": 0.05, "G": 0.15, "B": 0.10, "T": 0.05, "R": 0.10,
           "C": 0.05}


def install_episode_collector(env, family_columns):
    raw = env.unwrapped
    records: list[dict] = []
    original_reset = raw._reset_idx
    column_family = {column: family for family, columns in family_columns.items() for column in columns}
    import os
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
        nose_max = narrow.nose_max[ids].detach().cpu().tolist()
        nose_t10 = narrow.nose_t10[ids].detach().cpu().tolist()
        from gd_lab.gast_narrow.mdp_v2 import narrow_episode  # Isaac modules: import after the app starts
        from gd_lab.gast_narrow.mdp_cliff import cliff_episode
        from gd_lab.gast_narrow.mdp_turn import turn_episode
        episode = narrow_episode(self)
        blocked = episode.blocked[ids].detach().cpu().tolist()
        slack = episode.slack[ids].detach().cpu().tolist()
        margin = episode.margin[ids].detach().cpu().tolist()
        turn = turn_episode(self)
        turn_clean = turn.clean[ids].detach().cpu().tolist()
        leg_force = turn.peak_force[ids].detach().cpu().tolist()
        leg_impulse = turn.leg_impulse[ids].detach().cpu().tolist()
        turnable = turn.turnable[ids].detach().cpu().tolist()
        cliff = cliff_episode(self)
        fell = cliff.fell[ids].detach().cpu().tolist()
        e_min = cliff.e_min[ids].detach().cpu().tolist()
        direction = cliff.direction[ids].detach().cpu().tolist()
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
            is_cliff = family == CLIFF_FAMILY
            failed = bool(base[index]) or (is_cliff and bool(fell[index]))
            row = {"family": family, "level": int(levels[index]),
                   "gap_success": float(bool(achieved[index]) and not bypassed[index] and not base[index]),
                   "gap_clean_success": float(bool(clean[index]) and not bypassed[index] and not base[index]),
                   "base_contact": float(failed),
                   "progress": min(1.0, max(0.0, distance[index] / max(commanded[index], 0.1))),
                   "tracking": tracking, "energy": 1.0 / (1.0 + penalties / seconds),
                   "return": 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -total_return / seconds /
                                                                  (1.0 + levels[index] / max_level)))))}
            if family.startswith(v2.NARROW_PREFIX):
                ok = not failed
                row.update(blocked=float(bool(blocked[index])),
                           passage=float(bool(passed[index]) and ok),
                           hold=float(bool(held[index]) and not breached[index] and ok),
                           breach=float(bool(breached[index])),
                           slack=float(slack[index]), margin=float(margin[index]),
                           touch_free=float(touch[index] < 0.1),
                           min_clearance=float(min(clearance[index], 1.0)),
                           nose_up_max=float(math.degrees(nose_max[index])), nose_up_t10=float(nose_t10[index]))
            if family.startswith(v24.TURN_PREFIX):
                row.update(turn_clean=float(bool(turn_clean[index]) and bool(passed[index])),
                           turn_leg_force=float(leg_force[index]), turn_leg_impulse=float(leg_impulse[index]),
                           turnable=float(bool(turnable[index])))
            if is_cliff:
                row.update(fell=float(bool(fell[index])), e_min=float(e_min[index]), direction=int(direction[index]))
            if online_top5.validate_episode(row):
                records.append(row)
                if dump and (family.startswith(v24.TURN_PREFIX) or is_cliff):
                    row_dump = dict(row, seconds=seconds, held=bool(held[index]), breached=bool(breached[index]),
                                    base=bool(base[index]),
                                    terms=[name for name, flags in term_flags.items() if flags[index]])
                    with open(dump, "a") as f:
                        f.write(json.dumps(row_dump) + "\n")
        return original_reset(ids)

    raw._reset_idx = types.MethodType(record_and_reset, raw)
    return records


def score_episodes(episodes):
    cliff = [row for row in episodes if row["family"] == CLIFF_FAMILY]
    result = v24.score_episodes([row for row in episodes if row["family"] != CLIFF_FAMILY])
    if result is None:
        return None
    strata = defaultdict(list)
    for row in cliff:
        strata[row["level"]].append(row)

    def macro(value, select=lambda row: True):
        rates = [sum(value(row) for row in chosen) / len(chosen)
                 for chosen in ([row for row in rows if select(row)] for rows in strata.values()) if chosen]
        return sum(rates) / len(rates) if rates else None

    def pooled(value, rows):
        return sum(value(row) for row in rows) / len(rows) if rows else None

    c = macro(lambda row: row["hold"])
    components = dict(result["components"])
    components["C"] = 0.0 if c is None else c
    result["components"] = components
    result["score"] = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    result["score_definition"] = WEIGHTS
    turn = [row for row in episodes if row["family"].startswith(v24.TURN_PREFIX)]
    turnable = [row for row in turn if row.get("turnable")]
    forces = sorted(row["turn_leg_force"] for row in turnable)
    impulses = sorted(row["turn_leg_impulse"] for row in turnable)
    stood = [row for row in episodes if row["family"].startswith(v2.NARROW_PREFIX) and row.get("hold")]
    result["criteria_metrics"].update(
        cliff_hold_rate=c, cliff_breach_rate=macro(lambda row: row["breach"]),
        cliff_fall_rate=macro(lambda row: row["fell"]), cliff_samples=len(cliff),
        cliff_hold_by_direction=[pooled(lambda row: row["hold"], [row for row in cliff if row["direction"] == d])
                                 for d in range(4)],
        turn_clean_rate=pooled(lambda row: row["turn_clean"], turnable),
        turn_leg_force_p90=forces[min(len(forces) - 1, int(0.9 * len(forces)))] if forces else None,
        turn_leg_impulse_p90=impulses[min(len(impulses) - 1, int(0.9 * len(impulses)))] if impulses else None,
        turn_leg_touch_rate=pooled(lambda row: float(row["turn_leg_impulse"] > 0.5), turnable),
        stand_nose_up_t10_mean=pooled(lambda row: row["nose_up_t10"], stood))
    return result


def gates(candidate, reference, criteria=None):
    result = v2.gates(candidate, reference, criteria)
    criteria = criteria or {}
    m = candidate["criteria_metrics"]
    checks = (("min_cliff_hold_rate", "cliff_hold_rate", True), ("max_cliff_breach_rate", "cliff_breach_rate", False),
              ("max_stand_nose_up_t10", "stand_nose_up_t10_mean", False))
    for key, metric, is_min in checks:
        if key in criteria:
            value = m.get(metric)
            ok = value is None or (value >= criteria[key] if is_min else value <= criteria[key])
            result[metric + "_pass"] = ok
            result["terrain_mean_level_pass"] = result["terrain_mean_level_pass"] and ok
    return result


def install():
    v24.install()
    from gd_lab.rl import runner
    runner.install_episode_collector = install_episode_collector
    online_top5.score_episodes = score_episodes
    online_top5.gates = gates
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("min_cliff_hold_rate", 0.0)
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("max_cliff_breach_rate", 1.0)
    online_top5.DEFAULT_TOP5_CRITERIA.setdefault("max_stand_nose_up_t10", 1.0e9)
