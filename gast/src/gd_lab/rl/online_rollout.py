"""Capture completed episode statistics from the existing training rollout."""

from __future__ import annotations

import math
import types

import torch

from gd_lab.rl.online_top5 import validate_episode


def install_episode_collector(env, family_columns: dict[str, list[int]]) -> list[dict]:
    """Observe pre-reset state without changing simulator steps or rewards."""
    raw = env.unwrapped
    records: list[dict] = []
    original_reset = raw._reset_idx
    column_family = {column: family for family, columns in family_columns.items() for column in columns}

    def record_and_reset(self, ids):
        terrain = self.scene.terrain
        types = terrain.terrain_types[ids].detach().cpu().tolist()
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
        # Gap fine-tuning tasks: the monitor's one-shot clean bonus, still held before the reward reset.
        clean = None
        if "platform_gap_monitor" in self.reward_manager.active_terms:
            monitor = self.reward_manager.get_term_cfg("platform_gap_monitor").func
            clean = monitor.tracker.paid[ids].any(dim=1).detach().cpu().tolist()
        max_level = max(1, self.scene.terrain.cfg.terrain_generator.num_rows - 1)
        for index, column in enumerate(types):
            family = column_family.get(column)
            if family is None:
                continue
            seconds = duration[index]
            track = sum(sums.get(name, [0.0] * len(types))[index] for name in
                        ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"))
            # The two tracking rewards have declared weights 1.0 and 0.8.
            tracking = min(1.0, max(0.0, track / (1.8 * seconds)))
            penalties = sum(abs(sums.get(name, [0.0] * len(types))[index]) for name in
                            ("dof_torques_l2", "joint_power", "action_rate_l2", "smoothness"))
            total_return = sum(values[index] for values in sums.values())
            row = {"family": family, "level": int(levels[index]),
                   "gap_success": float(bool(achieved[index]) and not bypassed[index] and not base[index]),
                   "base_contact": float(bool(base[index])),
                   "progress": min(1.0, max(0.0, distance[index] / max(commanded[index], 0.1))),
                   "tracking": tracking, "energy": 1.0 / (1.0 + penalties / seconds),
                   "return": 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -total_return / seconds /
                                                                  (1.0 + levels[index] / max_level)))))}
            if clean is not None:
                row["gap_clean_success"] = float(bool(clean[index]) and not bypassed[index] and not base[index])
            if validate_episode(row):
                records.append(row)
        return original_reset(ids)

    raw._reset_idx = types.MethodType(record_and_reset, raw)
    return records
