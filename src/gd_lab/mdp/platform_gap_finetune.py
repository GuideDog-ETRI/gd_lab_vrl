"""Opt-in gap fine-tuning terms (thin simulator adapters over ``platform_gap_attempts``).

Registration order is part of the contract (checked at the first call):
``platform_gap_crossing`` -> ``platform_gap_monitor`` -> ``platform_gap_intrusion`` -> ``platform_gap_clean``.
The monitor has weight 1.0 and always returns zeros: it runs in every arm (so the
diagnostics are comparable) without changing the objective. The penalty and bonus
terms only read its results, so a zero weight there merely skips the read.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from gd_lab.mdp.platform_gap_attempts import CONTACT_FORCE, GapAttemptTracker
from gd_lab.mdp.platform_gap_terms import boarding_family_mask
from gd_lab.mdp.terrain_families import family_column_masks

MONITOR, CROSSING = "platform_gap_monitor", "platform_gap_crossing"
TERM_ORDER = (CROSSING, MONITOR, "platform_gap_intrusion", "platform_gap_clean")


class GapTileGeometry:
    """GPU lookup of exact (row, col) tile metadata captured before mesh insertion."""

    def __init__(self, env):
        cfg = env.scene.terrain.cfg.terrain_generator
        metadata = getattr(cfg, "gap_tile_metadata", None)
        if metadata is None:
            raise ValueError("gap fine-tuning requires GapMetadataTerrainGenerator")
        rows, cols = len(metadata), len(metadata[0])
        self.slots = torch.zeros(rows, cols, 2, 2, device=env.device)
        self.lower_decks = torch.zeros(rows, cols, 2, device=env.device)
        present = torch.zeros(rows, cols, dtype=torch.bool, device=env.device)
        for row, row_tiles in enumerate(metadata):
            for col, tile in enumerate(row_tiles):
                if tile is not None:
                    self.slots[row, col] = torch.tensor(tile["slots"], device=env.device)
                    self.lower_decks[row, col] = torch.tensor(tile["lower_decks"], device=env.device)
                    present[row, col] = True
        gap_cols = family_column_masks(env).get("platform_gap")
        if gap_cols is None or not present[:, gap_cols].all():
            raise ValueError("every platform_gap tile needs metadata (check GapMetadataTerrainGenerator)")

    def for_envs(self, env):
        terrain = env.scene.terrain
        row, col = terrain.terrain_levels.long(), terrain.terrain_types.long()
        return self.slots[row, col], self.lower_decks[row, col]


class GapMonitor(ManagerTermBase):
    """Computes intrusion cost and the attempt state machine every step; returns zeros."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry = GapTileGeometry(env)
        self.tracker = GapAttemptTracker(env.num_envs, env.device)
        self.cost = torch.zeros(env.num_envs, device=env.device)
        self.clean_event = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._crossing = None
        self._records = None

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.cost[ids] = 0.0
        self.clean_event[ids] = False

    def _check_order(self, env):
        names = list(env.reward_manager.active_terms)
        present = [name for name in TERM_ORDER if name in names]
        if [names.index(name) for name in present] != sorted(names.index(name) for name in present) or CROSSING not in present:
            raise RuntimeError(f"gap reward terms must be ordered {TERM_ORDER}; found {names}")
        crossing = env.reward_manager.get_term_cfg(CROSSING)
        if not crossing.weight:  # RewardManager never calls a zero-weight term, so no event would be produced
            raise RuntimeError(f"{CROSSING} has weight 0: the crossing events this monitor reads would never update")
        self._crossing = crossing.func

    @staticmethod
    def _check_feet(env, asset_cfg, sensor_cfg):
        """Per-foot arrays from the robot and from the contact sensor must refer to the same feet."""
        robot, sensor = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
        robot_names, sensor_names = getattr(robot, "body_names", None), getattr(sensor, "body_names", None)
        if robot_names is None or sensor_names is None:
            print("[WARN] gap monitor: foot-order check skipped (no body_names on the robot or the contact sensor)", flush=True)
            return
        pick = lambda names, ids: list(names[ids]) if isinstance(ids, slice) else [names[i] for i in ids]  # noqa: E731
        if pick(robot_names, asset_cfg.body_ids) != pick(sensor_names, sensor_cfg.body_ids):
            raise RuntimeError("robot and contact-sensor foot order differ: per-foot contact would be mismatched")

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
        if self._crossing is None:
            self._check_order(env)
            self._check_feet(env, asset_cfg, sensor_cfg)
        robot = env.scene[asset_cfg.name]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
        origins = env.scene.env_origins
        slots, lower = self.geometry.for_envs(env)
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        result = self.tracker.update(
            feet[..., 0] - origins[:, None, 0], feet[..., 2], slots, lower + origins[:, None, 2],
            robot.data.heading_w.cos() >= 0, boarding_family_mask(env), self._crossing.last_event, force > CONTACT_FORCE,
        )
        self.cost, self.clean_event = result.cost, result.clean_event
        return torch.zeros(env.num_envs, device=env.device)

    def write_records(self, env, records):
        log_dir = getattr(env.cfg, "log_dir", None)
        if not records or log_dir is None:
            return
        if self._records is None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._records = open(Path(log_dir) / "gap_attempts.jsonl", "a", buffering=1)  # noqa: SIM115
        step = int(env.common_step_counter)
        for record in records:
            self._records.write(json.dumps({"step": step, **record}) + "\n")


def gap_intrusion_penalty(env):
    """Cost in [0, 1]; the arm's negative weight turns it into a penalty per second."""
    return env.reward_manager.get_term_cfg(MONITOR).func.cost


def gap_clean_bonus(env):
    """One-shot event divided by dt, because RewardManager multiplies every term by dt."""
    return env.reward_manager.get_term_cfg(MONITOR).func.clean_event.float() / env.step_dt


def platform_gap_diagnostics(env, env_ids):
    """Monitoring-only curriculum term: scalar stats for the resetting envs, raw records to disk.

    CurriculumManager.compute runs before RewardManager.reset, so the monitor state
    is still intact here. Open attempts are failed as ``episode_end`` first.
    """
    monitor = env.reward_manager.get_term_cfg(MONITOR).func
    monitor.tracker.close_episode(env_ids)
    stats, records = monitor.tracker.collect(env_ids)
    monitor.write_records(env, records)
    return stats
