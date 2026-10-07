"""v2.1 final (main GAST teacher training): v2 + the Codex/Claude cross-review of 2026-10-07 + user decisions.

Speed and command
- Gaps and stairs up to a 1.2 m/s COMMAND (usual 0.8-1.0 m/s). Gap tiles and half of the stair-tile resamples
  (straight up/down runs) draw from ``V21_SPEED_MIX``; draws above the current ceiling are redrawn in
  [0.2, ceiling] (no clamping pile-up). The ceiling ramps 0.6 -> 1.2 m/s over GD_LAB_V21_SPEED_RAMP_STEPS env
  steps of this process (default 300k; 0 = full range). These envs are never "standing" envs (IsaacLab zeroes a
  standing env's command every step).
Gap
- Clean split: +0.25 per foot that lands on the far side of a slot without a violation (contact or a 3 cm deep
  dip while over the slot +- foot radius - 5 mm), once per foot and slot; the all-four clean event keeps +0.5.
  Strict clean (all-four) uses the same boundary.
- Hind hop: both hind feet airborne (15 N / 5 N hysteresis) for >= 50 ms within 0.6 m of a slot: -1 /s.
- Overlift: a foot-sphere bottom more than 0.20 m above the higher deck within 0.3 m of a slot: -0.25 /s
  (saturating at +0.15 m).
- Foothold margin -0.25 per fully violating touchdown (v2: -0.5).
Stairs
- Stall: forward command > 0.3 m/s on the stair band or its 0.6 m approach with forward progress below
  max(0.1, 0.2 * command) for > 1.5 s: -0.5 /s.
- Disturbance eligibility is latched for 2 s once a forward run up/down was detected (slowing down does not dodge
  it); recovery window after the force 2 s (v2: 1 s).
Events
- Termination: -2 per non-timeout termination (was -2 * dt = -0.02). Disturbance fall: extra -3 (total -5).
Observations unchanged. BIVT-Ray and GAST teachers import this one module.
"""

from __future__ import annotations

import math
import os

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from gd_lab.mdp.gap_stair_v2 import V2_DISTURBANCE, _StairGeometry, _gap_mask, add_v2_terms, chebyshev_frame
from gd_lab.mdp.gap_stair_v2_math import (  # noqa: F401
    FOOT_RADIUS,
    V21_SPEED_MIX,
    hysteresis_contact,
    overlift_cost,
    sample_v21_speed,
    sample_v21_speed_in_ceiling,
    slot_side,
    speed_ceiling,
)
from gd_lab.mdp.platform_gap_commands import BoardingVelocityCommand
from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
from gd_lab.mdp.platform_gap_terms import boarding_family_mask
from gd_lab.mdp.terrain_families import family_column_masks

V21_STAIR_FORWARD_PROB = 0.5
V21_WEIGHTS = {
    "gap_foothold_margin": -0.25,  # per fully violating touchdown
    "gap_hind_hop": -1.0,  # per second
    "overspeed": -1.0,  # per second, (m/s above command + 0.2)
    "stair_stall": -0.5,  # per second
    "gap_overlift": -0.25,  # per second
    "gap_foot_clean": 0.25,  # per foot and slot
    "platform_gap_clean": 0.5,  # all four feet clean (v2: 1.5)
    "termination_penalty": -2.0,  # per non-timeout termination (event / dt)
    "stair_push_fall": -3.0,  # extra on a disturbance fall -> total -5 with the termination
}
V21_DISTURBANCE = {**V2_DISTURBANCE, "latch_s": 2.0, "after_s": 2.0}
EDGE_TOLERANCE = 0.005  # m inside the foot radius for the slot boundary (contact / clean)
CONTACT_ON, CONTACT_OFF = 15.0, 5.0


class V21VelocityCommand(BoardingVelocityCommand):
    """Boarding command (gap tiles face their crossing) with the v2.1 speeds, plus straight stair runs."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stairs = None
        self.ramp_steps = int(os.environ.get("GD_LAB_V21_SPEED_RAMP_STEPS", 300_000))
        print(f"[INFO] v2.1 command: speed ceiling ramp steps={self.ramp_steps}", flush=True)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        ceiling = speed_ceiling(int(self._env.common_step_counter), self.ramp_steps)
        gap = ids[boarding_family_mask(self._env)[ids]]
        if len(gap):
            self.vel_command_b[gap, 0] = sample_v21_speed_in_ceiling(len(gap), ceiling, self.device)
            self.is_standing_env[gap] = False  # a standing env's command is zeroed every step by IsaacLab
        if self._stairs is None:
            self._stairs = _StairGeometry(self._env)
        col = self._env.scene.terrain.terrain_types.long()[ids]
        stair = ids[self._stairs.is_stair[col] & (torch.rand(len(ids), device=self.device) < V21_STAIR_FORWARD_PROB)]
        if len(stair):
            rel = self.robot.data.root_pos_w[stair, :2] - self._env.scene.env_origins[stair, :2]
            _, outward = chebyshev_frame(rel)
            inward = torch.rand(len(stair), device=self.device) < 0.5  # up or down, whichever way the tile goes
            yaw = torch.atan2(outward[:, 1], outward[:, 0]) + inward.float() * math.pi
            self.heading_target[stair] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            self.is_heading_env[stair] = True
            self.vel_command_b[stair, 0] = sample_v21_speed_in_ceiling(len(stair), ceiling, self.device)
            self.vel_command_b[stair, 1] = 0.0
            self._pulse_active[stair] = False
            self.is_standing_env[stair] = False


class GapHindHop(ManagerTermBase):
    """Per second: both hind feet airborne (hysteresis) for >= ``min_air_s`` within ``reach`` m of a gap slot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry = GapTileGeometry(env)
        self.contact = torch.ones(env.num_envs, 2, device=env.device)
        self.air_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.contact[ids] = 1.0
        self.air_time[ids] = 0.0

    def __call__(self, env, sensor_cfg: SceneEntityCfg, reach: float = 0.6, min_air_s: float = 0.05):
        slots, _ = self.geometry.for_envs(env)
        x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        near = ((x[:, None] > slots[..., 0] - reach) & (x[:, None] < slots[..., 1] + reach)).any(-1)
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        self.contact = hysteresis_contact(force, self.contact, CONTACT_ON, CONTACT_OFF)
        both_air = (self.contact < 0.5).all(-1)
        self.air_time = torch.where(both_air, self.air_time + env.step_dt, torch.zeros_like(self.air_time))
        return (near & (self.air_time >= min_air_s) & _gap_mask(env)).float()


class _UpperDecks(GapTileGeometry):
    def __init__(self, env):
        super().__init__(env)
        cfg = env.scene.terrain.cfg.terrain_generator
        rows, cols = len(cfg.gap_tile_metadata), len(cfg.gap_tile_metadata[0])
        self.upper = torch.zeros(rows, cols, 2, device=env.device)
        for r, row_tiles in enumerate(cfg.gap_tile_metadata):
            for c, tile in enumerate(row_tiles):
                if tile is not None:
                    d = tile["decks"]
                    self.upper[r, c] = torch.tensor([max(d[0], d[1]), max(d[1], d[2])], device=env.device)

    def upper_for_envs(self, env):
        t = env.scene.terrain
        return self.upper[t.terrain_levels.long(), t.terrain_types.long()]


class GapOverlift(ManagerTermBase):
    """Per second: max over feet of the overlift cost (foot-sphere bottom > higher deck + 0.20 m) near a slot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry = _UpperDecks(env)

    def __call__(self, env, asset_cfg: SceneEntityCfg, reach: float = 0.3):
        feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
        slots, _ = self.geometry.for_envs(env)
        upper = self.geometry.upper_for_envs(env) + env.scene.env_origins[:, None, 2]  # [N, 2]
        x = feet[..., 0] - env.scene.env_origins[:, None, 0]  # [N, F]
        near_slot = (x[..., None] > slots[:, None, :, 0] - reach) & (x[..., None] < slots[:, None, :, 1] + reach)
        # reference: the higher deck of the nearest slot the foot is near
        ref = torch.where(near_slot[..., 1], upper[:, None, 1], upper[:, None, 0])
        bottom = feet[..., 2] - FOOT_RADIUS
        return overlift_cost(bottom, ref, near_slot.any(-1)) * _gap_mask(env)


class GapFootClean(ManagerTermBase):
    """Event (/dt): number of feet that just landed on the far side of a slot with no violation since they left
    the near side, paid once per foot and slot per episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry = GapTileGeometry(env)
        n, dev = env.num_envs, env.device
        self.last_side = torch.zeros(n, 4, 2, device=dev)  # -1 / +1 side of each slot at the last stance, 0 none
        self.violated = torch.zeros(n, 4, 2, dtype=torch.bool, device=dev)
        self.paid = torch.zeros(n, 4, 2, dtype=torch.bool, device=dev)
        self.contact = torch.ones(n, 4, device=dev)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_side[ids] = 0
        self.violated[ids] = False
        self.paid[ids] = False
        self.contact[ids] = 1.0

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, reach: float = 0.6):
        feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
        slots, lower = self.geometry.for_envs(env)
        lo, hi = slots[:, None, :, 0], slots[:, None, :, 1]  # [N, 1, 2]
        x = (feet[..., 0] - env.scene.env_origins[:, None, 0])[..., None]  # [N, F, 1]
        margin = FOOT_RADIUS - EDGE_TOLERANCE
        side = slot_side(x, lo, hi, margin)  # [N, F, 2]
        over = side == 0
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        previous = self.contact
        self.contact = hysteresis_contact(force, previous, CONTACT_ON, CONTACT_OFF)
        touchdown = ((self.contact > 0.5) & (previous < 0.5))[..., None]  # [N, F, 1]
        stance = (self.contact > 0.5)[..., None]
        deck = lower[:, None, :] + env.scene.env_origins[:, None, None, 2]
        deep = (deck - (feet[..., 2][..., None] - FOOT_RADIUS)) >= 0.03
        self.violated |= over & (stance | deep)
        near = (x > lo - reach) & (x < hi + reach)
        crossed = touchdown & (side != 0) & (self.last_side != 0) & (side != self.last_side) & near
        pay = crossed & ~self.violated & ~self.paid & _gap_mask(env)[:, None, None]
        self.paid |= pay
        landed = stance & (side != 0)
        self.last_side = torch.where(landed, side, self.last_side)
        self.violated = self.violated & ~landed  # a new stance on either side starts a fresh crossing
        return pay.float().sum((1, 2)) / env.step_dt


def _gap_or_stair(env) -> torch.Tensor:
    cols = family_column_masks(env)
    names = [n for n in cols if n == "platform_gap" or n.startswith("pyramid_stairs")]
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    col = env.scene.terrain.terrain_types.long()
    for name in names:
        mask |= torch.isin(col, cols[name])
    return mask


def overspeed(env, margin: float = 0.2):
    """Per second: forward body speed above the forward command + margin (m/s) on gap and stair tiles."""
    command = env.command_manager.get_command("base_velocity")[:, 0]
    speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    excess = (speed - command - margin).clamp(min=0) * (command > 0)
    return excess * _gap_or_stair(env)


class StairStall(ManagerTermBase):
    """Per second once a forward command (> 0.3 m/s) on the stair band or its ``approach`` m approach has made
    forward progress below max(0.1, 0.2 * command) for more than ``hold_s``."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stairs = _StairGeometry(env)
        self.timer = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.timer[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_s: float = 1.5, approach: float = 0.6):
        robot = env.scene["robot"]
        col = env.scene.terrain.terrain_types.long()
        rel = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        cheb, _ = chebyshev_frame(rel)
        zone = self.stairs.is_stair[col] & (cheb > self.stairs.inner[col] - approach) & \
            (cheb < self.stairs.outer[col] + approach)
        command = env.command_manager.get_command("base_velocity")[:, 0]
        slow = robot.data.root_lin_vel_b[:, 0] < torch.clamp(0.2 * command, min=0.1)
        stuck = zone & (command > 0.3) & slow
        self.timer = torch.where(stuck, self.timer + env.step_dt, torch.zeros_like(self.timer))
        return (self.timer > hold_s).float()


def termination_event(env):
    """Non-timeout termination as an event (/dt), so the weight is per termination."""
    return env.termination_manager.terminated.float() / env.step_dt


def _apply_v21(cfg) -> None:
    from isaaclab.managers import RewardTermCfg

    w = V21_WEIGHTS
    feet = SceneEntityCfg("robot", body_names=".*_foot")
    contacts = SceneEntityCfg("contact_forces", body_names=".*_foot")
    cfg.commands.base_velocity.class_type = V21VelocityCommand
    cfg.rewards.gap_foothold_margin.weight = w["gap_foothold_margin"]
    cfg.rewards.platform_gap_clean.weight = w["platform_gap_clean"]
    cfg.rewards.termination_penalty.func = termination_event
    cfg.rewards.termination_penalty.weight = w["termination_penalty"]
    cfg.rewards.stair_push_fall.weight = w["stair_push_fall"]
    cfg.rewards.stair_handle_disturbance.params.update(V21_DISTURBANCE)
    hind = SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"])
    cfg.rewards.gap_hind_hop = RewardTermCfg(func=GapHindHop, weight=w["gap_hind_hop"], params={"sensor_cfg": hind})
    cfg.rewards.gap_overlift = RewardTermCfg(func=GapOverlift, weight=w["gap_overlift"], params={"asset_cfg": feet})
    cfg.rewards.gap_foot_clean = RewardTermCfg(
        func=GapFootClean, weight=w["gap_foot_clean"], params={"asset_cfg": feet, "sensor_cfg": contacts})
    cfg.rewards.overspeed = RewardTermCfg(func=overspeed, weight=w["overspeed"], params={})
    cfg.rewards.stair_stall = RewardTermCfg(func=StairStall, weight=w["stair_stall"], params={})


def add_v21_terms(cfg) -> None:
    """v2 terms + the v2.1 final command and terms (BIVT-Ray CleanV21 and the GAST v2.1 teachers)."""
    add_v2_terms(cfg)
    _apply_v21(cfg)


def apply_v21_after_v2(cfg) -> None:
    """For a cfg on which ``add_v2_terms`` already ran (e.g. the GAST scratch v2 teacher)."""
    _apply_v21(cfg)


def add_student_v21_env(cfg) -> None:
    """Distillation env of a v2.1 teacher: the v2 student env (26 cm gaps, stair disturbance) + v2.1 commands."""
    from gd_lab.mdp.gap_stair_v2 import add_student_v2_env

    add_student_v2_env(cfg)
    cfg.commands.base_velocity.class_type = V21VelocityCommand
    cfg.rewards.stair_handle_disturbance.params.update(V21_DISTURBANCE)
