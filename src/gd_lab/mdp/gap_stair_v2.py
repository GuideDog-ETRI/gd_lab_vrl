"""v2 teacher terms: gap/stair foothold margin, slot probing, and the stair hip-handle disturbance.

Thin IsaacLab adapters over ``gap_stair_v2_math`` (the math is CPU-tested there).

Disturbance (``StairHandleDisturbance``): on pyramid-stair tiles, while the robot walks FORWARD up or
down the slope, a force is applied at the hip handle (base frame ``handle_pos``) pointing DOWNHILL:
ascending -- a person below pulls the handle back and down (nose-up moment, the real failure where the
front feet lift and the robot tips backward); descending -- a push from behind. The force keeps its world
direction for its duration (re-expressed in the body frame once per policy step). The term itself returns
the front-lift cost inside the disturbance window (force on + ``after_s``); ``stair_push_fall`` and
``stair_push_slip`` read its state. Nothing new enters the observations: while the force acts, the
velocity change it causes over 0.2 s is written to the existing ``push_delta_v`` critic buffer, so a
checkpoint of the same task family resumes unchanged. The regular interval push may overwrite that stamp
for one step; both are critic-only.
"""

from __future__ import annotations

import math
import os

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from gd_lab.mdp.gap_stair_v2_math import (
    chebyshev_frame,
    downhill_force,
    front_lift_cost,
    gap_edge_margin_cost,
    nosed_edges,
    pyramid_step_edges,
    slot_low_or_contact_cost,
    stair_nose_margin_cost,
    stair_push_modes,
)
from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
from gd_lab.mdp.terrain_families import family_column_masks

STAIR_FAMILIES = {  # name -> inverted (centre low)
    "pyramid_stairs": False, "pyramid_stairs_wide": False, "pyramid_stairs_nose": False,
    "pyramid_stairs_inv": True, "pyramid_stairs_inv_wide": True, "pyramid_stairs_inv_nose": True,
}
CONTACT_N = 10.0
DISTURBANCE = "stair_handle_disturbance"


def _column(env):
    return env.scene.terrain.terrain_types.long()


class _StairGeometry:
    """Per-column pyramid edge radii (NaN padded), inverted flag, stair-band limits."""

    def __init__(self, env):
        gen = env.scene.terrain.cfg.terrain_generator
        cols = family_column_masks(env)
        n = gen.num_cols
        edges_per_col: list[list[float]] = [[] for _ in range(n)]
        self.is_stair = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.inverted = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.inner = torch.zeros(n, device=env.device)
        self.outer = torch.zeros(n, device=env.device)
        for name, inverted in STAIR_FAMILIES.items():
            if name not in gen.sub_terrains or name not in cols:
                continue
            sub = gen.sub_terrains[name]
            size = float(min(gen.size))  # the generator builds every sub-terrain at its own tile size
            edges = pyramid_step_edges(size, float(sub.border_width), float(sub.platform_width), float(sub.step_width))
            # Nosing lips (gd_lab.mdp.terrains.nosing_stairs) move the real drop-off by nose_depth: outward at
            # every pyramid edge; inward at every inverted edge except the outermost one, which has no lip.
            edges = nosed_edges(edges, float(getattr(sub, "nose_depth", 0.0) or 0.0), inverted)
            for col in cols[name].tolist():
                edges_per_col[col] = edges
                self.is_stair[col] = True
                self.inverted[col] = inverted
                self.inner[col] = min(edges)
                self.outer[col] = max(edges)
        width = max([len(e) for e in edges_per_col] + [1])
        table = torch.full((n, width), float("nan"), device=env.device)
        for col, edges in enumerate(edges_per_col):
            if edges:
                table[col, : len(edges)] = torch.tensor(edges, device=env.device)
        self.edges = table


class _UpperDeckGeometry(GapTileGeometry):
    """GapTileGeometry plus the HIGHER deck next to each slot (metadata ``decks``: left, platform, right)."""

    def __init__(self, env):
        super().__init__(env)
        cfg = env.scene.terrain.cfg.terrain_generator
        rows, cols = len(cfg.gap_tile_metadata), len(cfg.gap_tile_metadata[0])
        self.upper_decks = torch.zeros(rows, cols, 2, device=env.device)
        for row, row_tiles in enumerate(cfg.gap_tile_metadata):
            for col, tile in enumerate(row_tiles):
                if tile is not None:
                    d = tile["decks"]
                    self.upper_decks[row, col] = torch.tensor([max(d[0], d[1]), max(d[1], d[2])], device=env.device)

    def upper_for_envs(self, env):
        terrain = env.scene.terrain
        return self.upper_decks[terrain.terrain_levels.long(), terrain.terrain_types.long()]


def _gap_mask(env):
    cols = family_column_masks(env).get("platform_gap")
    if cols is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.isin(_column(env), cols)


class FootholdMargin(ManagerTermBase):
    """Touchdown-event cost ``(margin - d)/margin`` per foot near a drop-off edge (``mode`` gap|stair).

    Returned as an event divided by dt (the RewardManager multiplies by dt), so the weight is per
    fully-violating touchdown. Off below ``min_command`` m/s, where short steps cannot avoid edges."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        mode = cfg.params.get("mode")
        if mode not in ("gap", "stair"):
            raise ValueError("FootholdMargin mode must be 'gap' or 'stair'")
        self.gap = GapTileGeometry(env) if mode == "gap" else None
        self.stairs = _StairGeometry(env) if mode == "stair" else None
        self.violations = torch.zeros(env.num_envs, device=env.device)
        self.touchdowns = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.violations[ids] = 0
        self.touchdowns[ids] = 0

    def __call__(self, env, mode: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 margin: float = 0.04, min_command: float = 0.2):
        robot, sensor = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
        touchdown = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [N, F]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
        command = env.command_manager.get_command("base_velocity")[:, :2].norm(dim=-1)
        if mode == "gap":
            slots, _ = self.gap.for_envs(env)
            cost = gap_edge_margin_cost(feet[..., 0], slots, margin) * _gap_mask(env)[:, None]
        else:
            col = _column(env)
            cheb = torch.maximum(feet[..., 0].abs(), feet[..., 1].abs())
            cost = stair_nose_margin_cost(cheb, self.stairs.edges[col], self.stairs.inverted[col], margin)
            cost = cost * self.stairs.is_stair[col][:, None]
        cost = cost * touchdown * (command >= min_command)[:, None]
        self.touchdowns += touchdown.sum(-1)
        self.violations += (cost > 0).sum(-1)
        return cost.sum(-1) / env.step_dt


class SlotProbe(ManagerTermBase):
    """Per-second cost: mean over feet over a gap slot that touch or sit below the HIGHER deck."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry = _UpperDeckGeometry(env)

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, clearance: float = 0.02):
        robot, sensor = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
        slots, _ = self.geometry.for_envs(env)
        upper = self.geometry.upper_for_envs(env) + env.scene.env_origins[:, None, 2]
        contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > CONTACT_N
        x = feet[..., 0] - env.scene.env_origins[:, None, 0]
        return slot_low_or_contact_cost(x, feet[..., 2], contact, slots, upper, clearance) * _gap_mask(env)


class StairHandleDisturbance(ManagerTermBase):
    """Applies the hip-handle force on pyramid stairs; returns the front-lift cost inside the window."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.robot = env.scene[p["asset_cfg"].name]
        self.trunk = self.robot.body_names.index(p.get("body_name", "trunk"))
        self.stairs = _StairGeometry(env)
        n, dev = env.num_envs, env.device
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ends = torch.zeros(n, device=dev)  # time the force stops
        self.window_end = torch.full((n,), -1.0, device=dev)  # force end + after_s
        self.cooldown_until = torch.zeros(n, device=dev)
        self.force_w = torch.zeros(n, 3, device=dev)
        self.time = torch.zeros(n, device=dev)
        self.in_window = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fell = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fall_paid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fall_event = torch.zeros(n, dtype=torch.bool, device=dev)
        self.events = torch.zeros(n, device=dev)
        self.falls = torch.zeros(n, device=dev)
        self.window_steps = torch.zeros(n, device=dev)
        self.lift_steps = torch.zeros(n, device=dev)
        self.handle = torch.tensor(p.get("handle_pos", (-0.33, 0.0, 0.12)), device=dev)
        self.mass = torch.ones(n, device=dev)
        print(f"[INFO] stair handle disturbance: force ramp steps="
              f"{os.environ.get('GD_LAB_V2_FORCE_RAMP_STEPS', p.get('ramp_steps', 150_000))}", flush=True)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for t in (self.active, self.in_window, self.fell, self.fall_paid, self.fall_event):
            t[ids] = False
        for t in (self.ends, self.cooldown_until, self.time, self.events, self.falls, self.window_steps,
                  self.lift_steps):
            t[ids] = 0
        self.window_end[ids] = -1.0
        self.force_w[ids] = 0

    def _scale(self, env, ramp_steps):
        """Force ramp over this process's env steps. ``common_step_counter`` restarts at 0 on every launch, so a
        run that resumes an already-trained v2 checkpoint (or a smoke) sets GD_LAB_V2_FORCE_RAMP_STEPS=0."""
        ramp = int(os.environ.get("GD_LAB_V2_FORCE_RAMP_STEPS", ramp_steps))
        return 1.0 if ramp <= 0 else min(1.0, float(env.common_step_counter) / ramp)

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 front_feet: tuple = ("FL_foot", "FR_foot"),
                 rate_hz: float = 0.3, cooldown_s: float = 2.0, after_s: float = 1.0, ramp_steps: int = 150_000,
                 ascend_force=(0.0, 200.0), ascend_duration=(0.3, 1.0), ascend_angle_deg=(20.0, 45.0),
                 descend_force=(0.0, 150.0), descend_duration=(0.1, 0.5), descend_angle_deg=(-10.0, 30.0),
                 handle_pos=(-0.33, 0.0, 0.12), body_name: str = "trunk", fall_tilt_cos: float = 0.5):
        robot, sensor = self.robot, env.scene[sensor_cfg.name]
        dt, dev, n = env.step_dt, env.device, env.num_envs
        self.time += dt
        col = _column(env)
        origin = env.scene.env_origins
        rel = robot.data.root_pos_w[:, :2] - origin[:, :2]
        cheb, outward = chebyshev_frame(rel)
        stair = self.stairs.is_stair[col]
        band = stair & (cheb > self.stairs.inner[col] + 0.15) & (cheb < self.stairs.outer[col] - 0.10)
        uphill = torch.where(self.stairs.inverted[col][:, None], outward, -outward)
        heading_w = robot.data.heading_w
        heading = torch.stack((heading_w.cos(), heading_w.sin()), -1)
        command_vx = env.command_manager.get_command("base_velocity")[:, 0]
        ascending, descending = stair_push_modes(uphill, heading, robot.data.root_lin_vel_w[:, :2], command_vx)

        # finish forces
        finished = self.active & (self.time >= self.ends)
        self.active &= ~finished
        self.force_w[finished] = 0
        # start new forces
        # never start on the step an env terminates (it is reset right after the reward computation)
        idle = ~self.active & (self.time >= self.cooldown_until) & band & (ascending | descending) & ~env.reset_buf
        start = idle & (torch.rand(n, device=dev) < rate_hz * dt)
        if start.any():
            u = lambda r: r[0] + (r[1] - r[0]) * torch.rand(n, device=dev)  # noqa: E731
            scale = self._scale(env, ramp_steps)
            magnitude = torch.where(ascending, u(ascend_force), u(descend_force)) * scale
            duration = torch.where(ascending, u(ascend_duration), u(descend_duration))
            angle = torch.deg2rad(torch.where(ascending, u(ascend_angle_deg), u(descend_angle_deg)))
            force = downhill_force(uphill, magnitude, angle)
            self.force_w = torch.where(start[:, None], force, self.force_w)
            self.active |= start
            self.ends = torch.where(start, self.time + duration, self.ends)
            self.window_end = torch.where(start, self.time + duration + after_s, self.window_end)
            self.cooldown_until = torch.where(start, self.time + duration + cooldown_s, self.cooldown_until)
            self.events += start.float()
            # live mass, including the randomized payload / base mass of this episode
            masses = robot.root_physx_view.get_masses().sum(-1).to(dev)
            self.mass = torch.where(start, masses, self.mass)

        # Critic-only stamp in the existing push buffer (no new observation dims), refreshed every step while the
        # force acts: the velocity change it would cause over the buffer's 0.2 s hold (200 N / 29 kg -> 1.4 m/s,
        # inside the +-2 observation clip). It stays visible until 0.2 s after the force ends.
        if self.active.any():
            if not hasattr(env, "push_delta_v_buf"):
                env.push_delta_v_buf = torch.zeros(n, 3, device=dev)
                env.push_step_buf = torch.full((n,), -(10**9), dtype=torch.long, device=dev)
            delta_v = self.force_w * 0.2 / self.mass[:, None]
            yaw_frame = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_v)
            stamp = self.active & ~env.reset_buf  # a terminating env is reset after this: no stale stamp
            env.push_delta_v_buf[stamp] = yaw_frame[stamp]
            env.push_step_buf[stamp] = int(env.common_step_counter)

        # apply (body frame, at the handle); zero for everyone else
        forces_b = quat_apply_inverse(robot.data.root_quat_w, self.force_w) * self.active[:, None]
        positions = self.handle.expand(n, 3)
        robot.set_external_force_and_torque(forces_b[:, None, :], torch.zeros_like(forces_b)[:, None, :],
                                            positions=positions[:, None, :], body_ids=[self.trunk])

        # window bookkeeping and costs
        self.in_window = self.time < self.window_end
        front_ids = [sensor.body_names.index(name) for name in front_feet]
        front_contact = sensor.data.net_forces_w[:, front_ids].norm(dim=-1) > CONTACT_N
        nose_up_rate = -robot.data.root_ang_vel_b[:, 1]
        lift = front_lift_cost(front_contact, nose_up_rate) * self.in_window
        tipped = robot.data.projected_gravity_b[:, 2] > -fall_tilt_cos  # tilt beyond ~60 deg
        self.fell = self.in_window & (env.termination_manager.terminated | tipped)
        self.fall_event = self.fell & ~self.fall_paid  # once per disturbance window
        self.fall_paid = (self.fall_paid | self.fell) & self.in_window
        self.falls += self.fall_event.float()
        self.window_steps += self.in_window.float()
        self.lift_steps += (lift > 0).float()
        return lift


def _disturbance(env) -> StairHandleDisturbance:
    cfg = env.reward_manager.get_term_cfg(DISTURBANCE)
    if not cfg.weight:  # RewardManager skips zero-weight terms: no force would be applied, state would be stale
        raise RuntimeError(f"{DISTURBANCE} has weight 0; give it a nonzero weight or remove the push fall/slip terms")
    return cfg.func


def stair_push_fall(env):
    """One-shot per disturbance window (/dt): fell (terminated or tilted > ~60 deg) inside the window."""
    return _disturbance(env).fall_event.float() / env.step_dt


def stair_push_slip(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Stance-foot horizontal speed^2 inside the disturbance window."""
    term = _disturbance(env)
    robot, sensor = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > CONTACT_N
    speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].square().sum(-1)
    return (speed * contact).sum(-1) * term.in_window


def gap_stair_v2_diagnostics(env, env_ids):
    """Curriculum-slot logger (no curriculum change): rates over the resetting envs."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    out = {}
    manager = env.reward_manager
    for name in ("gap_foothold_margin", "stair_foothold_margin"):
        if name in manager.active_terms:
            term = manager.get_term_cfg(name).func
            out[f"{name}_violation_rate"] = float(term.violations[ids].sum() / term.touchdowns[ids].sum().clamp_min(1))
    if DISTURBANCE in manager.active_terms:
        term = _disturbance(env)
        events = term.events[ids].sum()
        out["stair_push_events_per_episode"] = float(events / max(1, len(ids)))
        out["stair_push_fall_rate"] = float(term.falls[ids].sum() / events.clamp_min(1))
        window = term.window_steps[ids].sum().clamp_min(1)
        out["stair_push_front_lift_fraction"] = float(term.lift_steps[ids].sum() / window)
        ramp = manager.get_term_cfg(DISTURBANCE).params.get("ramp_steps", 150_000)
        out["stair_push_force_scale"] = term._scale(env, ramp)
    return out


# One definition for every teacher that uses v2 (BIVT-Ray CleanV2 and the GAST CleanV2 teacher), so the
# two teachers are trained on the same objective. gast/src keeps a byte-identical copy of this file
# (tests/test_gap_stair_v2.py pins it).
V2_WEIGHTS = {
    "gap_foothold_margin": -0.5,  # per fully violating touchdown
    "stair_foothold_margin": -0.2,
    "gap_slot_probe": -1.0,  # per second, mean over feet
    "stair_handle_disturbance": -1.0,  # front-lift cost per second inside the disturbance window
    "stair_push_fall": -5.0,  # per disturbance window, on top of termination_penalty
    "stair_push_slip": -0.2,
}
V2_GAP_WIDTH_RANGE = (0.02, 0.26)
# MuJoCo baseline of BIVT-Ray 21068 on the 15 cm stair course (2026-10-06): a 0.6 s back-and-down handle pull
# of 100 N was absorbed, 150 N lifted both front feet for 0.5 s, 200 N flipped the robot backward down the
# stairs; downhill pushes of 80-120 N were absorbed. Training covers that failure range.
V2_DISTURBANCE = {"ascend_force": (0.0, 200.0), "descend_force": (0.0, 150.0)}


def add_v2_terms(cfg) -> None:
    """Add the v2 gap/stair terms to a Clean-arm teacher cfg (after its gap monitor). Observations unchanged."""
    from isaaclab.managers import CurriculumTermCfg, RewardTermCfg

    feet = SceneEntityCfg("robot", body_names=".*_foot")
    contacts = SceneEntityCfg("contact_forces", body_names=".*_foot")
    cfg.scene.terrain.terrain_generator.sub_terrains["platform_gap"].gap_width_range = V2_GAP_WIDTH_RANGE
    cfg.rewards.platform_gap_monitor.params["strict_contact"] = True
    w = V2_WEIGHTS
    cfg.rewards.gap_foothold_margin = RewardTermCfg(
        func=FootholdMargin, weight=w["gap_foothold_margin"],
        params={"mode": "gap", "asset_cfg": feet, "sensor_cfg": contacts, "margin": 0.04, "min_command": 0.2})
    cfg.rewards.stair_foothold_margin = RewardTermCfg(
        func=FootholdMargin, weight=w["stair_foothold_margin"],
        params={"mode": "stair", "asset_cfg": feet, "sensor_cfg": contacts, "margin": 0.04, "min_command": 0.2})
    cfg.rewards.gap_slot_probe = RewardTermCfg(
        func=SlotProbe, weight=w["gap_slot_probe"], params={"asset_cfg": feet, "sensor_cfg": contacts})
    # Order matters: the disturbance term first; fall/slip read its state in the same step.
    cfg.rewards.stair_handle_disturbance = RewardTermCfg(
        func=StairHandleDisturbance, weight=w["stair_handle_disturbance"],
        params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": contacts, **V2_DISTURBANCE})
    cfg.rewards.stair_push_fall = RewardTermCfg(func=stair_push_fall, weight=w["stair_push_fall"], params={})
    cfg.rewards.stair_push_slip = RewardTermCfg(
        func=stair_push_slip, weight=w["stair_push_slip"], params={"asset_cfg": feet, "sensor_cfg": contacts})
    cfg.curriculum.gap_stair_v2_diagnostics = CurriculumTermCfg(func=gap_stair_v2_diagnostics, params={})
