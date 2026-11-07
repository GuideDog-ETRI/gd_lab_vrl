"""Backward gap stride terms. All are zero outside gap tiles while moving backward (gast_gait.terms.back_gap_mask)."""
from __future__ import annotations

import os

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from gd_lab.gast_gait import terms as gait

# A18: BACKSTRIDE_BOTH_DIR=1 applies the trot / stride / landing terms to forward gaps too (R17 off both ways).
BOTH_DIR = os.environ.get("BACKSTRIDE_BOTH_DIR", "0") == "1"


def _stride_mask(env) -> torch.Tensor:
    """[N] gap tiles while moving backward, or in either direction with BACKSTRIDE_BOTH_DIR=1."""
    if BOTH_DIR:
        return gait.boarding_family_mask(env) & (gait.motion_direction(env) != 0)
    return gait.back_gap_mask(env)


_back_gap_mask = None  # [num_envs] bool, this env step: _stride_mask (set by stance_spacing_floor; read by gast_backstride.anchor)


def back_gap_step_mask():
    return _back_gap_mask


def _side_pairs(robot, asset_cfg: SceneEntityCfg) -> list[tuple[int, int]]:
    """(front, rear) column pairs of asset_cfg.body_ids for the left and the right side (RBQ10 FL/FR/RL/RR_foot)."""
    names = [robot.body_names[i] for i in asset_cfg.body_ids]
    return [(names.index(f"F{s}_foot"), names.index(f"R{s}_foot")) for s in "LR"]


def stance_spacing_floor(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, floor: float = 0.40):
    """Per second on backward gaps: how far the same-side front-to-hind foot spacing (body x) falls short of
    ``floor`` m while both feet are in stance, summed over the two sides. Flat walking spacing is 0.6 m; the A'
    shuffle after a backward slot reaches 0 (front foot on the hind foot)."""
    global _back_gap_mask
    back_gap = gait.back_gap_mask(env)
    _back_gap_mask = _stride_mask(env)  # A18 (BOTH_DIR): forward gap steps leave the retention anchor too
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
    x = quat_apply_inverse(quat, feet - robot.data.root_pos_w[:, None, :])[..., 0]  # [N, F] body frame
    contact = gait._in_contact(env, sensor_cfg)
    cost = torch.zeros(env.num_envs, device=env.device)
    for front, rear in _side_pairs(robot, asset_cfg):
        both = contact[:, front] & contact[:, rear]
        cost = cost + (floor - (x[:, front] - x[:, rear])).clamp_min(0.0) * both
    return cost * back_gap.float()


def hind_stance_height_floor(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, floor: float = 0.45):
    """Per second on backward gaps: how far a planted hind foot sits less than ``floor`` m below the trunk (body z),
    summed over the hind feet (A6). Isaac -0.3 m/s replays (11-04): flat depth 0.48~0.50 m; near the gap p10 6566
    0.41, a1993 0.39, A3 0.37, A5 0.41 -- the hind legs fold and the trunk sits back while the front feet follow."""
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
    depth = -quat_apply_inverse(quat, feet - robot.data.root_pos_w[:, None, :])[..., 2]  # [N, F]
    contact = gait._in_contact(env, sensor_cfg)
    cost = torch.zeros(env.num_envs, device=env.device)
    for _, rear in _side_pairs(robot, asset_cfg):
        cost = cost + (floor - depth[:, rear]).clamp_min(0.0) * contact[:, rear]
    return cost * gait.back_gap_mask(env).float()


class FrontLiftOverSlot(ManagerTermBase):
    """Per second on backward gaps: a front foot in swing above a slot (x within ``reach`` of it) whose sole is less
    than ``target`` m above the deck around it (highest height-scan hit within ``radius`` of the foot). The trailing
    front feet are what shuffled across; the forward crossing lifts its leading feet clear of the far deck."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.slots = gait._GapSlots(env)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, target: float = 0.08,
                 reach: float = 0.05, radius: float = 0.20, scanner: str = "height_scanner"):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        robot = env.scene[asset_cfg.name]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids]  # [N, F, 3]
        slots, _ = self.slots.geometry.for_envs(env)          # [N, 2, 2] env-frame x0, x1
        x = feet[..., 0] - env.scene.env_origins[:, None, 0]
        over = ((x[..., None] > slots[:, None, :, 0] - reach) & (x[..., None] < slots[:, None, :, 1] + reach)).any(-1)
        hits = env.scene.sensors[scanner].data.ray_hits_w      # [N, R, 3]
        z = torch.nan_to_num(hits[..., 2], nan=-1e3, posinf=-1e3, neginf=-1e3)
        near = ((feet[:, :, None, :2] - hits[:, None, :, :2]) ** 2).sum(-1) < radius ** 2  # [N, F, R]
        deck = torch.where(near, z[:, None, :], torch.full_like(near, -1e3, dtype=z.dtype)).amax(-1)
        clearance = feet[..., 2] - gait.FOOT_RADIUS - deck
        names = [robot.body_names[i] for i in asset_cfg.body_ids]
        front = torch.tensor([n.startswith("F") for n in names], device=env.device)
        swing = ~gait._in_contact(env, sensor_cfg)
        short = (target - clearance).clamp_min(0.0) * (over & swing & front[None] & (deck > -1e2))
        return short.sum(dim=1) * gait.back_gap_mask(env).float()


def back_gap_overextension(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg,
                           limit: float = 0.30):
    """R9 on backward gaps only, at ``limit`` (A' R19 turned R9 off there): a brief hind-foot brace stays free, a
    stance foot left more than 30 cm behind its hip along the motion is charged."""
    cost = gait.stance_overextension(env, sensor_cfg, asset_cfg, hip_cfg, limit=limit, gap_limit=limit, gap_scale=1.0)
    return cost * gait.back_gap_mask(env).float()


def touchdown_placement(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg,
                        front_max: float = 0.15, hind_min: float = 0.18):
    """Event (/dt) on backward gaps: where each foot lands relative to its hip along the motion (body x; + = placed
    toward the motion). Isaac back-gap replay -0.3 m/s (11-03), touchdown median: a1993 front +0.21 / hind +0.12 m,
    6566 front +0.12 / hind +0.24 -- the same stride length, but A' lands the trailing front feet deep under the
    body and does not reach back with the leading hind feet, so the front feet come down on the hind ones. Charged:
    a front foot landing more than ``front_max`` toward the motion, a hind foot landing less than ``hind_min``
    (the forward crossing mirrored: the leading feet reach, the trailing feet land near their hips)."""
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    hips = robot.data.body_pos_w[:, hip_cfg.body_ids]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
    dx = quat_apply_inverse(quat, feet - hips)[..., 0]                   # [N, F] foot ahead of hip (+x body)
    reach = gait.motion_direction(env)[:, None] * dx                     # + = toward the motion
    names = [robot.body_names[i] for i in asset_cfg.body_ids]
    front = torch.tensor([n.startswith("F") for n in names], device=env.device)[None]
    miss = torch.where(front, (reach - front_max).clamp_min(0.0), (hind_min - reach).clamp_min(0.0))
    sensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    return (miss * first).sum(dim=1) * gait.back_gap_mask(env).float() / env.step_dt


class GapSupportForwardOnly(gait.GapSupport):
    """A' R17 (at least 3 feet down near a slot) everywhere except backward gaps (A4). MuJoCo A3 (11-03): the hind
    foot a crossing front foot will land on keeps bracing instead of stepping back out of its way; the trunk sinks
    backward behind the hind support ("sits down, then goes back"). Stepping that hind foot back while the other front
    foot is in the air means two feet down -- exactly what R17 charges."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, min_feet: int = 3,
                 command_name: str = "base_velocity"):
        cost = super().__call__(env, sensor_cfg, asset_cfg, min_feet, command_name)
        return cost * (~gait.back_gap_mask(env)).float()


class GapSupportBackBoost(gait.GapSupport):
    """A' R17 with the backward-gap part scaled by ``back_scale`` (A8). The user's sketch (11-04): cross a backward slot
    one foot at a time -- hind 1 across, the same-side front 3 into 1's vacated print, hind 2 across, front 4 into 2's
    print, then 3 and 4 across -- three feet down throughout. R17 already prices feet short of three near a slot, but
    at -1 /s the backward crossing stays a trot (diagonal liftoff lag: the hind foot first 52-84 %, front liftoff on two
    feet 60-77 %)."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, min_feet: int = 3,
                 command_name: str = "base_velocity", back_scale: float = 4.0):
        cost = super().__call__(env, sensor_cfg, asset_cfg, min_feet, command_name)
        return cost * (1.0 + (back_scale - 1.0) * gait.back_gap_mask(env).float())


class _BackGapSequence(ManagerTermBase):
    """Footstep sequence of a backward crossing, the forward one mirrored (A5). Forward gap replay a1993 +0.5 m/s
    (11-03): the leading front foot lands first (97 %), the trailing hind foot lifts with the other three feet down
    (81 %) and lands 0.30 m from the leader's latest footprint. Backward (a1993 / A3 / 6566 at -0.3 m/s): trailing
    front feet lift on two feet 61 / 76 / 77 % of the time, 0.36 / 0.56 / 0.60 m from the leading hind footprint --
    a trot, the two-point moment that wobbles. Per side the leader is the hind foot, the trailer the front foot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset = cfg.params["asset_cfg"]
        robot = env.scene[asset.name]
        names = [robot.body_names[i] for i in asset.body_ids]
        sensor_cfg = cfg.params["sensor_cfg"]
        sensed = [env.scene.sensors[sensor_cfg.name].body_names[i] for i in sensor_cfg.body_ids]
        if sensed != names:  # contact columns and body positions are indexed alike in every term below
            raise ValueError(f"backstride: contact sensor feet {sensed} != robot feet {names}")
        self.pairs = [(names.index(f"R{s}_foot"), names.index(f"F{s}_foot")) for s in "LR"]  # (leader, trailer)
        self.leader_stepped = torch.ones(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.leader_print = torch.zeros(env.num_envs, 2, device=env.device)  # world x of the latest hind touchdown
        self.has_print = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.leader_stepped[ids] = True
        self.has_print[ids] = False

    def _events(self, env, sensor_cfg, asset_cfg):
        sensor = env.scene.sensors[sensor_cfg.name]
        first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        first_air = sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
        contact = gait._in_contact(env, sensor_cfg)
        feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
        return first_contact, first_air, contact, feet


class TrailerLiftoffOrder(_BackGapSequence):
    """Event (/dt) on backward gaps: a front foot lifting before its same-side hind foot stepped since the front
    foot's last touchdown (mirror of "the leading foot crosses first")."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
        first_contact, first_air, _, _ = self._events(env, sensor_cfg, asset_cfg)
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, trail) in enumerate(self.pairs):
            self.leader_stepped[:, k] |= first_contact[:, lead]
            cost = cost + (first_air[:, trail] & ~self.leader_stepped[:, k]).float()
            self.leader_stepped[:, k] &= ~first_contact[:, trail]
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class TrailerLiftoffSupport(_BackGapSequence):
    """Event (/dt) on backward gaps: feet short of three planted (the other three) when a front foot lifts."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
        _, first_air, contact, _ = self._events(env, sensor_cfg, asset_cfg)
        down = contact.sum(dim=1).float()
        cost = torch.zeros(env.num_envs, device=env.device)
        for _, trail in self.pairs:
            cost = cost + first_air[:, trail].float() * (3.0 - down).clamp_min(0.0)
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class TrailerFootprint(_BackGapSequence):
    """Event (/dt) on backward gaps: a front foot touchdown farther than ``tol`` from ``target`` m (world x) off the
    same-side hind foot's latest footprint (forward crossing: the trailer lands 0.30 m from the leader's print)."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, target: float = 0.30,
                 tol: float = 0.10):
        first_contact, _, _, feet = self._events(env, sensor_cfg, asset_cfg)
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, trail) in enumerate(self.pairs):
            land = first_contact[:, trail] & self.has_print[:, k]
            miss = ((feet[:, trail, 0] - self.leader_print[:, k]).abs() - target).abs() - tol
            cost = cost + miss.clamp_min(0.0) * land
            new = first_contact[:, lead]
            self.leader_print[:, k] = torch.where(new, feet[:, lead, 0], self.leader_print[:, k])
            self.has_print[:, k] |= new
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class TrailerIntoVacated(_BackGapSequence):
    """Event (/dt) on backward gaps: a front foot touchdown farther than ``tol`` m (world x) from the spot its same-side
    hind foot vacated (the hind foot's position at its latest liftoff) -- the sketch's "3 into 1's print" (A8)."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, tol: float = 0.08):
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, trail) in enumerate(self.pairs):
            land = first_contact[:, trail] & self.has_print[:, k]
            cost = cost + ((feet[:, trail, 0] - self.leader_print[:, k]).abs() - tol).clamp_min(0.0) * land
            new = first_air[:, lead]
            self.leader_print[:, k] = torch.where(new, feet[:, lead, 0], self.leader_print[:, k])
            self.has_print[:, k] |= new
        return cost * gait.back_gap_mask(env).float() / env.step_dt


def _travel_x(env) -> torch.Tensor:
    """[N] +1 / -1: the world x direction of travel (command sign x heading); gap slots run across x."""
    heading = env.scene["robot"].data.heading_w
    return gait.motion_direction(env) * torch.sign(torch.cos(heading))


class _SlotSequence(_BackGapSequence):
    """Footstep sequence relative to the slots (A9). The user's sketch (11-04): hind 1 crosses, front 3 steps into
    the spot 1 stood on *before* the slot, hind 2 crosses, front 4 into 2's pre-slot spot, then 3 and 4 cross. A8 kept
    the hind print of every liftoff, including the steps on the far deck, so the front target drifted across."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.slots = gait._GapSlots(env)

    def _slot_frame(self, env, feet):
        """(foot x minus the nearest slot centre, signed along travel [N, F]; half widths [N, F]); far side > 0."""
        slots, _ = self.slots.geometry.for_envs(env)                       # [N, S, 2] env-frame x0, x1
        x = feet[..., 0] - env.scene.env_origins[:, None, 0]               # [N, F]
        centre = slots.mean(-1)                                             # [N, S]
        rel = x[..., None] - centre[:, None, :]                             # [N, F, S]
        j = rel.abs().argmin(-1, keepdim=True)
        rel = rel.gather(-1, j).squeeze(-1)
        half = ((slots[..., 1] - slots[..., 0]) / 2)[:, None, :].expand(-1, x.shape[1], -1).gather(-1, j).squeeze(-1)
        return rel * _travel_x(env)[:, None], half


class TrailerIntoPreSlotSpot(_SlotSequence):
    """Event (/dt) on backward gaps: a front foot, still before the slot, touching down farther than ``tol`` m (world
    x) from the spot its same-side hind foot left when it crossed (hind liftoff before the slot, within ``reach`` m of
    its near edge). One target per crossing; cleared when the front lands on it or crosses."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, tol: float = 0.08,
                 reach: float = 0.35):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        rel, half = self._slot_frame(env, feet)
        before = rel < -half                                                  # on the near deck
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, trail) in enumerate(self.pairs):
            launch = first_air[:, lead] & before[:, lead] & (rel[:, lead] > -half[:, lead] - reach) & before[:, trail]
            self.leader_print[:, k] = torch.where(launch, feet[:, lead, 0], self.leader_print[:, k])
            self.has_print[:, k] |= launch
            land = first_contact[:, trail] & self.has_print[:, k] & before[:, trail]
            miss = ((feet[:, trail, 0] - self.leader_print[:, k]).abs() - tol).clamp_min(0.0)
            cost = cost + miss * land
            self.has_print[:, k] &= ~((land & (miss == 0)) | ~before[:, trail])
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class FrontCrossesFirst(_SlotSequence):
    """Per second on backward gaps: a front foot over the slot (within ``reach`` of it, in swing) while a hind foot
    is still before it -- the sketch crosses both hind feet before either front foot."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, reach: float = 0.05):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
        rel, half = self._slot_frame(env, feet)
        swing = ~gait._in_contact(env, sensor_cfg)
        leads = [lead for lead, _ in self.pairs]; trails = [trail for _, trail in self.pairs]
        hind_before = (rel[:, leads] < -half[:, leads]).any(-1)
        front_over = ((rel[:, trails].abs() < half[:, trails] + reach) & swing[:, trails]).any(-1)
        return (front_over & hind_before).float() * gait.back_gap_mask(env).float()


class FrontCrossFromPreSlotSpot(_SlotSequence):
    """Event (/dt) on backward gaps, at a front foot's crossing (its first touchdown past the slot after a liftoff
    before it): how far beyond ``tol`` m (world x) that liftoff was from the spot its same-side hind foot crossed
    from (A10). MuJoCo A9 (11-04): the front feet stopped 0.31-0.35 m short of that spot and crossed in one long step;
    a charge on every front touchdown (A9) left that route free."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lift_x = torch.zeros(env.num_envs, 2, device=env.device)
        self.lift_before = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.lift_before[ids] = False

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, tol: float = 0.12,
                 reach: float = 0.35):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        rel, half = self._slot_frame(env, feet)
        before = rel < -half
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, trail) in enumerate(self.pairs):
            launch = first_air[:, lead] & before[:, lead] & (rel[:, lead] > -half[:, lead] - reach) & before[:, trail]
            self.leader_print[:, k] = torch.where(launch, feet[:, lead, 0], self.leader_print[:, k])
            self.has_print[:, k] |= launch
            lift = first_air[:, trail]
            self.lift_x[:, k] = torch.where(lift, feet[:, trail, 0], self.lift_x[:, k])
            self.lift_before[:, k] = torch.where(lift, before[:, trail], self.lift_before[:, k])
            cross = first_contact[:, trail] & ~before[:, trail] & self.lift_before[:, k] & self.has_print[:, k]
            miss = ((self.lift_x[:, k] - self.leader_print[:, k]).abs() - tol).clamp_min(0.0)
            cost = cost + miss * cross
            self.has_print[:, k] &= ~cross
            self.lift_before[:, k] &= ~first_contact[:, trail]
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class HindStepsAsideFirst(_SlotSequence):
    """Event (/dt) on backward gaps: a front foot touching down before the slot after its same-side hind foot crossed
    but before that hind foot took one more step on the far deck (A11). MuJoCo A10 (11-04): hip spacing 0.62 m, so
    with the hind one step (~0.3 m) past the slot the front cannot reach the hind's pre-slot spot; the sketch's
    "3 into 1's spot" needs 1 to step back once more first."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hind_lift_before = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.crossed = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.aside = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.hind_lift_before[ids] = False
        self.crossed[ids] = False
        self.aside[ids] = False

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        rel, half = self._slot_frame(env, feet)
        before = rel < -half
        past = rel > half
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, trail) in enumerate(self.pairs):
            self.hind_lift_before[:, k] = torch.where(first_air[:, lead], before[:, lead], self.hind_lift_before[:, k])
            land = first_contact[:, lead] & past[:, lead]
            self.aside[:, k] |= land & self.crossed[:, k]                    # second far-deck touchdown
            new = land & self.hind_lift_before[:, k] & ~self.crossed[:, k]   # the crossing touchdown
            self.crossed[:, k] |= new
            self.aside[:, k] &= ~new
            front_down = first_contact[:, trail]
            cost = cost + (front_down & before[:, trail] & self.crossed[:, k] & ~self.aside[:, k]).float()
            done = front_down & ~before[:, trail]                            # the front crossed: next slot
            self.crossed[:, k] &= ~done
            self.aside[:, k] &= ~done
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class BackGapNoseDown(ManagerTermBase):
    """Per second, a bonus on backward gaps while a foot is within 0.5 m of a slot: the trunk pitched nose-down (rear
    raised) -- 0 at level, 1 from ``low_deg`` to ``high_deg``, back to 0 at ``zero_deg`` (R1 charges beyond 10 deg near
    slots) (A12). Leg 0.66 m: 10 deg lowers the front hips 5 cm and lets the front feet reach 0.47 m (0.41 level)
    behind them, toward the hind pre-slot spot; MuJoCo A11 crossed at +2.3 deg nose-up with the rear sagging."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.slots = gait._GapSlots(env)

    def __call__(self, env, asset_cfg: SceneEntityCfg, low_deg: float = 3.0, high_deg: float = 8.0,
                 zero_deg: float = 10.0):
        down = torch.rad2deg(-gait.nose_up_angle(env))
        ramp_up = (down / low_deg).clamp(0.0, 1.0)
        ramp_down = ((zero_deg - down) / (zero_deg - high_deg)).clamp(0.0, 1.0)
        feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
        near = self.slots.foot_over_slot(env, feet)
        return torch.minimum(ramp_up, ramp_down) * near.float() * gait.back_gap_mask(env).float()


def touchdown_onto_pair_foot(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, clear: float = 0.30):
    """Event (/dt) on backward gaps: a foot landing with its planted same-side partner closer than ``clear`` m
    (front-to-hind spacing, body x), per m short (A13). MuJoCo A10 / A11 (11-04): after a backward slot the
    following foot lands on or right next to the foot that crossed (spacing min -0.03 / 0.04 m) and the stance
    collapses; the per-second floor (0.40 m, -6) is too soft to stop it."""
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact = gait._in_contact(env, sensor_cfg)
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
    x = quat_apply_inverse(quat, feet - robot.data.root_pos_w[:, None, :])[..., 0]
    cost = torch.zeros(env.num_envs, device=env.device)
    for front, rear in _side_pairs(robot, asset_cfg):
        land = (first_contact[:, front] & contact[:, rear]) | (first_contact[:, rear] & contact[:, front])
        cost = cost + (clear - (x[:, front] - x[:, rear])).clamp_min(0.0) * land
    return cost * gait.back_gap_mask(env).float() / env.step_dt


class LeaderCrossingReach(_SlotSequence):
    """Event (/dt) on backward gaps: a hind foot's crossing touchdown (first landing past the slot after a liftoff
    before it) less than ``target`` m beyond its hip along the motion, per m short (A14). The user's real-car clip
    (11-04): the first crossing foot stretches as far as it can so the following foot has room behind it; MuJoCo
    A10-A13 hind feet land 0.12-0.18 m past the hip and the following foot comes down on them. Leg 0.66 m at
    0.50 m hip height reaches about 0.41 m."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lift_before = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.lift_before[ids] = False

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg,
                 target: float = 0.32):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        rel, half = self._slot_frame(env, feet)
        robot = env.scene[asset_cfg.name]
        hips = robot.data.body_pos_w[:, hip_cfg.body_ids]
        quat = robot.data.root_quat_w[:, None, :].expand(-1, feet.shape[1], -1)
        reach = gait.motion_direction(env)[:, None] * quat_apply_inverse(quat, feet - hips)[..., 0]
        cost = torch.zeros(env.num_envs, device=env.device)
        for k, (lead, _) in enumerate(self.pairs):
            self.lift_before[:, k] = torch.where(first_air[:, lead], rel[:, lead] < -half[:, lead],
                                                 self.lift_before[:, k])
            cross = first_contact[:, lead] & (rel[:, lead] > half[:, lead]) & self.lift_before[:, k]
            cost = cost + (target - reach[:, lead]).clamp_min(0.0) * cross
            self.lift_before[:, k] &= ~first_contact[:, lead]
        return cost * gait.back_gap_mask(env).float() / env.step_dt


class BackGapStrideFloor(_SlotSequence):
    """Event (/dt) on backward gaps while a foot is within 0.5 m of a slot: a touchdown whose step (world x from its
    liftoff, along the motion) is shorter than ``floor`` m, per m short, every foot (A15). The user (11-04): cross in
    a wide trot instead of one foot at a time. A14 strides 0.36 / 0.50 m crossed every MuJoCo gap with 22 % of the
    front liftoffs on three feet; equal long steps on all four legs keep the same-side spacing near the flat 0.6 m."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lift_x = torch.zeros(env.num_envs, 4, device=env.device)
        self.lifted = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.lifted[ids] = False

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, floor: float = 0.40):
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        self.lift_x = torch.where(first_air, feet[..., 0], self.lift_x)
        self.lifted |= first_air
        step = _travel_x(env)[:, None] * (feet[..., 0] - self.lift_x)
        land = first_contact & self.lifted
        cost = ((floor - step).clamp_min(0.0) * land).sum(dim=1)
        self.lifted &= ~first_contact
        near = self.slots.foot_over_slot(env, feet)
        return cost * near.float() * _stride_mask(env).float() / env.step_dt


class BackGapCrossingStep(_SlotSequence):
    """Event (/dt) on backward gaps: a foot's crossing step (liftoff before the slot, first touchdown past it) shorter
    than ``floor`` m, or landing less than ``land_past`` m beyond the far edge, per m short; all four feet (A16). The
    user (11-04, A15 side view): the trot is right but the reversing hind feet, and the following front feet, cross
    in short steps (A15 near-slot steps 0.37 m at -0.3 m/s). Leg reach ~0.41 m either side of the hip."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lift_x = torch.zeros(env.num_envs, 4, device=env.device)
        self.lift_before = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.lift_before[ids] = False

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, floor: float = 0.55,
                 land_past: float = 0.15):
        if self.slots.geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        rel, half = self._slot_frame(env, feet)
        self.lift_x = torch.where(first_air, feet[..., 0], self.lift_x)
        self.lift_before = torch.where(first_air, rel < -half, self.lift_before)
        cross = first_contact & self.lift_before & (rel > half)
        step = _travel_x(env)[:, None] * (feet[..., 0] - self.lift_x)
        short = (floor - step).clamp_min(0.0) + (land_past - (rel - half)).clamp_min(0.0)
        self.lift_before &= ~first_contact
        return (short * cross).sum(dim=1) * gait.back_gap_mask(env).float() / env.step_dt


class HindSwingDiagonalPlanted(_SlotSequence):
    """/s on backward gaps while a foot is within 0.5 m of a slot: a hind foot in the air while its diagonal front
    foot is planted (a three-point crossing), per hind foot (A17). Replay 11-05 (two seeds): A15 hind steps near the
    slot fly on three planted feet 14-20 % (flat 1-4 %), the diagonal front airborne with them 79-83 % (flat 95-99 %);
    a1993 learned the three-point crossing from R17 and 150 it with R17 off backward left it."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene[cfg.params["asset_cfg"].name]
        names = [robot.body_names[i] for i in cfg.params["asset_cfg"].body_ids]
        self.diag = [(names.index("RL_foot"), names.index("FR_foot")), (names.index("RR_foot"), names.index("FL_foot"))]

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
        _, _, contact, feet = self._events(env, sensor_cfg, asset_cfg)
        cost = sum((~contact[:, h] & contact[:, f]).float() for h, f in self.diag)
        near = self.slots.foot_over_slot(env, feet)
        return cost * near.float() * _stride_mask(env).float()


class CrossingLandingMargin(_SlotSequence):
    """Event (/dt) on gap tiles (backward, or both ways with BACKSTRIDE_BOTH_DIR=1): a foot's first touchdown past
    the slot (lifted before it) less than ``land_past`` m beyond the far edge, per m short, every foot (A18). A17
    -0.5 m/s replay (11-05): 3 of 4 falls were a crossing hind foot landing on the far edge and sliding in (hind
    landing past the edge p10 0.03 m); A16 paired this with a 0.55 m step floor and only the front steps grew."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lift_before = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.lift_before[ids] = False

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, land_past: float = 0.10):
        first_contact, first_air, _, feet = self._events(env, sensor_cfg, asset_cfg)
        rel, half = self._slot_frame(env, feet)
        self.lift_before = torch.where(first_air, rel < -half, self.lift_before)
        # past the near edge: the far deck, its edge, or inside the slot (review 11-05: inside / on-edge landings
        # must not be free); inside costs the full land_past plus the depth short of the far edge, capped at 2 x
        cross = first_contact & self.lift_before & (rel > -half)
        short = (land_past - (rel - half)).clamp(0.0, 2.0 * land_past)
        self.lift_before &= ~first_contact
        return (short * cross).sum(dim=1) * _stride_mask(env).float() / env.step_dt
