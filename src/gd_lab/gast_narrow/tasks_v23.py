"""GAST-Narrow v2.3: v2.2 (Top-1 371) + dead-end / L-bend emphasis, safe speed cap, leg-wall clearance.

E2 (2026-10-27) / MuJoCo on 371: gap, stairs, corridor and doors met, but
- dead end L7: stand 50 % (465: 100 %); L-bend tight (0.65 m): pass 62.5 % left and right;
- MuJoCo: legs catch the L's inner corner; a high joystick command (up to 1.2 m/s) drives into the walls --
  narrow commands were 0.35-0.80 m/s only, and tracking / overspeed / pace all followed the command.
v2.3 changes (anchor, observations, network, Top-5 as v2.2):
- terrain 36 -> 44 columns, nothing removed: dead end 2 -> 4, bend left/right 3 -> 4 each; the retention half kept
  at 50 % with platform_gap 8 -> 10 and pyramid_stairs / pyramid_stairs_inv 1 -> 2 each.
- commands: narrow forward command 0.35-1.20 m/s; half of the bend episodes aim the heading command at 0.4 m ahead
  (less inner-corner cutting), the other half keep 0.8 m (corner-cutting joystick input stays in the data).
- safe speed on narrow tiles before the exit: V_SAFE corridor/dead end 0.60, door/bend 0.45 m/s. Tracking pays for
  min(|command|, V_SAFE); overspeed and the pass-bonus pace use the limit min(|command| + 0.15, V_SAFE + 0.10).
- legs: 3 shin points per leg (knee -> foot) join the wall clearance (proximity penalty, touch, privileged d_l/d_r);
  narrow_leg_touch -1 /s per leg with any point within TOUCH of a wall.
- gap nose-up (MuJoCo on 371: the front rises ~20-30 deg, maybe less, crossing gaps forward AND backward): v2.2 had
  backward_nose_up only (-5, backward commands) and a forward pitch term of ~0.07 at 15 deg. gap_nose_up replaces it:
  -NOSE_WEIGHT per second per rad of nose-up above NOSE_LIMIT_DEG on gap tiles, both directions. The spikes are short
  (~0.07 s above 15 deg per episode), hence the large weight. The 6566 anchor skips the samples this term flags
  (gap tile and above the limit; launcher NARROW_ANCHOR_GAP_NOSE=1) -- 6566 itself lifts its front as much.
"""
import math

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from gd_lab.gast_narrow import anchor, geometry, mdp_v2
from gd_lab.gast_narrow.state import TRUNK_POINTS, narrow_state
from gd_lab.gast_narrow.tasks_v21 import OVERSPEED_MARGIN
from gd_lab.gast_narrow.tasks_v22 import PACE_EXTRA_M, GastNarrowV22RunnerCfg, GastNarrowV22TeacherCfg
from gd_lab.mdp.platform_gap_terms import boarding_family_mask
from gd_lab.mdp.rewards.tracking import _velocity_gate

COLUMNS = {"narrow_dead_end": 4.0, "narrow_bend_left": 4.0, "narrow_bend_right": 4.0,
           "platform_gap": 10.0, "pyramid_stairs": 2.0, "pyramid_stairs_inv": 2.0}
NUM_COLS = 44
VX_RANGE = (0.35, 1.20)
BEND_LOOKAHEAD, BEND_SHORT_P = 0.4, 0.5
V_SAFE = {"corridor": 0.60, "door": 0.45, "bend": 0.45, "dead_end": 0.60}
SAFE_TOLERANCE = 0.10
SHIN_FRACS = (0.25, 0.5, 0.75)
LEG_TOUCH_WEIGHT = -1.0
NOSE_LIMIT_DEG, NOSE_WEIGHT = 10.0, -30.0  # E2 (GH200, 2026-10-28): 371 forward L14 p90 25 deg, backward 9-11 deg


def _v_safe_table(device):
    table = torch.full((max(geometry.KINDS.values()) + 1,), 1.0e3, device=device)
    for kind, v in V_SAFE.items():
        table[geometry.KINDS[kind]] = v
    return table


def _capped(env):
    """(command [N,2], capped target [N,2], speed limit [N], planar velocity [N,2]) -- cap on narrow tiles only."""
    state = narrow_state(env)
    table = getattr(env, "_narrow_v23_vsafe", None)
    if table is None:
        table = env._narrow_v23_vsafe = _v_safe_table(env.device)
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    mag = cmd.norm(dim=-1)
    active = state.narrow & ~state.past_exit
    v_safe = torch.where(active, table[state.kind.long()], torch.full_like(mag, 1.0e3))
    target = cmd * (v_safe / mag.clamp_min(1e-6)).clamp(max=1.0)[:, None]
    limit = torch.minimum(mag + OVERSPEED_MARGIN, v_safe + SAFE_TOLERANCE)
    vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return cmd, target, limit, vel


def track_lin_vel_xy_capped(env, command_name: str, std_base: float = 0.3, alpha: float = 0.2,
                            cmd_threshold: float = 0.1, vel_frac_low: float = 0.4, vel_frac_high: float = 0.8,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """v2.2's gated tracking with the narrow target min(|cmd|, V_SAFE) (same kernel, gate and zone gating)."""
    _, target, _, vel = _capped(env)
    target_mag = target.norm(dim=-1)
    std = std_base + alpha * target_mag
    reward = torch.exp(-torch.sum(torch.square(target - vel), dim=-1) / (std * std))
    gate = _velocity_gate(target_mag, vel.norm(dim=-1), cmd_threshold, vel_frac_low, vel_frac_high)
    return reward * gate * mdp_v2._not_in_zone(env)


def narrow_overspeed_capped(env):
    """Per second on narrow tiles: planar speed above min(|cmd| + 0.15, V_SAFE + 0.10)."""
    _, _, limit, vel = _capped(env)
    return (vel.norm(dim=-1) - limit).clamp_min(0.0) * narrow_state(env).narrow.float()


class NarrowPacedPassBonusCapped(ManagerTermBase):
    """v2.2 paced pass bonus with the extra distance measured above the capped limit."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.extra = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.extra[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env):
        state = narrow_state(env)
        _, _, limit, vel = _capped(env)
        inside = state.narrow & ~state.past_exit
        self.extra += (vel.norm(dim=-1) - limit).clamp_min(0.0) * inside.float() * env.step_dt
        pace = (1.0 - self.extra / PACE_EXTRA_M).clamp(0.0, 1.0)
        return mdp_v2.narrow_pass_bonus(env) * pace


def narrow_leg_touch(env):
    """Per second on narrow tiles: number of legs (0-4) with a hip/calf/shin/foot point within TOUCH of a wall."""
    state = narrow_state(env)
    touch = state.clearance[:, len(TRUNK_POINTS):] < mdp_v2.TOUCH
    legs = getattr(env, "_narrow_v23_leg_index", None)
    if legs is None:
        legs = env._narrow_v23_leg_index = _leg_index(env, state)
    per_leg = torch.zeros(touch.shape[0], 4, device=env.device).index_add_(1, legs, touch.float())
    return (per_leg > 0).float().sum(-1) * state.narrow.float()


def _leg_index(env, state):
    """Leg id (0-3) of every non-trunk clearance point, in NarrowState's order (bodies, then shins)."""
    names = env.scene["robot"].body_names
    legs = sorted({names[i].split("_")[0] for i in state.body_ids.tolist()})
    index = [legs.index(names[i].split("_")[0]) for i in state.body_ids.tolist()]
    for calf in state.shin_calf:
        index += [legs.index(names[calf].split("_")[0])] * len(state.shin_fracs)
    return torch.tensor(index, device=env.device, dtype=torch.long)


def gap_nose_up(env, limit_deg: float = NOSE_LIMIT_DEG):
    """Per second on gap tiles, any command direction: nose-up pitch above ``limit_deg`` (rad).
    nose-up = asin(-g_x), g = projected gravity in the body frame (as gast_backgap backward_nose_up)."""
    g = env.scene["robot"].data.projected_gravity_b
    excess = (torch.asin((-g[:, 0]).clamp(-1.0, 1.0)) - math.radians(limit_deg)).clamp_min(0.0)
    gap = boarding_family_mask(env)
    anchor.set_gap_nose_mask(gap & (excess > 0))  # the 6566 anchor skips exactly these samples
    return excess * gap.float()


class NarrowV23VelocityCommand(mdp_v2.NarrowV2VelocityCommand):
    vx_range = VX_RANGE

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.short_lookahead = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if len(ids):
            self.short_lookahead[ids] = torch.rand(len(ids), device=self.device) < BEND_SHORT_P
        return super().reset(env_ids)

    def _lookahead(self, state):
        bend = state.kind == geometry.KINDS["bend"]
        short = torch.full_like(state.s, BEND_LOOKAHEAD)
        return torch.where(bend & self.short_lookahead, short, torch.full_like(state.s, mdp_v2.LOOKAHEAD))


@configclass
class GastNarrowV23TeacherCfg(GastNarrowV22TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        for name, columns in COLUMNS.items():
            gen.sub_terrains[name].proportion = columns
        gen.num_cols = int(sum(sub.proportion for sub in gen.sub_terrains.values()))
        assert gen.num_cols == NUM_COLS, gen.num_cols
        self.narrow_shin_fracs = SHIN_FRACS
        self.commands.base_velocity.class_type = NarrowV23VelocityCommand
        assert self.rewards.track_lin_vel_xy_exp.func is mdp_v2.track_lin_vel_xy_gated
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_capped
        self.rewards.narrow_overspeed.func = narrow_overspeed_capped
        self.rewards.narrow_success.func = NarrowPacedPassBonusCapped
        self.rewards.narrow_leg_touch = RewardTermCfg(func=narrow_leg_touch, weight=LEG_TOUCH_WEIGHT)
        del self.rewards.backward_nose_up  # superseded by gap_nose_up (both directions)
        self.rewards.gap_nose_up = RewardTermCfg(func=gap_nose_up, weight=NOSE_WEIGHT,
                                                 params={"limit_deg": NOSE_LIMIT_DEG})


@configclass
class GastNarrowV23RunnerCfg(GastNarrowV22RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/narrow_v23"


TASK = "GastNarrowV23"
TASK_ID = "Gd-GastNarrowV23-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV22-Rbq10-Dreamwaq-v0"
