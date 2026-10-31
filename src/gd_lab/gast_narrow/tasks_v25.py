"""GAST-Narrow v2.5: v2.4 + cliff tiles (stop before an uncrossable drop, all four sides) + turn success redefined.

Design: docs/design/GAST_NARROW_V25_DESIGN.md. Start: v24h Top-1 it653 (stand anchor target stays 602).

Changes over v2.4 (tasks_v24):
- terrain 62 -> 65 columns: narrow_cliff x3, a square deck (half-size 2.2 -> 1.6 m) raised 1.0 m, drop on all sides.
- cliff episodes (mdp_cliff): constant body-frame command toward one side (front 40 %, back / left / right 20 %),
  zone 0.6 m from the edge: tracking off, cliff_stand +1 /s, v2 hold bonus after 1.5 s filtered stillness; a point
  over the edge = breach (-4, episode over), trunk below the deck = fall (episode over). Own curriculum rule.
- turn tiles: pass = reached + drift <= 0.30 + leg contact peak < 150 N (trunk contact / fall end the episode); clean
  (drift <= 0.10, no touch) is reported. turn_drift -3 /s (was -6), wall-proximity sum capped at 2 /s, leg touch -0.5.
- narrow_nose_up -3 /s above 10 deg when standing slowly where the robot must stand, above 20 deg elsewhere on
  narrow-space tiles (replaces v2.4e stand_pitch).
- base_height on cliff tiles ignores rays over the void; NarrowStallPenalty / centering off on cliff tiles; the
  v2.4e stand-anchor extension to blocked zones stays off (anchor.py, NARROW_STAND_ANCHOR_ZONE).
"""
from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_narrow import mdp_cliff, mdp_turn, mdp_v2
from gd_lab.gast_narrow.state import narrow_state
from gd_lab.gast_narrow.tasks_v22 import NarrowStallPenalty
from gd_lab.gast_narrow.tasks_v23 import track_lin_vel_xy_capped
from gd_lab.gast_narrow.tasks_v24 import GastNarrowV24RunnerCfg, GastNarrowV24TeacherCfg, NarrowV24VelocityCommand
from gd_lab.gast_narrow.terrain import NarrowPassageTerrainCfg

CLIFF_FAMILY = ("narrow_cliff", "cliff", 1, 3.0, 2.2, 1.6)  # name, kind, sign, columns, L at level 0, L at level 14
NUM_COLS = 65
CLIFF_STAND_WEIGHT, NOSE_UP_WEIGHT = 1.0, -3.0
TURN_DRIFT_WEIGHT, LEG_TOUCH_WEIGHT = -3.0, -0.5


class NarrowV25Monitor(mdp_turn.NarrowV24Monitor):
    """v2 monitor + turn + cliff trackers, plus the nose-up posture record used by the online collector."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self.nose_max = torch.zeros(n, device=d)
        self.nose_t10 = torch.zeros(n, device=d)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.nose_max[ids] = 0.0
        self.nose_t10[ids] = 0.0

    def __call__(self, env):
        out = mdp_v2.NarrowV2Monitor.__call__(self, env)
        mdp_turn.update_turn(env, self)
        mdp_cliff.update_cliff(env, self)
        g = env.scene["robot"].data.projected_gravity_b
        nose = torch.asin((-g[:, 0]).clamp(-1.0, 1.0))
        narrow = narrow_state(env).narrow
        self.nose_max = torch.where(narrow, torch.maximum(self.nose_max, nose), self.nose_max)
        self.nose_t10 += (narrow & (nose > 0.17453)).float() * env.step_dt
        mdp_turn.anchor.set_stand_mask(mdp_turn.should_stand(env))
        return out


class NarrowStallPenaltyV25(NarrowStallPenalty):
    """v2.2's stall penalty, off on cliff tiles (standing is the task there; blocked is False so v2.2 would fire)."""

    def __call__(self, env):
        return super().__call__(env) * (narrow_state(env).kind != mdp_cliff.CLIFF).float()


def track_lin_vel_xy_cliff(env, command_name: str, std_base: float = 0.3, alpha: float = 0.2,
                           cmd_threshold: float = 0.1, vel_frac_low: float = 0.4, vel_frac_high: float = 0.8,
                           asset_cfg=None):
    """v2.3 tracking, also off in the cliff zone (the command keeps asking to move there)."""
    kwargs = dict(std_base=std_base, alpha=alpha, cmd_threshold=cmd_threshold, vel_frac_low=vel_frac_low,
                  vel_frac_high=vel_frac_high)
    if asset_cfg is not None:
        kwargs["asset_cfg"] = asset_cfg
    return track_lin_vel_xy_capped(env, command_name, **kwargs) * (~mdp_cliff.cliff_zone(env)).float()


def narrow_centering_no_turn_no_cliff(env, sigma: float = 0.10):
    return mdp_turn.narrow_centering_no_turn(env, sigma) * (narrow_state(env).kind != mdp_cliff.CLIFF).float()


def narrow_levels_v25(env, env_ids, min_moving_time: float = 2.0):
    """v2.4 rule off the cliff tiles; cliff tiles: mdp_cliff.cliff_levels."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    on_cliff = (narrow_state(env).kind[ids] == mdp_cliff.CLIFF)
    other = ids[~on_cliff]
    if len(other):
        mdp_turn.narrow_levels_v24(env, other, min_moving_time)
    mdp_cliff.cliff_levels(env, ids[on_cliff])
    return env.scene.terrain.terrain_levels.float().mean()


class NarrowV25VelocityCommand(mdp_cliff.NarrowV25CliffCommandMixin, NarrowV24VelocityCommand):
    def _apply_narrow(self):
        NarrowV24VelocityCommand._apply_narrow(self)
        self._apply_cliff()


@configclass
class GastNarrowV25TeacherCfg(GastNarrowV24TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        size = next(iter(gen.sub_terrains.values())).size
        name, kind, sign, columns, l_max, l_min = CLIFF_FAMILY
        gen.sub_terrains[name] = NarrowPassageTerrainCfg(proportion=columns, size=size, kind=kind, sign=sign,
                                                         w_max=l_max, w_min=l_min)
        gen.num_cols = int(sum(sub.proportion for sub in gen.sub_terrains.values()))
        assert gen.num_cols == NUM_COLS, gen.num_cols

        self.observations.narrow_priv.state = ObservationTermCfg(func=mdp_cliff.narrow_privileged_v25)
        self.commands.base_velocity.class_type = NarrowV25VelocityCommand
        self.events.reset_cliff = EventTermCfg(func=mdp_cliff.reset_cliff, mode="reset")  # after reset_turn / back

        assert self.rewards.narrow_monitor.func is mdp_turn.NarrowV24Monitor
        self.rewards.narrow_monitor.func = NarrowV25Monitor
        self.rewards.cliff_stand = RewardTermCfg(func=mdp_cliff.cliff_stand, weight=CLIFF_STAND_WEIGHT)
        self.rewards.stand_pitch = None  # v2.4e term, replaced by narrow_nose_up
        self.rewards.narrow_nose_up = RewardTermCfg(func=mdp_turn.narrow_nose_up, weight=NOSE_UP_WEIGHT)
        self.rewards.turn_drift.weight = TURN_DRIFT_WEIGHT
        self.rewards.narrow_leg_touch.weight = LEG_TOUCH_WEIGHT
        assert self.rewards.narrow_stall.func is NarrowStallPenalty
        self.rewards.narrow_stall.func = NarrowStallPenaltyV25
        assert self.rewards.track_lin_vel_xy_exp.func is track_lin_vel_xy_capped
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_cliff
        assert self.rewards.narrow_centering.func is mdp_turn.narrow_centering_no_turn
        self.rewards.narrow_centering.func = narrow_centering_no_turn_no_cliff
        assert self.rewards.narrow_wall_proximity.func is mdp_v2.narrow_wall_proximity_v2
        self.rewards.narrow_wall_proximity.func = mdp_turn.narrow_wall_proximity_capped
        from gd_lab.mdp.platform_gap_terms import platform_gap_base_height
        assert self.rewards.base_height.func is platform_gap_base_height
        self.rewards.base_height.func = mdp_cliff.base_height_cliff
        self.terminations.cliff_fall = TerminationTermCfg(func=mdp_cliff.cliff_fall_done)
        self.curriculum.terrain_levels.func = narrow_levels_v25


@configclass
class GastNarrowV25RunnerCfg(GastNarrowV24RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/narrow_v25"


TASK = "GastNarrowV25"
TASK_ID = "Gd-GastNarrowV25-Rbq10-Dreamwaq-v0"
