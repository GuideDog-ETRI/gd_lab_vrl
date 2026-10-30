"""GAST-Narrow v2.4: v2.3 (Hangang it602) + in-place turns in confined pockets (design: GAST_NARROW_V24_TURN_DESIGN.md).

Changes (network, anchor, other narrow tiles and the retention tiles as v2.3):
- terrain 44 -> 58 columns: turn pockets aisle 2, vestibule 1+1, room 1+1 (free radius 1.00 m at level 0 -> 0.45 m
  at level 14) and a tight room (0.50 -> 0.40 m, never turnable), plus platform_gap +4, pyramid_stairs /
  pyramid_stairs_inv / random_rough +1 each to keep the retention half at 50 %.
- LiDAR: head Ouster + head Mid-360 + rear Mid-360 on the tail, trunk self-occlusion, same map [2, 30, 48]
  (lidar.NarrowLidarMapV24).
- turn tiles (mdp_turn): command vx = vy = 0, wz = +/-(0.3..0.8) rad/s for a 45..180 deg target (right 70 %);
  turnable = r_free >= R_SWEEP + m; success -> the v2 pass / hold / breach events; new rewards turn_stand +0.5,
  turn_drift -2, turn_overspin -1; yaw tracking off where the turn is not possible; centering off; drift > 0.2 m ends.
- critic: the same 9 narrow inputs, turn state in slots 2 / 4 / 7 / 8 on turn tiles.
- standing under a moving command (turn impossible, v2 blocked zone): feet_cadence_overrun off there (it paid for
  marching in place), stand_still_feet_contact on there.
- gap nose-up (602 still lifts its front crossing gaps: E2 mean 10.4 deg, p90 13 deg; MuJoCo more): gap_nose_up limit
  10 -> 5 deg (the anchor skips the flagged samples as before), and gap_nose_peak pays -NOSE_PEAK_WEIGHT per rad of the
  crossing's peak nose-up above 5 deg at the moment the crossing is credited (the per-second term alone is cheap for
  short spikes: ~0.07 s above 15 deg per crossing).
"""
import math

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

import gd_lab.mdp.rewards as gd_rew
from gd_lab.gast_narrow import mdp, mdp_turn, mdp_v2
from gd_lab.gast_narrow.lidar import NarrowLidarMapV24
from gd_lab.gast_narrow.tasks_v23 import GastNarrowV23RunnerCfg, GastNarrowV23TeacherCfg, NarrowV23VelocityCommand
from gd_lab.gast_narrow.terrain import NarrowPassageTerrainCfg
from gd_lab.mdp.platform_gap_terms import boarding_family_mask

TURN_FAMILIES = {
    # name: (shape, sign, columns)
    "narrow_turn_aisle": ("turn_aisle", 1, 2.0),
    "narrow_turn_vestibule": ("turn_vestibule", 1, 1.0),
    "narrow_turn_vestibule_neg": ("turn_vestibule", -1, 1.0),
    "narrow_turn_room": ("turn_room", 1, 1.0),
    "narrow_turn_room_neg": ("turn_room", -1, 1.0),
}
TURN_R_MAX, TURN_R_MIN = 1.00, 0.45
# Always too tight for the turn (r 0.50 -> 0.40 < R_SWEEP + m at every m): the "stop" case from level 0 on. In the
# other pockets r falls below the threshold only from level 10 (3 %) to 13 (100 %), so without this family the policy
# would meet "turn impossible" only late in the curriculum (like v2's dead ends, which are blocked at every level).
TIGHT_FAMILY = ("narrow_turn_tight", "turn_room", 1, 1.0, 0.50, 0.40)
RETENTION_EXTRA = {"platform_gap": 4.0, "pyramid_stairs": 1.0, "pyramid_stairs_inv": 1.0, "random_rough": 1.0}
NUM_COLS = 58
TURN_STAND_WEIGHT, TURN_DRIFT_WEIGHT, TURN_OVERSPIN_WEIGHT, TURN_STALL_WEIGHT = 0.5, -2.0, -1.0, -1.0
NOSE_LIMIT_DEG, NOSE_PEAK_WEIGHT = 5.0, -6.0


class GapNosePeak(ManagerTermBase):
    """Per crossing: the peak nose-up (rad above NOSE_LIMIT_DEG) seen on the gap tile since the previous credited
    crossing, paid once when platform_gap_crossing credits a crossing (its last_event; that term runs earlier)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.peak = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.peak[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, limit_deg: float = NOSE_LIMIT_DEG):
        g = env.scene["robot"].data.projected_gravity_b
        nose = torch.asin((-g[:, 0]).clamp(-1.0, 1.0))
        gap = boarding_family_mask(env)
        self.peak = torch.where(gap, torch.maximum(self.peak, nose), self.peak)
        event = env.reward_manager.get_term_cfg("platform_gap_crossing").func.last_event.any(dim=1) & gap
        excess = (self.peak - math.radians(limit_deg)).clamp_min(0.0)
        out = torch.where(event, excess, torch.zeros_like(excess)) / env.step_dt
        self.peak = torch.where(event, torch.zeros_like(self.peak), self.peak)
        return out


class NarrowV24VelocityCommand(mdp_turn.NarrowV24VelocityCommand, NarrowV23VelocityCommand):
    def __init__(self, cfg, env):
        NarrowV23VelocityCommand.__init__(self, cfg, env)
        self._init_turn()

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if len(ids):  # before super(): its resample already applies the turn command (timer, rate, pauses)
            self._reset_turn(ids)
        return NarrowV23VelocityCommand.reset(self, env_ids)

    def _apply_narrow(self):
        NarrowV23VelocityCommand._apply_narrow(self)
        self._apply_turn()


@configclass
class GastNarrowV24TeacherCfg(GastNarrowV23TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        for name, columns in RETENTION_EXTRA.items():
            gen.sub_terrains[name].proportion += columns
        size = next(iter(gen.sub_terrains.values())).size
        for name, (shape, sign, columns) in TURN_FAMILIES.items():
            gen.sub_terrains[name] = NarrowPassageTerrainCfg(proportion=columns, size=size, kind=shape, sign=sign,
                                                             w_max=TURN_R_MAX, w_min=TURN_R_MIN)
        name, shape, sign, columns, r_max, r_min = TIGHT_FAMILY
        gen.sub_terrains[name] = NarrowPassageTerrainCfg(proportion=columns, size=size, kind=shape, sign=sign,
                                                         w_max=r_max, w_min=r_min)
        gen.num_cols = int(sum(sub.proportion for sub in gen.sub_terrains.values()))
        assert gen.num_cols == NUM_COLS, gen.num_cols

        self.observations.narrow_lidar.lidar_map = ObservationTermCfg(func=NarrowLidarMapV24)
        self.observations.narrow_priv.state = ObservationTermCfg(func=mdp_turn.narrow_privileged_v24)
        self.commands.base_velocity.class_type = NarrowV24VelocityCommand
        self.events.reset_turn = EventTermCfg(func=mdp_turn.reset_turn, mode="reset")  # after reset_narrow_v2

        assert self.rewards.narrow_monitor.func is mdp_v2.NarrowV2Monitor
        self.rewards.narrow_monitor.func = mdp_turn.NarrowV24Monitor
        assert self.rewards.track_ang_vel_z_exp.func is mdp_v2.track_ang_vel_z_gated
        self.rewards.track_ang_vel_z_exp.func = mdp_turn.track_ang_vel_z_turn_gated
        assert self.rewards.narrow_centering.func is mdp.narrow_centering
        self.rewards.narrow_centering.func = mdp_turn.narrow_centering_no_turn
        self.rewards.turn_stand = RewardTermCfg(func=mdp_turn.turn_stand, weight=TURN_STAND_WEIGHT)
        self.rewards.turn_drift = RewardTermCfg(func=mdp_turn.turn_drift, weight=TURN_DRIFT_WEIGHT)
        self.rewards.turn_overspin = RewardTermCfg(func=mdp_turn.turn_overspin, weight=TURN_OVERSPIN_WEIGHT)
        self.rewards.turn_stall = RewardTermCfg(func=mdp_turn.TurnStall, weight=TURN_STALL_WEIGHT)
        self.terminations.turn_drift = TerminationTermCfg(func=mdp_turn.turn_drift_done)
        # Standing where the command still asks to move (turn impossible, blocked zone): the cadence bound off,
        # the lifted-feet count on (see mdp_turn.should_stand).
        assert self.rewards.feet_cadence_overrun.func is gd_rew.feet_cadence_overrun
        self.rewards.feet_cadence_overrun.func = mdp_turn.feet_cadence_overrun_stand_gated
        assert self.rewards.stand_still_feet_contact.func is gd_rew.stand_still_feet_contact
        self.rewards.stand_still_feet_contact.func = mdp_turn.stand_still_feet_contact_stand
        self.rewards.gap_nose_up.params["limit_deg"] = NOSE_LIMIT_DEG
        self.rewards.gap_nose_peak = RewardTermCfg(func=GapNosePeak, weight=NOSE_PEAK_WEIGHT,
                                                   params={"limit_deg": NOSE_LIMIT_DEG})
        self.curriculum.terrain_levels.func = mdp_turn.narrow_levels_v24


@configclass
class GastNarrowV24RunnerCfg(GastNarrowV23RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/narrow_v24"


TASK = "GastNarrowV24"
TASK_ID = "Gd-GastNarrowV24-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV23-Rbq10-Dreamwaq-v0"
