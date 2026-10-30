"""GAST-Narrow v2.4: v2.3 (Hangang it602) + in-place turns in confined pockets (design: GAST_NARROW_V24_TURN_DESIGN.md).

Changes (network, anchor, other narrow tiles and the retention tiles as v2.3):
- terrain 44 -> 56 columns: turn pockets aisle / vestibule / room 2 each (free radius 1.00 m at level 0 -> 0.45 m at
  level 14), plus platform_gap +4 and pyramid_stairs / pyramid_stairs_inv +1 each to keep the retention half at 50 %.
- LiDAR: head Ouster + head Mid-360 + rear Mid-360 on the tail, trunk self-occlusion, same map [2, 30, 48]
  (lidar.NarrowLidarMapV24).
- turn tiles (mdp_turn): command vx = vy = 0, wz = +/-(0.3..0.8) rad/s for a 45..180 deg target (right 70 %);
  turnable = r_free >= R_SWEEP + m; success -> the v2 pass / hold / breach events; new rewards turn_stand +0.5,
  turn_drift -2, turn_overspin -1; yaw tracking off where the turn is not possible; centering off; drift > 0.2 m ends.
- critic: the same 9 narrow inputs, turn state in slots 2 / 4 / 7 / 8 on turn tiles.
"""
import torch
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_narrow import mdp, mdp_turn, mdp_v2
from gd_lab.gast_narrow.lidar import NarrowLidarMapV24
from gd_lab.gast_narrow.tasks_v23 import GastNarrowV23RunnerCfg, GastNarrowV23TeacherCfg, NarrowV23VelocityCommand
from gd_lab.gast_narrow.terrain import NarrowPassageTerrainCfg

TURN_FAMILIES = {
    # name: (shape, sign, columns)
    "narrow_turn_aisle": ("turn_aisle", 1, 2.0),
    "narrow_turn_vestibule": ("turn_vestibule", 1, 1.0),
    "narrow_turn_vestibule_neg": ("turn_vestibule", -1, 1.0),
    "narrow_turn_room": ("turn_room", 1, 1.0),
    "narrow_turn_room_neg": ("turn_room", -1, 1.0),
}
TURN_R_MAX, TURN_R_MIN = 1.00, 0.45
RETENTION_EXTRA = {"platform_gap": 4.0, "pyramid_stairs": 1.0, "pyramid_stairs_inv": 1.0}
NUM_COLS = 56
TURN_STAND_WEIGHT, TURN_DRIFT_WEIGHT, TURN_OVERSPIN_WEIGHT = 0.5, -2.0, -1.0


class NarrowV24VelocityCommand(mdp_turn.NarrowV24VelocityCommand, NarrowV23VelocityCommand):
    def __init__(self, cfg, env):
        NarrowV23VelocityCommand.__init__(self, cfg, env)
        self._init_turn()

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        out = NarrowV23VelocityCommand.reset(self, env_ids)
        if len(ids):
            self._reset_turn(ids)
        return out

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
        self.terminations.turn_drift = TerminationTermCfg(func=mdp_turn.turn_drift_done)
        self.curriculum.terrain_levels.func = mdp_turn.narrow_levels_v24


@configclass
class GastNarrowV24RunnerCfg(GastNarrowV23RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/narrow_v24"


TASK = "GastNarrowV24"
TASK_ID = "Gd-GastNarrowV24-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV23-Rbq10-Dreamwaq-v0"
