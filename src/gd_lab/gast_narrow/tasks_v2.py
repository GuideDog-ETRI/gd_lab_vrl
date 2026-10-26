"""GAST-Narrow v2: v1 1748 (6566 + head LiDAR), clearance-margin input, blocked passages -> stand (design: GAST_NARROW_V2_DESIGN.md).

Built on the v1 config (6566 retention tiles unchanged); only the narrow parts change:
- terrain (31 columns): retention 13 + corridor 4, door +/- 3+3, bend left/right 3+3, dead end 2.
  Width 1.20 m (level 0) -> 0.50 m (level 14); dead ends 1.20 -> 0.80 m with the end wall moving toward the mouth.
- inputs: LiDAR map as v1 ([2, 30, 48], 5 cm; warm start from v1 1748), margin m (1); critic privileged 9.
- command/reset/rewards/termination/curriculum from ``mdp_v2``.
"""
from isaaclab.managers import (EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg,
                               TerminationTermCfg)
from isaaclab.utils import configclass

import gd_lab.mdp.rewards as gd_rew
from gd_lab.gast_narrow import mdp_v2
from gd_lab.gast_narrow.lidar import NarrowLidarMapV2
from gd_lab.gast_narrow.tasks import RETENTION_COLUMNS, GastNarrowV1TeacherCfg
from gd_lab.gast_narrow.terrain import NarrowPassageTerrainCfg

NARROW_FAMILIES = {
    # name: (kind, sign, columns, w_max, w_min)
    "narrow_corridor": ("corridor", 1, 4.0, 1.20, 0.50),
    "narrow_door_pos": ("door", 1, 3.0, 1.20, 0.50),
    "narrow_door_neg": ("door", -1, 3.0, 1.20, 0.50),
    "narrow_bend_left": ("bend", 1, 3.0, 1.20, 0.50),
    "narrow_bend_right": ("bend", -1, 3.0, 1.20, 0.50),
    "narrow_dead_end": ("dead_end", 1, 2.0, 1.20, 0.80),
}


@configclass
class NarrowLidarV2Group(ObservationGroupCfg):
    lidar_map = ObservationTermCfg(func=NarrowLidarMapV2)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class NarrowMarginGroup(ObservationGroupCfg):
    margin = ObservationTermCfg(func=mdp_v2.narrow_margin_obs)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class NarrowPrivilegedV2Group(ObservationGroupCfg):
    state = ObservationTermCfg(func=mdp_v2.narrow_privileged_v2)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GastNarrowV2TeacherCfg(GastNarrowV1TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        for name in [n for n in gen.sub_terrains if n.startswith("narrow_")]:
            del gen.sub_terrains[name]
        for name, sub in gen.sub_terrains.items():
            sub.proportion = RETENTION_COLUMNS.get(name, 1.0)
        size = next(iter(gen.sub_terrains.values())).size
        for name, (kind, sign, columns, w_max, w_min) in NARROW_FAMILIES.items():
            gen.sub_terrains[name] = NarrowPassageTerrainCfg(proportion=columns, size=size, kind=kind, sign=sign,
                                                             w_max=w_max, w_min=w_min)
        gen.num_cols = int(sum(sub.proportion for sub in gen.sub_terrains.values()))
        assert gen.num_cols == 31, gen.num_cols

        self.observations.narrow_lidar = NarrowLidarV2Group()
        self.observations.narrow_margin = NarrowMarginGroup()
        self.observations.narrow_priv = NarrowPrivilegedV2Group()

        self.commands.base_velocity.class_type = mdp_v2.NarrowV2VelocityCommand
        del self.events.reset_narrow_pose
        self.events.reset_narrow_v2 = EventTermCfg(func=mdp_v2.reset_narrow_v2, mode="reset")

        for name in ("narrow_monitor", "narrow_success", "narrow_wall_proximity"):
            delattr(self.rewards, name)
        centering = self.rewards.narrow_centering
        delattr(self.rewards, "narrow_centering")
        # Monitor first: bonuses / terminations / curriculum / Top-5 read its per-episode tracker.
        self.rewards.narrow_monitor = RewardTermCfg(func=mdp_v2.NarrowV2Monitor, weight=1.0)
        self.rewards.narrow_success = RewardTermCfg(func=mdp_v2.narrow_pass_bonus, weight=8.0)
        self.rewards.narrow_hold = RewardTermCfg(func=mdp_v2.narrow_hold_bonus, weight=4.0)
        self.rewards.narrow_breach = RewardTermCfg(func=mdp_v2.narrow_breach_penalty, weight=-4.0)
        self.rewards.narrow_stand = RewardTermCfg(func=mdp_v2.narrow_stand_in_zone, weight=0.5)
        self.rewards.narrow_wall_proximity = RewardTermCfg(func=mdp_v2.narrow_wall_proximity_v2, weight=-1.0)
        self.rewards.narrow_centering = centering
        # Tracking off in the blocked zone (the command keeps pushing; standing must not cost tracking reward).
        assert self.rewards.track_lin_vel_xy_exp.func is gd_rew.track_lin_vel_xy_velocity_scaled_exp
        assert self.rewards.track_ang_vel_z_exp.func is gd_rew.track_ang_vel_z_velocity_scaled_exp
        self.rewards.track_lin_vel_xy_exp.func = mdp_v2.track_lin_vel_xy_gated
        self.rewards.track_ang_vel_z_exp.func = mdp_v2.track_ang_vel_z_gated

        self.terminations.narrow_success_done = TerminationTermCfg(func=mdp_v2.narrow_success_done_v2, time_out=True)
        self.terminations.narrow_breach = TerminationTermCfg(func=mdp_v2.narrow_breach_done)
        self.curriculum.terrain_levels.func = mdp_v2.narrow_levels_v2


TASK = "GastNarrowV2"
TASK_ID = "Gd-GastNarrowV2-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0"
