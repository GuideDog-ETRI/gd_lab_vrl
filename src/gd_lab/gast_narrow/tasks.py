"""GAST-Narrow v1: narrow-passage fine-tune of backgap V2 update 6566 (design: docs/design/GAST_NARROW_V1_DESIGN.md).

Only new classes. The 6566 task (Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0) is the parent and stays unchanged.

Terrain (31 columns x 15 levels, 8 m tiles):
- retention (13): the 11 families of 6566, platform_gap 3 columns, the others 1 each -- same terms as 6566.
- narrow (18): corridor 4, door +/- 3+3, bend left/right 4+4. Width 1.20 m (level 0) -> 0.60 m (level 14).
New inputs: head LiDAR map (actor, group narrow_lidar), wall/path privileged state (critic, group narrow_priv).
New terms on narrow tiles only: gdm-like command, spawn before the passage, wall proximity -1/s per point,
centering +0.3/s, passage bonus +8, end 0.5 s after the exit (time-out type), width curriculum.
"""
from isaaclab.managers import (EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg,
                               TerminationTermCfg)
from isaaclab.utils import configclass

from gd_lab.gast_backgap.tasks import GastRayIrV21Gap50BackV2TeacherCfg
from gd_lab.gast_narrow import mdp
from gd_lab.gast_narrow.lidar import NarrowLidarMap
from gd_lab.gast_narrow.terrain import NarrowPassageTerrainCfg, NarrowTerrainGenerator

RETENTION_COLUMNS = {"platform_gap": 3.0}
NARROW_FAMILIES = {
    "narrow_corridor": ("corridor", 1, 4.0),
    "narrow_door_pos": ("door", 1, 3.0),
    "narrow_door_neg": ("door", -1, 3.0),
    "narrow_bend_left": ("bend", 1, 4.0),
    "narrow_bend_right": ("bend", -1, 4.0),
}


@configclass
class NarrowLidarGroup(ObservationGroupCfg):
    lidar_map = ObservationTermCfg(func=NarrowLidarMap)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class NarrowPrivilegedGroup(ObservationGroupCfg):
    state = ObservationTermCfg(func=mdp.narrow_privileged)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GastNarrowV1TeacherCfg(GastRayIrV21Gap50BackV2TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.class_type = NarrowTerrainGenerator
        for name, sub in gen.sub_terrains.items():
            sub.proportion = RETENTION_COLUMNS.get(name, 1.0)
        size = next(iter(gen.sub_terrains.values())).size
        for name, (kind, sign, columns) in NARROW_FAMILIES.items():
            gen.sub_terrains[name] = NarrowPassageTerrainCfg(proportion=columns, size=size, kind=kind, sign=sign)
        gen.num_cols = int(sum(sub.proportion for sub in gen.sub_terrains.values()))
        assert gen.num_cols == 31, gen.num_cols

        self.observations.narrow_lidar = NarrowLidarGroup()
        self.observations.narrow_priv = NarrowPrivilegedGroup()

        self.commands.base_velocity.class_type = mdp.NarrowVelocityCommand
        self.events.reset_narrow_pose = EventTermCfg(func=mdp.reset_narrow_pose, mode="reset")

        # Monitor first: the bonus / termination / curriculum / Top-5 read its per-episode tracker.
        self.rewards.narrow_monitor = RewardTermCfg(func=mdp.NarrowMonitor, weight=1.0)
        self.rewards.narrow_success = RewardTermCfg(func=mdp.narrow_success_bonus, weight=8.0)
        self.rewards.narrow_wall_proximity = RewardTermCfg(func=mdp.narrow_wall_proximity, weight=-1.0)
        self.rewards.narrow_centering = RewardTermCfg(func=mdp.narrow_centering, weight=0.3)

        self.terminations.narrow_success_done = TerminationTermCfg(func=mdp.narrow_success_done, time_out=True)
        self.curriculum.terrain_levels.func = mdp.narrow_levels


TASK = "GastNarrowV1"
TASK_ID = "Gd-GastNarrowV1-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0"
