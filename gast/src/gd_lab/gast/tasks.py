import torch
from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg, TerminationTermCfg
from isaaclab.utils import configclass
from gd_lab.core import registry
from gd_lab.tasks.blind_rough import BlindRoughSceneCfg
from gd_lab.teachers.cvtt.tasks import VrlTeacherEnvCfg, VrlSceneCfg
from gd_lab.gast.observations import NoisyTerrain, StudentTerrain, clean_terrain, history_observation


@configclass
class CleanTargets(ObservationGroupCfg):
    target = ObservationTermCfg(func=clean_terrain)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HistoryObservations(CleanTargets):
    target = ObservationTermCfg(func=history_observation)


def nonfinite_state(env):
    """Reset envs whose physics went NaN/inf before their observations are computed.

    Without it a single exploded env stays NaN until time-out, and its observations
    poison the shared normalizers (arm4 crashes at iter 8727 and 10342).
    """
    data = env.scene['robot'].data
    return ~(torch.isfinite(data.root_state_w).all(-1) & torch.isfinite(data.joint_pos).all(-1)
             & torch.isfinite(data.joint_vel).all(-1))


@configclass
class GastTeacherCfg(VrlTeacherEnvCfg):
    scene: BlindRoughSceneCfg = BlindRoughSceneCfg(num_envs=256, env_spacing=2.5)
    uses_cameras: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(func=NoisyTerrain)
        self.observations.gast_clean = CleanTargets()
        self.observations.gast_history = HistoryObservations()
        self.terminations.nonfinite_state = TerminationTermCfg(func=nonfinite_state)


@configclass
class GastStudentCfg(GastTeacherCfg):
    scene: VrlSceneCfg = VrlSceneCfg(num_envs=16, env_spacing=2.5)
    uses_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(func=StudentTerrain)


@configclass
class GastGapCleanTeacherCfg(GastTeacherCfg):
    """GAST teacher with the Clean gap rewards of the BIVT-Ray gap fine-tuning (same terms, same weights).

    Used to continue a BIVT-Ray Clean teacher as a GAST teacher (gd_lab.gast.warm_start): intrusion of any
    foot into a gap slot is penalised (-3, per second, cost in [0, 1]) and a crossing with no foot deeper
    than the clean threshold earns +1.5, on top of the inherited foot_drop (-2) / crossing (+0.5).
    """

    intrusion_weight: float = -3.0
    clean_weight: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg
        from gd_lab.mdp.platform_gap_finetune import (
            GapMonitor, gap_clean_bonus, gap_intrusion_penalty, platform_gap_diagnostics)
        from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator

        self.scene.terrain.terrain_generator.class_type = GapMetadataTerrainGenerator
        # The warm-started teacher already walks the full command range (as in the BIVT-Ray gap run).
        for name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            term = getattr(self.curriculum, name, None)
            if term is not None:
                term.params["range_multiplier"] = (1.0, 1.0)
        # Order matters: after platform_gap_crossing (appended by the base task), monitor first.
        self.rewards.platform_gap_monitor = RewardTermCfg(
            func=GapMonitor, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
        self.rewards.platform_gap_intrusion = RewardTermCfg(
            func=gap_intrusion_penalty, weight=self.intrusion_weight, params={})
        self.rewards.platform_gap_clean = RewardTermCfg(func=gap_clean_bonus, weight=self.clean_weight, params={})
        self.curriculum.platform_gap_diagnostics = CurriculumTermCfg(func=platform_gap_diagnostics, params={})


@configclass
class GastGapCleanV2TeacherCfg(GastGapCleanTeacherCfg):
    """GAST teacher with the v2 gap/stair terms of the BIVT-Ray CleanV2 arm (same function, same weights,
    same disturbance), so the two teacher families are compared on one objective."""

    def __post_init__(self):
        super().__post_init__()
        from gd_lab.mdp.gap_stair_v2 import add_v2_terms

        add_v2_terms(self)


@configclass
class GastScratchV2TeacherCfg(GastTeacherCfg):
    """Experiment 3: GAST teacher trained from scratch on the full v2 objective -- the Clean gap terms of
    GastGapCleanTeacherCfg (same weights) plus ``add_v2_terms`` -- but WITHOUT the warm-start shortcut that
    starts the command curriculum at full range. Use a long force ramp (GAST_V2_FORCE_RAMP_STEPS, e.g. 1e6)."""

    intrusion_weight: float = -3.0
    clean_weight: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg
        from gd_lab.mdp.gap_stair_v2 import add_v2_terms
        from gd_lab.mdp.platform_gap_finetune import (
            GapMonitor, gap_clean_bonus, gap_intrusion_penalty, platform_gap_diagnostics)
        from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator

        self.scene.terrain.terrain_generator.class_type = GapMetadataTerrainGenerator
        self.rewards.platform_gap_monitor = RewardTermCfg(
            func=GapMonitor, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
        self.rewards.platform_gap_intrusion = RewardTermCfg(
            func=gap_intrusion_penalty, weight=self.intrusion_weight, params={})
        self.rewards.platform_gap_clean = RewardTermCfg(func=gap_clean_bonus, weight=self.clean_weight, params={})
        self.curriculum.platform_gap_diagnostics = CurriculumTermCfg(func=platform_gap_diagnostics, params={})
        add_v2_terms(self)


registry.register_task(task='Gast', robot='Rbq10', method='Dreamwaq',
    env_cfg='gd_lab.gast.tasks:GastTeacherCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
registry.register_task(task='GastGapClean', robot='Rbq10', method='Dreamwaq',
    env_cfg='gd_lab.gast.tasks:GastGapCleanTeacherCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
registry.register_task(task='GastGapCleanV2', robot='Rbq10', method='Dreamwaq',
    env_cfg='gd_lab.gast.tasks:GastGapCleanV2TeacherCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
registry.register_task(task='GastScratchV2', robot='Rbq10', method='Dreamwaq',
    env_cfg='gd_lab.gast.tasks:GastScratchV2TeacherCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
registry.register_task(task='Gast', robot='Rbq10', method='Dreamwaq', mode='Vision',
    env_cfg='gd_lab.gast.tasks:GastStudentCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
