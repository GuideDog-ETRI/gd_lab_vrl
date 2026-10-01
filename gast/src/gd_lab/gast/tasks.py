from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg
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


@configclass
class GastTeacherCfg(VrlTeacherEnvCfg):
    scene: BlindRoughSceneCfg = BlindRoughSceneCfg(num_envs=256, env_spacing=2.5)
    uses_cameras: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(func=NoisyTerrain)
        self.observations.gast_clean = CleanTargets()
        self.observations.gast_history = HistoryObservations()


@configclass
class GastStudentCfg(GastTeacherCfg):
    scene: VrlSceneCfg = VrlSceneCfg(num_envs=16, env_spacing=2.5)
    uses_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(func=StudentTerrain)


registry.register_task(task='Gast', robot='Rbq10', method='Dreamwaq',
    env_cfg='gd_lab.gast.tasks:GastTeacherCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
registry.register_task(task='Gast', robot='Rbq10', method='Dreamwaq', mode='Vision',
    env_cfg='gd_lab.gast.tasks:GastStudentCfg', agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
