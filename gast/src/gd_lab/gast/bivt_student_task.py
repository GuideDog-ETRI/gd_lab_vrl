"""Frozen BIVT-Ray teacher with GAST visual student auxiliary labels."""
from isaaclab.utils import configclass
from gd_lab.core import registry
from gd_lab.teachers.bivt.student_task import RayStudentEnvCfg
from gd_lab.gast.tasks import CleanTargets


@configclass
class BivtGastStudentCfg(RayStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.gast_clean = CleanTargets()
        # Teacher blackout zeroes the Ray labels while the student's camera frames are
        # fine (~15-20% of steps): visibility 0, zero latent, blind action. Simulated
        # missing frames already train the blind route.
        self.observations.terrain.camera_visible.params.update(start_prob=0.0, episode_prob=0.0)


registry.register_task(task='BivtGastStudent', robot='Rbq10', method='Dreamwaq',
    mode='Vision', env_cfg='gd_lab.gast.bivt_student_task:BivtGastStudentCfg',
    agent_cfg='gd_lab.teachers.bivt.agent_cfg:BlindStartRunnerCfg')
