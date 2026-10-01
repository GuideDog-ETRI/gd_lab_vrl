"""Original frozen CVTT observations with clean GAST auxiliary targets only."""
from isaaclab.utils import configclass
from gd_lab.core import registry
from gd_lab.teachers.cvtt.student_env import VisionRoughEnvCfg
from gd_lab.gast.tasks import CleanTargets


@configclass
class CvttGastStudentCfg(VisionRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Preserve CVTT camera-visible terrain and actor/CENet observation groups.
        # Clean geometry is auxiliary supervision, never an actor/student input.
        self.observations.gast_clean = CleanTargets()


registry.register_task(task='CvttGastStudent', robot='Rbq10', method='Dreamwaq',
    mode='Vision', env_cfg='gd_lab.gast.cvtt_student_task:CvttGastStudentCfg',
    agent_cfg='gd_lab.teachers.cvtt.agent_cfg:DreamwaqVrlRunnerCfg')
