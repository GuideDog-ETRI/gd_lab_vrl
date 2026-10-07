"""Frozen GAST teacher -> GAST camera student.

The student env is the BIVT-Ray student env (cameras, camera-visible Ray labels for the student's spatial
heads) plus the GAST teacher's own input: ``gast_history`` built from the clean full grid. The teacher
checkpoint loads through GastRunnerCfg; GastTeacherView feeds it that input instead of the camera-visible
``terrain`` group, so its latent and actions are the ones it learned with full sight.
"""
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass
from gd_lab.core import registry
from gd_lab.gast.bivt_student_task import BivtGastStudentCfg
from gd_lab.gast.observations import CleanTerrainHistory
from gd_lab.gast.tasks import CleanTargets


@configclass
class CleanHistoryObservations(CleanTargets):
    target = ObservationTermCfg(func=CleanTerrainHistory)


@configclass
class GastTeacherGastStudentCfg(BivtGastStudentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.gast_history = CleanHistoryObservations()


class GastTeacherView:
    """A frozen GastActorCritic whose terrain inputs come from the clean ``gast_history`` group.

    ``terrain`` (its no-scan gate) becomes the current clean frame of that history; the student-side
    ``terrain`` group (camera-visible Ray scan) is left untouched for the student labels.
    """

    def __init__(self, teacher):
        self.teacher = teacher

    def _teacher_obs(self, obs):
        view = obs.copy()
        view["terrain"] = obs["gast_history"].reshape(obs.batch_size[0], 8, 375)[:, -1, :374]
        return view

    def terrain_latent(self, obs):
        return self.teacher.terrain_latent(self._teacher_obs(obs))

    def _actor_input(self, obs, *, inference):
        return self.teacher._actor_input(self._teacher_obs(obs), inference=inference)

    def act_inference(self, obs):
        return self.teacher.act_with_terrain_latent(obs, self.terrain_latent(obs))

    def __getattr__(self, name):
        return getattr(self.teacher, name)


registry.register_task(task='GastTeacherGastStudent', robot='Rbq10', method='Dreamwaq', mode='Vision',
    env_cfg='gd_lab.gast.gast_teacher_student_task:GastTeacherGastStudentCfg',
    agent_cfg='gd_lab.gast.teacher:GastRunnerCfg')
