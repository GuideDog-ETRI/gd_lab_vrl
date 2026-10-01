"""Raycast teacher targets with rendered camera snapshots for student distillation."""
from isaaclab.utils import configclass
from gd_lab.core import registry
from gd_lab.mdp.camera_observations import CameraVisibleTerrain
from .tasks import BlindStartRaycastEnvCfg
from .raycast_terrain import RaycastVisibleTerrainDropout


class RayTeacherWithCameraSnapshots(RaycastVisibleTerrainDropout):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.camera_snapshots = CameraVisibleTerrain(cfg, env)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'camera_snapshots'):
            self.camera_snapshots.reset(env_ids)

    def __call__(self, env, start_prob=0.0, duration_steps=(50,300), episode_prob=0.0):
        self.camera_snapshots(env)  # Snapshot side effect only; discard rendered terrain targets.
        return super().__call__(env, start_prob, duration_steps, episode_prob)


@configclass
class RayStudentEnvCfg(BlindStartRaycastEnvCfg):
    uses_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible.func = RayTeacherWithCameraSnapshots


TASK = registry.register_task(
    task='VrlRayStudent', robot='Rbq10', method='Dreamwaq', mode='Vision',
    env_cfg='gd_lab.teachers.bivt.student_task:RayStudentEnvCfg',
    agent_cfg='gd_lab.teachers.bivt.agent_cfg:BlindStartRunnerCfg')
