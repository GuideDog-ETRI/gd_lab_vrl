"""Blind-start VRL tasks, built on the unchanged gd_lab VRL teacher task.

2-A  camera-free: fully visible height scan (fast, no rendering).
2-B  camera-visible: the regular teacher observation.
Both add terrain blackouts; ``BLACKOUT`` holds the defaults (100 Hz policy).
"""

from gd_lab.tasks.blind_rough import BlindRoughSceneCfg
from gd_lab.tasks.vrl_teacher import VrlTeacherEnvCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

from .terrain_dropout import CameraVisibleTerrainDropout, FullVisibleTerrainDropout

# ~0.5-3 s segments starting about once per 10 s, plus 5% fully blind episodes:
# roughly 15-20% of steps without terrain.
BLACKOUT = {"start_prob": 0.001, "duration_steps": (50, 300), "episode_prob": 0.05}


@configclass
class BlindStartEnvCfg(VrlTeacherEnvCfg):
    scene: BlindRoughSceneCfg = BlindRoughSceneCfg(num_envs=256, env_spacing=2.5)
    uses_cameras: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(
            func=FullVisibleTerrainDropout, params=dict(BLACKOUT))


@configclass
class BlindStartCameraEnvCfg(VrlTeacherEnvCfg):
    uses_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(
            func=CameraVisibleTerrainDropout, params=dict(BLACKOUT))
