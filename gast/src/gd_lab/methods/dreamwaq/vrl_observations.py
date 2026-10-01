"""VRL retains policy/critic layouts and adds a separately stored terrain group."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from gd_lab.mdp.camera_observations import CameraVisibleTerrain
from gd_lab.methods.dreamwaq.observations import DreamwaqObservationsCfg


@configclass
class VrlObservationsCfg(DreamwaqObservationsCfg):
    @configclass
    class VisibleTerrainCfg(ObservationGroupCfg):
        camera_visible = ObservationTermCfg(func=CameraVisibleTerrain)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain: VisibleTerrainCfg = VisibleTerrainCfg()
