"""Shared teacher/student VRL recipe: original terrain families + platform_gap only."""

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from gd_lab.core.camera_contract import CAMERA_NAMES, DEFAULT_CAMERA_PROFILE
from gd_lab.mdp.platform_gap_commands import BoardingVelocityCommand, reset_boarding_root
from gd_lab.mdp.platform_gap_terms import (
    PlatformGapCrossing,
    platform_gap_base_height,
    platform_gap_foot_drop,
    platform_gap_levels,
    platform_gap_pitch,
)
from gd_lab.mdp.terrains.platform_gap import PlatformGapTerrainCfg
from gd_lab.methods.dreamwaq.vrl_observations import VrlObservationsCfg
from gd_lab.tasks.blind_rough import BlindRoughEnvCfg, BlindRoughSceneCfg, _apply_play
from gd_lab.tasks.vrl_cameras import default_vrl_camera


@configclass
class VrlSceneCfg(BlindRoughSceneCfg):
    """Four rendered cameras are shared by teacher visibility and student input."""
    front_depth_camera0 = default_vrl_camera(CAMERA_NAMES[0])
    front_depth_camera1 = default_vrl_camera(CAMERA_NAMES[1])
    hind_depth_camera2 = default_vrl_camera(CAMERA_NAMES[2])
    hind_depth_camera3 = default_vrl_camera(CAMERA_NAMES[3])


@configclass
class VrlTeacherEnvCfg(BlindRoughEnvCfg):
    scene: VrlSceneCfg = VrlSceneCfg(num_envs=256, env_spacing=2.5)
    observations: VrlObservationsCfg = VrlObservationsCfg()
    camera_profile: str = DEFAULT_CAMERA_PROFILE
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        # Exact column allocation: 10 existing families x 1, boarding gaps x 5.
        gen.num_cols = 15
        for sub in gen.sub_terrains.values():
            sub.proportion = 1.0
        gen.sub_terrains["platform_gap"] = PlatformGapTerrainCfg(proportion=5.0)
        self.commands.base_velocity.class_type = BoardingVelocityCommand
        self.events.reset_base.func = reset_boarding_root
        self.rewards.platform_gap_foot_drop = RewardTermCfg(
            func=platform_gap_foot_drop, weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        self.rewards.platform_gap_crossing = RewardTermCfg(
            func=PlatformGapCrossing, weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "hold_time": 0.1,
            },
        )
        self.rewards.base_height.func = platform_gap_base_height
        self.rewards.flat_orientation_pitch_l2.func = platform_gap_pitch
        self.rewards.lin_vel_z_l2.params["family_weight_scales"]["platform_gap"] = 0.25
        self.rewards.feet_touchdown.params["family_weight_scales"]["platform_gap"] = 0.5
        excluded = self.rewards.stand_still.params["exclude_terrain_families"]
        self.rewards.stand_still.params["exclude_terrain_families"] = (*excluded, "platform_gap")
        self.curriculum.terrain_levels.func = platform_gap_levels


def apply_vrl_play(cfg):
    _apply_play(cfg)
    # Random terrain generation does not preserve family columns, which gates
    # rewards incorrectly. Keep deterministic family columns, disable only updates.
    gen = cfg.scene.terrain.terrain_generator
    gen.num_cols = 15
    gen.curriculum = True
    gen.difficulty_range = (0.4, 0.4)
    cfg.curriculum.terrain_levels = None


@configclass
class VrlSceneCfg(BlindRoughSceneCfg):
    """Four rendered cameras are shared by teacher visibility and student input."""
    front_depth_camera0 = default_vrl_camera(CAMERA_NAMES[0])
    front_depth_camera1 = default_vrl_camera(CAMERA_NAMES[1])
    hind_depth_camera2 = default_vrl_camera(CAMERA_NAMES[2])
    hind_depth_camera3 = default_vrl_camera(CAMERA_NAMES[3])


@configclass
class VrlTeacherEnvCfg_PLAY(VrlTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_vrl_play(self)
