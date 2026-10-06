"""Blind-start VRL tasks, built on the unchanged gd_lab VRL teacher task.

2-A  camera-free: fully visible height scan (fast, no rendering).
2-B  camera-visible: the regular teacher observation.
2-R  camera-free, raycast camera visibility (no rendering; mask close to 2-B).
Both add terrain blackouts; ``BLACKOUT`` holds the defaults (100 Hz policy).
GAP  opt-in gap fine-tuning arms on top of 2-R (baseline / intrusion / clean).
"""

from isaaclab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from gd_lab.mdp.platform_gap_finetune import (
    GapMonitor,
    gap_clean_bonus,
    gap_intrusion_penalty,
    platform_gap_diagnostics,
)
from gd_lab.mdp.gap_stair_v2 import (
    FootholdMargin,
    SlotProbe,
    StairHandleDisturbance,
    gap_stair_v2_diagnostics,
    stair_push_fall,
    stair_push_slip,
)
from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator
from gd_lab.tasks.blind_rough import BlindRoughSceneCfg
from gd_lab.teachers.cvtt.tasks import VrlTeacherEnvCfg

from .raycast_terrain import RaycastVisibleTerrainDropout
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


@configclass
class BlindStartRaycastEnvCfg(BlindStartEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.terrain.camera_visible = ObservationTermCfg(
            func=RaycastVisibleTerrainDropout, params=dict(BLACKOUT))


@configclass
class GapFinetuneRaycastEnvCfg(BlindStartRaycastEnvCfg):
    """Opt-in gap fine-tuning base; the arms below differ ONLY in the two weights.

    The monitor (weight 1.0, always returns zeros) runs in every arm so that the
    diagnostics are comparable. A zero-weight penalty/bonus is skipped by the
    RewardManager, which is fine because the monitor has already computed them.
    """

    intrusion_weight: float = 0.0  # per second, applied to a cost in [0, 1]
    clean_weight: float = 0.0  # per clean crossing event

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.class_type = GapMetadataTerrainGenerator
        # Start at the final command range (the 17206 run recovered it within ~500 updates anyway);
        # the gap command ceiling (0.6 m/s) is untouched.
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (1.0, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (1.0, 1.0)
        # Order matters: after platform_gap_crossing (appended by the base task), monitor first.
        self.rewards.platform_gap_monitor = RewardTermCfg(
            func=GapMonitor, weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.platform_gap_intrusion = RewardTermCfg(
            func=gap_intrusion_penalty, weight=self.intrusion_weight, params={})
        self.rewards.platform_gap_clean = RewardTermCfg(func=gap_clean_bonus, weight=self.clean_weight, params={})
        self.curriculum.platform_gap_diagnostics = CurriculumTermCfg(func=platform_gap_diagnostics, params={})


@configclass
class GapFinetuneBaselineRaycastEnvCfg(GapFinetuneRaycastEnvCfg):
    intrusion_weight: float = 0.0
    clean_weight: float = 0.0


@configclass
class GapFinetuneIntrusionRaycastEnvCfg(GapFinetuneRaycastEnvCfg):
    intrusion_weight: float = -3.0
    clean_weight: float = 0.0


@configclass
class GapFinetuneCleanRaycastEnvCfg(GapFinetuneRaycastEnvCfg):
    intrusion_weight: float = -3.0
    clean_weight: float = 1.5


@configclass
class GapFinetuneCleanV2RaycastEnvCfg(GapFinetuneCleanRaycastEnvCfg):
    """Clean arm + v2 terms (2026-10-06 review, claude_handoff/TEACHER_REWARD_V2_REVIEW_AND_DESIGN_20261006.md).

    Gap: drop-off foothold margin at touchdown, slot probing/edge contact below the higher deck, strict
    clean (no contact over the slot), widths up to 26 cm. Stairs: nose-side foothold margin and the hip-handle
    disturbance (pull back-and-down while ascending, push while descending) with front-lift / fall / slip costs.
    Observations are unchanged, so a Clean-arm checkpoint resumes as is.
    """

    gap_margin_weight: float = -0.5  # per fully violating touchdown
    stair_margin_weight: float = -0.2
    slot_probe_weight: float = -1.0  # per second, mean over feet
    front_lift_weight: float = -1.0  # per second inside the disturbance window
    push_fall_weight: float = -5.0  # per disturbance window, on top of termination_penalty
    push_slip_weight: float = -0.2

    def __post_init__(self):
        super().__post_init__()
        feet = SceneEntityCfg("robot", body_names=".*_foot")
        contacts = SceneEntityCfg("contact_forces", body_names=".*_foot")
        self.scene.terrain.terrain_generator.sub_terrains["platform_gap"].gap_width_range = (0.02, 0.26)
        self.rewards.platform_gap_monitor.params["strict_contact"] = True
        self.rewards.gap_foothold_margin = RewardTermCfg(
            func=FootholdMargin, weight=self.gap_margin_weight,
            params={"mode": "gap", "asset_cfg": feet, "sensor_cfg": contacts, "margin": 0.04, "min_command": 0.2})
        self.rewards.stair_foothold_margin = RewardTermCfg(
            func=FootholdMargin, weight=self.stair_margin_weight,
            params={"mode": "stair", "asset_cfg": feet, "sensor_cfg": contacts, "margin": 0.04, "min_command": 0.2})
        self.rewards.gap_slot_probe = RewardTermCfg(
            func=SlotProbe, weight=self.slot_probe_weight, params={"asset_cfg": feet, "sensor_cfg": contacts})
        # Order matters: the disturbance term first; fall/slip read its state in the same step.
        self.rewards.stair_handle_disturbance = RewardTermCfg(
            func=StairHandleDisturbance, weight=self.front_lift_weight,
            params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": contacts})
        self.rewards.stair_push_fall = RewardTermCfg(func=stair_push_fall, weight=self.push_fall_weight, params={})
        self.rewards.stair_push_slip = RewardTermCfg(
            func=stair_push_slip, weight=self.push_slip_weight, params={"asset_cfg": feet, "sensor_cfg": contacts})
        self.curriculum.gap_stair_v2_diagnostics = CurriculumTermCfg(func=gap_stair_v2_diagnostics, params={})
