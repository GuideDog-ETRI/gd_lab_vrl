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
from gd_lab.mdp.gap_stair_v2 import add_v2_terms
from gd_lab.mdp.gap_stair_v21 import add_v21_terms
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
    Terms, weights and force ranges come from ``gd_lab.mdp.gap_stair_v2.add_v2_terms``, shared with the GAST
    CleanV2 teacher. Observations are unchanged, so a Clean-arm checkpoint resumes as is.
    """

    def __post_init__(self):
        super().__post_init__()
        add_v2_terms(self)


@configclass
class GapFinetuneCleanV21RaycastEnvCfg(GapFinetuneCleanRaycastEnvCfg):
    """Main 30k-update training (v2.1): v2 + gap/stair speeds up to 1.2 m/s (usual 0.8-1.0), straight stair runs,
    gap hind-hop / overspeed / stair-stall costs, halved gap margin (``gd_lab.mdp.gap_stair_v21.add_v21_terms``,
    shared with the GAST v2.1 teachers). Observations unchanged: a Clean or v2 checkpoint resumes as is."""

    def __post_init__(self):
        super().__post_init__()
        add_v21_terms(self)
