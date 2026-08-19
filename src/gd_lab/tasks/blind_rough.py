"""Blind rough-terrain locomotion task: scene, terrain, commands, terminations,
and the assembled train / play / gamepad environment configs."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
)

from gd_lab.mdp.actions import JointPositionActionWithLimitCfg
from gd_lab.mdp.commands_gamepad import GamepadVelocityCommandCfg
from gd_lab.mdp.terrains import (
    MeshInvertedPyramidStairsNosingTerrainCfg,
    MeshPyramidStairsNosingTerrainCfg,
    MultiRingFenceTerrainCfg,
)
from gd_lab.methods.dreamwaq import (
    DreamwaqCurriculumCfg,
    DreamwaqEventsCfg,
    DreamwaqObservationsCfg,
    DreamwaqRewardsCfg,
)
from gd_lab.robots import RBQ10_CFG


def _foot_scanner(body: str) -> RayCasterCfg:
    """2x2 downward ray-cast around one foot (size == resolution -> 2 points per axis)."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.10, size=(0.10, 0.10)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class BlindRoughSceneCfg(MySceneCfg):
    """Stock locomotion scene plus per-foot terrain scanners for the privileged critic."""

    height_scanner_feet_fl = _foot_scanner("FL_foot")
    height_scanner_feet_fr = _foot_scanner("FR_foot")
    height_scanner_feet_rl = _foot_scanner("RL_foot")
    height_scanner_feet_rr = _foot_scanner("RR_foot")


@configclass
class BlindRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: BlindRoughSceneCfg = BlindRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DreamwaqObservationsCfg = DreamwaqObservationsCfg()
    rewards: DreamwaqRewardsCfg = DreamwaqRewardsCfg()
    events: DreamwaqEventsCfg = DreamwaqEventsCfg()
    curriculum: DreamwaqCurriculumCfg = DreamwaqCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self._scene_init()
        self._action_init()
        self._terrain_init()
        self._command_init()
        self._terminations_init()
        self._validate()
        self._disable_zero_weight_rewards()

    def _scene_init(self):
        self.scene.robot = RBQ10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The articulation root link is "trunk".
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        policy_dt = self.decimation * self.sim.dt
        self.scene.height_scanner_feet_fl.update_period = policy_dt
        self.scene.height_scanner_feet_fr.update_period = policy_dt
        self.scene.height_scanner_feet_rl.update_period = policy_dt
        self.scene.height_scanner_feet_rr.update_period = policy_dt

    def _action_init(self):
        self.actions.joint_pos = JointPositionActionWithLimitCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=True,
            soft_margin_deg=5.0,
        )

    def _terrain_init(self):
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
        # Spawn on level 3: from higher initial levels a cold-start policy
        # faceplants on the hard tiles and never recovers.
        self.scene.terrain.max_init_terrain_level = 3
        gen = self.scene.terrain.terrain_generator
        gen.border_width = 30.0
        gen.num_rows = 15
        gen.num_cols = 12
        gen.slope_threshold = None
        sub = gen.sub_terrains
        sub["pyramid_stairs"].proportion = 0.2
        sub["pyramid_stairs"].step_width = 0.25
        sub["pyramid_stairs"].step_height_range = (0.05, 0.20)
        sub["pyramid_stairs_inv"].proportion = 0.2
        sub["pyramid_stairs_inv"].step_width = 0.25
        sub["pyramid_stairs_inv"].step_height_range = (0.05, 0.20)
        sub["pyramid_stairs_nose"] = MeshPyramidStairsNosingTerrainCfg(
            proportion=0.1,
            step_width=0.3,
            step_height_range=(0.05, 0.20),
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            nose_depth=0.03,
            nose_thickness=0.03,
        )
        sub["pyramid_stairs_inv_nose"] = MeshInvertedPyramidStairsNosingTerrainCfg(
            proportion=0.1,
            step_width=0.3,
            step_height_range=(0.05, 0.20),
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            nose_depth=0.03,
            nose_thickness=0.03,
        )
        # Upstream's random-block "boxes" family is replaced by concentric ring
        # fences: a leading leg clears the wall while the trailing leg can stub.
        del sub["boxes"]
        sub["ring_fence"] = MultiRingFenceTerrainCfg(
            proportion=0.1,
            n_rings=4,
            thickness=0.06,
            height_range=(0.03, 0.15),
            platform_width=2.0,
            border=0.3,
        )
        sub["random_rough"].proportion = 0.1
        sub["random_rough"].platform_width = 3.0
        sub["random_rough"].noise_range = (0.01, 0.10)
        sub["hf_pyramid_slope"].proportion = 0.1
        sub["hf_pyramid_slope"].slope_range = (0.1, 0.5)
        sub["hf_pyramid_slope"].platform_width = 2.0
        sub["hf_pyramid_slope_inv"].proportion = 0.1
        sub["hf_pyramid_slope_inv"].slope_range = (0.1, 0.5)
        sub["hf_pyramid_slope_inv"].platform_width = 2.0

    def _command_init(self):
        self.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(3.0, 7.0),
            rel_standing_envs=0.2,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            # The command-range curriculum ramps 0.1x -> 1.0x of this terminal range.
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.2, 1.2),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )

    def _terminations_init(self):
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk"]

    def _validate(self):
        # Curriculum terms resolve reward terms by name at runtime; check them
        # before the zero-weight strip so declared-at-0 terms still count.
        reward_terms = {n for n in vars(self.rewards) if getattr(self.rewards, n) is not None}
        for name in ("penalty_terrain_schedule", "feet_touchdown_schedule"):
            term = getattr(self.curriculum, name, None)
            if term is not None:
                missing = set(term.params["term_names"]) - reward_terms
                if missing:
                    raise ValueError(f"{name} references unknown reward terms {sorted(missing)}")

    def _disable_zero_weight_rewards(self):
        """Drop reward terms declared at exactly weight 0.0 (kept declared for overrides)."""
        for attr in list(vars(self.rewards).keys()):
            term = getattr(self.rewards, attr)
            if hasattr(term, "weight") and term.weight == 0.0:
                setattr(self.rewards, attr, None)


def _apply_play(env_cfg: BlindRoughEnvCfg) -> None:
    """Small non-curriculum scene, full command range, no observation noise or pushes.

    Startup and reset dynamics randomization (friction, mass, COM, actuator
    gains) still fires, so this is not a nominal-dynamics rollout.
    """
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    # The command-range curriculum would hold play at the 0.1x start range
    # (it only ramps from training reward), so drop it here.
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None


@configclass
class BlindRoughEnvCfg_PLAY(BlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class BlindRoughEnvCfg_GAMEPAD(BlindRoughEnvCfg_PLAY):
    """Free play driven by a live gamepad; runs until interrupted."""

    def __post_init__(self):
        super().__post_init__()
        # Full stick deflection must stay inside the trained command distribution.
        self.commands.base_velocity = GamepadVelocityCommandCfg(
            asset_name="robot",
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.8, 0.8),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
            dead_zone=0.05,
            # The yaw stick re-centers worse than it reads; without the wider
            # dead zone a sloppy release latches a standing turn command.
            yaw_dead_zone=0.10,
            input_device="xbox",
        )
        self.episode_length_s = 1.0e9
        self.terminations.base_contact = None
