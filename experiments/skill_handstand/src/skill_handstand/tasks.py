"""Handstand S1 task: the blind DreamWaQ environment on flat ground with the external mode command.

Observation contract = the blind DreamWaQ groups with ``skill_command`` (3) appended to the policy group (so it is in
every history frame) and ``skill_command`` + ``skill_curriculum`` (3 + 2) appended to the critic group.
Payload is always 0; pushes are off in S1.
"""
from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

import gd_lab.mdp.rewards as gd_rew
from gd_lab.methods.dreamwaq import DreamwaqObservationsCfg
from gd_lab.tasks.blind_rough import BlindRoughEnvCfg

from . import terms
from .commands import SkillModeCommandCfg

# (target pitch deg, assist scale)
S1_LEVELS = [(30, 1.0), (60, 1.0), (90, 1.0), (90, 0.6), (90, 0.3), (90, 0.0)]
# v2: no assist wrench; part of the resets start in a balanced front-feet pose instead (start_poses.json)
S1V2_LEVELS = [(30, 0.0), (60, 0.0), (90, 0.0)]


@configclass
class HandstandObservationsCfg(DreamwaqObservationsCfg):
    @configclass
    class PolicyCfg(DreamwaqObservationsCfg.PolicyCfg):
        skill_command = ObsTerm(func=terms.skill_command)

    @configclass
    class CriticCfg(DreamwaqObservationsCfg.CriticCfg):
        skill_command = ObsTerm(func=terms.skill_command)
        skill_curriculum = ObsTerm(func=terms.skill_curriculum)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HandstandCommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(1.0e9, 1.0e9), rel_standing_envs=1.0, heading_command=False,
        debug_vis=False, ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0),
                                                                 ang_vel_z=(0.0, 0.0)))
    skill_mode = SkillModeCommandCfg(levels=S1_LEVELS)


@configclass
class HandstandRewardsCfg:
    # -- skill (gated by the mode command)
    hs_orientation = RewTerm(func=terms.hs_orientation, weight=3.0)
    hs_front_support = RewTerm(func=terms.hs_front_support, weight=1.0)
    hs_hind_contact = RewTerm(func=terms.hs_hind_contact, weight=-1.0)
    hs_hold = RewTerm(func=terms.hs_hold, weight=2.0)
    hs_upright = RewTerm(func=terms.hs_upright, weight=2.0)
    hs_still_when_idle = RewTerm(func=terms.hs_still_when_idle, weight=-1.0)
    # -- safety / regularisation (DreamWaQ values where they exist)
    undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]), "threshold": 1.0})
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-2.0)
    torque_saturation = RewTerm(func=terms.torque_saturation, weight=-0.5)
    dof_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-2.5e-5)
    dof_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    joint_power = RewTerm(func=gd_rew.joint_power_l1, weight=-2e-5)
    dof_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-5e-4)
    smoothness = RewTerm(func=gd_rew.action_acceleration_l2, weight=-0.02)


@configclass
class HandstandCurriculumCfg:
    handstand_levels = CurrTerm(func=terms.handstand_levels)


@configclass
class HandstandS1EnvCfg(BlindRoughEnvCfg):
    observations: HandstandObservationsCfg = HandstandObservationsCfg()
    rewards: HandstandRewardsCfg = HandstandRewardsCfg()
    commands: HandstandCommandsCfg = HandstandCommandsCfg()
    curriculum: HandstandCurriculumCfg = HandstandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # payload 0 (still observed, always 0), no pushes in S1
        self.events.randomize_payload.params["payload_masses"] = (0.0,)
        self.events.randomize_payload.params["payload_probabilities"] = (1.0,)
        self.events.push_robot = None
        self.events.assist_wrench = EventTerm(func=terms.assist_wrench, mode="interval", interval_range_s=(0.0, 0.0))

    def _terrain_init(self):
        self.scene.terrain.max_init_terrain_level = None
        gen = self.scene.terrain.terrain_generator
        gen.border_width = 10.0
        gen.num_rows = 8
        gen.num_cols = 8
        gen.curriculum = False
        gen.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}

    def _command_init(self):
        pass   # HandstandCommandsCfg: zero velocity command (S1 does not walk) + the mode command


@configclass
class HandstandS1V2EnvCfg(HandstandS1EnvCfg):
    """v1 found the assist torque doing all the rotation (the policy never learned to lift itself; MuJoCo without the
    wrench did not move). v2: no wrench, 30 % of resets start in a statically balanced handstand pose."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.skill_mode.levels = S1V2_LEVELS
        self.events.assist_wrench = None
        self.events.start_in_handstand = EventTerm(func=terms.start_in_handstand, mode="reset",
                                                   params={"fraction": 0.3})


@configclass
class HandstandS1V3EnvCfg(HandstandS1V2EnvCfg):
    """v2 learned to hold from start poses but never lifted off from a stand (max pitch ~30 deg).
    v3: + lift-off start poses 10-40 deg (front feet under the COM; start_poses.json), 40 % start-pose resets,
    + front-load share reward (weight shift before lift-off), success tolerance 15 -> 10 deg (v2 passed 30 deg at 15 deg)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.hs_front_load = RewTerm(func=terms.hs_front_load, weight=1.5)
        self.commands.skill_mode.success_pitch_tol = 0.17
        self.events.start_in_handstand.params["fraction"] = 0.4


@configclass
class HandstandS2ObservationsCfg(HandstandObservationsCfg):
    @configclass
    class PolicyCfg(HandstandObservationsCfg.PolicyCfg):
        skill_walk = ObsTerm(func=terms.skill_walk)

    @configclass
    class CriticCfg(HandstandObservationsCfg.CriticCfg):
        skill_walk = ObsTerm(func=terms.skill_walk)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HandstandS2EnvCfg(HandstandS1V3EnvCfg):
    """S2 = stepping in place in the handstand (walk flag on, velocity command 0), from S1 v3 model_400.
    + vendor-safety margins from the RBQ SDK check: joint soft margin 5 -> 8 deg (Motion JOINT_LIMIT_PROTECTION,
    3 deg margin; S1 model_599 hit the knee limit), target-step penalty above 10 deg per step, stronger saturation
    penalty."""

    observations: HandstandS2ObservationsCfg = HandstandS2ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.skill_mode.levels = [(90, 0.0)]
        self.commands.skill_mode.walk_prob = 0.8
        self.actions.joint_pos.soft_margin_deg = 8.0
        self.rewards.hs_step_air_time = RewTerm(func=terms.hs_step_air_time, weight=4.0)
        self.rewards.hs_front_alternation = RewTerm(func=terms.hs_front_alternation, weight=0.5)
        self.rewards.hs_stay_in_place = RewTerm(func=terms.hs_stay_in_place, weight=-1.0)
        self.rewards.target_step_excess = RewTerm(func=terms.target_step_excess, weight=-1.0)
        self.rewards.torque_saturation.weight = -1.0


@configclass
class HandstandS2BEnvCfg(HandstandS2EnvCfg):
    """S2b: S2 v1 policies (model_100+) held the handstand in Isaac but overshot past vertical and fell in MuJoCo.
    + Coulomb joint friction 0-1 N m (FrictionDCMotor), overshoot and near-vertical pitch-rate penalties,
    + swing-height reward, min air 0.15 -> 0.10 s. The hold anchor to S1 model_400 is in the PPO (agent.py)."""

    def __post_init__(self):
        super().__post_init__()
        from .actuators import FrictionDCMotor
        for name in ("rbq_hip", "rbq_knee"):
            act = self.scene.robot.actuators[name]
            act.class_type = FrictionDCMotor
            act.friction_max = 1.0
        self.rewards.hs_overshoot = RewTerm(func=terms.hs_overshoot, weight=-5.0)
        self.rewards.hs_pitch_rate_near_vertical = RewTerm(func=terms.hs_pitch_rate_near_vertical, weight=-0.05)
        self.rewards.hs_swing_height = RewTerm(func=terms.hs_swing_height, weight=1.0)
        self.rewards.hs_step_air_time.params = {"min_air": 0.10, "max_air": 0.4}


@configclass
class HandstandS2CEnvCfg(HandstandS2BEnvCfg):
    """S2c: S2b model_650 steps in the handstand (31 front touchdowns / 6 s in MuJoCo) but drifts 0.4-0.6 m.
    + stay-in-place weight -1 -> -3, + distance-from-walk-start penalty (5 cm deadband)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.hs_stay_in_place.weight = -3.0
        self.rewards.hs_walk_drift = RewTerm(func=terms.hs_walk_drift, weight=-2.0)
