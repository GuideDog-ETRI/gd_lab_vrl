"""Configuration for the RBQ-10 quadruped."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from gd_lab.core.paths import ASSETS_DIR
from gd_lab.robots.actuators import DelayedDCMotorCfg

RBQ10_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSETS_DIR}/robots/rbq10/urdf/rbq10_simple.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "FL_HIP": 0.0,
            "HL_HIP": 0.0,
            "FR_HIP": 0.0,
            "HR_HIP": 0.0,
            "FL_THIGH": 0.76,
            "HL_THIGH": 0.76,
            "FR_THIGH": 0.76,
            "HR_THIGH": 0.76,
            "FL_KNEE": -1.45,
            "HL_KNEE": -1.45,
            "FR_KNEE": -1.45,
            "HR_KNEE": -1.45,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Hip roll/pitch spec: max torque 104 Nm, max angular vel 14.4 rad/s,
        # rotor inertia 0.014058. DC motor torque-speed curve: stall torque =
        # max torque, no-load speed = max angular vel. effort_limit is derated
        # to 90% of peak spec as a sim-side torque cap; saturation_effort and
        # velocity_limit stay at peak so the curve itself matches the real
        # motor. stiffness/damping match the robot's deploy gains.
        "rbq_hip": DelayedDCMotorCfg(
            joint_names_expr=[".*_HIP", ".*_THIGH"],
            effort_limit=93.6,  # 104 * 0.9
            saturation_effort=104,
            velocity_limit=14.4,
            stiffness=123.39,
            damping=2.5,
            armature=0.014058,
            min_delay=0,
            max_delay=1,
        ),
        # Knee spec: max torque 140 Nm, max angular vel 11.15 rad/s, rotor
        # inertia 0.0214816. Same derating and gain-matching as the hip.
        "rbq_knee": DelayedDCMotorCfg(
            joint_names_expr=[".*_KNEE"],
            effort_limit=126.0,  # 140 * 0.9
            saturation_effort=140,
            velocity_limit=11.15,
            stiffness=127.77,
            damping=2.5,
            armature=0.0214816,
            min_delay=0,
            max_delay=1,
        ),
    },
)
