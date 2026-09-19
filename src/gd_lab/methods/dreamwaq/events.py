"""Domain randomization plus the per-episode trunk payload for blind DreamWaQ."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import gd_lab.mdp.events as gd_events


@configclass
class DreamwaqEventsCfg:
    # startup
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.16, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # Build tolerance only; heavy-load coverage lives in randomize_payload.
    add_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    randomize_base_inertia = EventTerm(
        func=gd_events.randomize_body_inertia_diag,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "scale_range": (0.85, 1.15)},
    )
    # reset
    # A task variable, not DR: the drawn value is observed, so the policy
    # conditions on the load rather than averaging over it.
    randomize_payload = EventTerm(
        func=gd_events.randomize_payload_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "payload_masses": (0.0, 5.0),
        },
    )
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    # interval. Deliberately unrelated to the command phase: a disturbance tied
    # to the stand-to-walk transition, or a constant unobservable wrench, teaches
    # the policy to compensate a force it cannot identify at standstill, which
    # shows up on hardware as drift under a stand command.
    push_robot = EventTerm(
        func=gd_events.push_by_setting_velocity_tracked,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
