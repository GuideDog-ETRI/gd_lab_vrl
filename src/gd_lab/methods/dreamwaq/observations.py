"""IsaacLab observation groups for blind DreamWaQ, matching the spec in spec.py.

Term order, dims and history here must equal ``DREAMWAQ_SPEC``; the mirror
augmentation validates that equality against the live environment at startup.
The noise / clip / scale values are a deploy contract: policy and critic must
observe the same quantity the same way, and the on-robot controller scales its
inputs to match - so each lives in exactly one place (``_proprio_obs``).
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import gd_lab.mdp.observations as gd_obs
from gd_lab.core.types import FOOT_ORDER

from .spec import HISTORY_LENGTH

_PROPRIO_SPECS = {
    "base_ang_vel": dict(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-400.0, 400.0), scale=0.25),
    "projected_gravity": dict(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0)),
    "velocity_commands": dict(
        func=base_mdp.generated_commands,
        params={"command_name": "base_velocity"},
        scale=(2.0, 2.0, 0.25),
    ),
    "joint_pos": dict(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0), scale=1.0),
    "joint_vel": dict(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-2000.0, 2000.0), scale=0.05),
    "actions": dict(func=base_mdp.last_action),
    "base_lin_vel": dict(func=base_mdp.base_lin_vel, clip=(-50.0, 50.0), scale=2.0),
}


def _proprio_obs(name: str, **kwargs) -> ObsTerm:
    return ObsTerm(**_PROPRIO_SPECS[name], **kwargs)


def _feet_contact_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=list(FOOT_ORDER), preserve_order=True)


def _feet_around_height_params() -> dict:
    return {
        "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_ORDER), preserve_order=True),
        "sensor_cfg_fl": SceneEntityCfg("height_scanner_feet_fl"),
        "sensor_cfg_fr": SceneEntityCfg("height_scanner_feet_fr"),
        "sensor_cfg_rl": SceneEntityCfg("height_scanner_feet_rl"),
        "sensor_cfg_rr": SceneEntityCfg("height_scanner_feet_rr"),
    }


@configclass
class DreamwaqObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Noisy proprioception, stacked history for the CENet."""

        base_ang_vel = _proprio_obs("base_ang_vel")
        projected_gravity = _proprio_obs("projected_gravity")
        velocity_commands = _proprio_obs("velocity_commands")
        joint_pos = _proprio_obs("joint_pos")
        joint_vel = _proprio_obs("joint_vel")
        actions = _proprio_obs("actions")

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = HISTORY_LENGTH
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Clean current proprioception + privileged state for the asymmetric critic."""

        base_ang_vel = _proprio_obs("base_ang_vel")
        projected_gravity = _proprio_obs("projected_gravity")
        velocity_commands = _proprio_obs("velocity_commands")
        joint_pos = _proprio_obs("joint_pos")
        joint_vel = _proprio_obs("joint_vel")
        actions = _proprio_obs("actions")
        # CENet velocity target.
        base_lin_vel = _proprio_obs("base_lin_vel")
        # Terrain-relative per-foot heights (4 feet x 2x2 rays); world-frame z
        # would alias with the terrain level.
        feet_around_height_from_terrain = ObsTerm(
            func=gd_obs.feet_around_height_from_terrain,
            params=_feet_around_height_params(),
            clip=(-1.0, 1.0),
            scale=5.0,
        )
        feet_contact = ObsTerm(
            func=gd_obs.feet_contact_on_terrain,
            params={"sensor_cfg": _feet_contact_cfg()},
        )
        feet_contact_forces = ObsTerm(
            func=gd_obs.feet_contact_forces,
            params={"sensor_cfg": _feet_contact_cfg()},
            clip=(-1000.0, 1000.0),
            scale=0.002,
        )
        friction_coeff = ObsTerm(func=gd_obs.friction_coeff)
        base_mass_offset = ObsTerm(func=gd_obs.base_mass_offset, scale=0.2)
        actuator_gain_scale = ObsTerm(func=gd_obs.actuator_gain_scale)
        height_scan = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=5.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
