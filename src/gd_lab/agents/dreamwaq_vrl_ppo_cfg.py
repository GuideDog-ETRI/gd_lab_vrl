"""RSL-RL runner configuration for the stage-1 vision teacher.

Subclasses the blind config rather than copying it, so PPO hyperparameters,
symmetry and the observation-layout fields keep tracking upstream. Only the
actor-critic class and the terrain-encoder fields differ.
"""

from __future__ import annotations

import dataclasses

from isaaclab.utils import configclass

from gd_lab.agents.dreamwaq_ppo_cfg import DreamwaqActorCriticCfg, DreamwaqRunnerCfg
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC


@configclass
class DreamwaqVrlActorCriticCfg(DreamwaqActorCriticCfg):
    class_name: str = "gd_lab.rl.actor_critic_vrl:DreamwaqVrlActorCritic"

    # Keep the network env-agnostic: its layout comes from the method spec here.
    height_scan_start: int = DREAMWAQ_SPEC.critic.offset("height_scan")
    terrain_latent_dim: int = 32
    terrain_encoder_hidden_channels: tuple[int, int] = (8, 16)
    # GridPatternCfg(resolution=0.1, size=[1.6,1.0]) -> 17 along x, 11 along y.
    height_scan_grid_shape: tuple[int, int] = (11, 17)


def _vrl_policy() -> DreamwaqVrlActorCriticCfg:
    """The blind policy config, field for field, with the vision class swapped in.

    Copying the live instance instead of restating its literals means an
    upstream change to e.g. ``actor_hidden_dims`` reaches the teacher without
    anyone remembering to mirror it here.
    """
    # Instantiated, not read off the class: configclass exposes nested configs
    # through a default_factory, so the class attribute does not exist.
    base = DreamwaqRunnerCfg().policy
    cfg = DreamwaqVrlActorCriticCfg()
    for field in dataclasses.fields(base):
        if field.name != "class_name":
            setattr(cfg, field.name, getattr(base, field.name))
    return cfg


@configclass
class DreamwaqVrlRunnerCfg(DreamwaqRunnerCfg):
    # Terrain visibility is a second spatial channel and must mirror with the
    # same y-axis permutation as the height grid.
    symmetry_cfg = {"data_augmentation_func": "gd_lab.methods.dreamwaq.vrl_symmetry:mirror_vrl_observations"}
    experiment_name = "vision_rbq10_dreamwaq"
    policy = _vrl_policy()
