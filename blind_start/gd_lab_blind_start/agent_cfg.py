"""VRL runner config with the gated actor-critic; everything else inherited."""

from gd_lab.agents.dreamwaq_vrl_ppo_cfg import DreamwaqVrlRunnerCfg
from isaaclab.utils import configclass


@configclass
class BlindStartRunnerCfg(DreamwaqVrlRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "gd_lab_blind_start.actor_critic:DreamwaqVrlGatedActorCritic"
