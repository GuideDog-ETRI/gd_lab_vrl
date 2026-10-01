"""VRL runner config with the gated actor-critic; everything else inherited."""

from isaaclab.utils import configclass

from gd_lab.teachers.cvtt.agent_cfg import DreamwaqVrlRunnerCfg


@configclass
class BlindStartRunnerCfg(DreamwaqVrlRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "gd_lab.teachers.bivt.actor_critic:DreamwaqVrlGatedActorCritic"
