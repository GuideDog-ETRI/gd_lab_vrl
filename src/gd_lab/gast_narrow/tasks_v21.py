"""GAST-Narrow v2.1: v2 (Top-1 1111) + retention fix (design: GAST_NARROW_V21_DESIGN.md).

E2 (2026-10-27): 1111 kept stairs and narrow skills but its clean gap crossing fell below 6566 (forward 69 vs 77 %,
backward with the LiDAR off 68 vs 76 %). v1/v2 had cut the gap tiles from half of 3906/6566's terrain to 3 of 31
columns. v2.1 changes two things only:
- terrain: platform_gap 3 -> 8 columns (36 columns, gap 22 %); everything else as v2;
- algorithm: NarrowAnchorPPO -- after each update, the non-narrow samples are pulled toward the frozen 6566 action;
- reward: narrow_overspeed -2 /s per m/s of planar speed above |command| + 0.15 on narrow tiles. v1/v2 learned to
  hurry through passages (the one-shot +8 pass bonus ends the episode; 1111 ran ~1.5x its command in Isaac and
  2-4 m/s bursts in MuJoCo with the head LiDAR on, 2026-10-27).
Commands, observations, narrow tiles and the Top-5 score are v2's.
"""
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_narrow.model import GastNarrowV2RunnerCfg
from gd_lab.gast_narrow.tasks_v2 import GastNarrowV2TeacherCfg

GAP_COLUMNS = 8.0
OVERSPEED_MARGIN, OVERSPEED_WEIGHT = 0.15, -2.0


def narrow_overspeed(env, margin: float = OVERSPEED_MARGIN):
    """Per second on narrow tiles: planar speed above |command_xy| + margin (m/s), 0 elsewhere."""
    from gd_lab.gast_narrow.state import narrow_state
    command = env.command_manager.get_term("base_velocity").vel_command_b[:, :2].norm(dim=-1)
    speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
    return (speed - command - margin).clamp_min(0.0) * narrow_state(env).narrow.float()


@configclass
class GastNarrowV21TeacherCfg(GastNarrowV2TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains["platform_gap"].proportion = GAP_COLUMNS
        gen.num_cols = int(sum(sub.proportion for sub in gen.sub_terrains.values()))
        assert gen.num_cols == 36, gen.num_cols
        self.rewards.narrow_overspeed = RewardTermCfg(func=narrow_overspeed, weight=OVERSPEED_WEIGHT)


@configclass
class GastNarrowV21RunnerCfg(GastNarrowV2RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "gd_lab.gast_narrow.anchor:NarrowAnchorPPO"
        self.experiment_name = "gast/narrow_v21"


TASK = "GastNarrowV21"
TASK_ID = "Gd-GastNarrowV21-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV2-Rbq10-Dreamwaq-v0"
