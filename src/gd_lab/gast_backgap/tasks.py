"""Backward gap crossing on top of the original Ray+IR 3906 teacher (design: docs/design/GAST_BACKGAP_DESIGN_DRAFT_20261022.md).

Only new classes; the 3906 task (``Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0``) and its code stay unchanged.

- Gap tiles: with probability ``p_back`` (0.4 -> 0.7 over the first 200 updates) the boarding command is
  reversed: same heading (+/-x, facing away from the gap behind), vx = -|v|, |v| from the v2.1 speed mix capped
  at 1.0 m/s. Height offsets and widths follow the same curriculum as forward crossings.
- ``gap_hind_hop`` only for forward commands; ``gap_front_hop`` (front pair, same weight) for backward ones:
  the trailing pair is the one that hops.
- Command metrics (TensorBoard ``Metrics/base_velocity/*``): backward share of gap envs and crossings by direction.
"""
import math
import os

import torch
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from gd_lab.gast.tasks import GastRayIrV21Gap50TeacherCfg
from gd_lab.mdp.gap_stair_v2_math import sample_v21_speed_in_ceiling, speed_ceiling
from gd_lab.mdp.gap_stair_v21 import GapHindHop, V21VelocityCommand
from gd_lab.mdp.platform_gap_terms import boarding_family_mask

BACK_SPEED_MAX = 1.0
P_BACK_START, P_BACK_END = 0.4, 0.7
P_BACK_RAMP_STEPS = 200 * 100  # 200 PPO updates x 100 steps (common_step_counter)


def p_back(step: int, start: float = P_BACK_START, end: float = P_BACK_END, ramp: int = P_BACK_RAMP_STEPS) -> float:
    return end if ramp <= 0 else start + (end - start) * min(1.0, step / ramp)


class BackGapVelocityCommand(V21VelocityCommand):
    """v2.1 boarding command; a share of the gap-tile commands is reversed."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.backward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.p_back_ramp = int(os.environ.get("GD_LAB_BACKGAP_RAMP_STEPS", P_BACK_RAMP_STEPS))
        for key in ("gap_backward", "gap_forward", "crossed_backward", "crossed_forward"):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        gap = ids[boarding_family_mask(self._env)[ids]]
        self.backward[ids] = False
        if not len(gap):
            return
        step = int(self._env.common_step_counter)
        back = gap[torch.rand(len(gap), device=self.device) < p_back(step, ramp=self.p_back_ramp)]
        if len(back):
            ceiling = min(BACK_SPEED_MAX, speed_ceiling(step, self.ramp_steps))
            self.vel_command_b[back, 0] = -sample_v21_speed_in_ceiling(len(back), ceiling, self.device)
            self.vel_command_b[back, 1] = 0.0
            self.backward[back] = True
        if os.environ.get("GD_LAB_BACKGAP_DEBUG"):
            print(f"[BACKGAP] step={step} resample={len(ids)} gap={len(gap)} back={len(back)} "
                  f"vx_back={self.vel_command_b[back, 0][:4].tolist()}", flush=True)

    def _update_metrics(self):
        super()._update_metrics()
        gap = boarding_family_mask(self._env).float()
        back = self.backward.float() * gap
        self.metrics["gap_backward"] = back
        self.metrics["gap_forward"] = gap - back
        try:
            crossed = self._env.reward_manager.get_term_cfg("platform_gap_crossing").func.achieved.any(dim=1).float()
        except (AttributeError, ValueError):
            return
        self.metrics["crossed_backward"] = torch.maximum(self.metrics["crossed_backward"], crossed * back)
        self.metrics["crossed_forward"] = torch.maximum(self.metrics["crossed_forward"], crossed * (gap - back))


class DirectionalGapHop(GapHindHop):
    """``GapHindHop`` (both feet of ``sensor_cfg`` airborne near a slot), counted only for one command direction."""

    def __call__(self, env, sensor_cfg: SceneEntityCfg, reach: float = 0.6, min_air_s: float = 0.05,
                 backward: bool = False):
        value = super().__call__(env, sensor_cfg, reach, min_air_s)
        cmd = env.command_manager.get_term("base_velocity")
        moving_back = cmd.backward if hasattr(cmd, "backward") else cmd.vel_command_b[:, 0] < 0
        return value * (moving_back if backward else ~moving_back).float()


@configclass
class GastRayIrV21Gap50BackTeacherCfg(GastRayIrV21Gap50TeacherCfg):
    """3906 task + reversed gap commands + direction-aware hop penalties."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type = BackGapVelocityCommand
        hind = self.rewards.gap_hind_hop
        self.rewards.gap_hind_hop = RewardTermCfg(func=DirectionalGapHop, weight=hind.weight,
                                                  params={**hind.params, "backward": False})
        front = SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot"])
        self.rewards.gap_front_hop = RewardTermCfg(func=DirectionalGapHop, weight=hind.weight,
                                                   params={"sensor_cfg": front, "backward": True})


TASK = "GastRayIrV21Gap50Back"
TASK_ID = "Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0"
assert math.isclose(BACK_SPEED_MAX, 1.0)
