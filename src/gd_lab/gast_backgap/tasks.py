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
from gd_lab.mdp.platform_gap_attempts import CONTACT_FORCE
from gd_lab.mdp.platform_gap_finetune import GapMonitor

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


class MotionDirectionGapMonitor(GapMonitor):
    """V2 (2026-10-24): ``GapMonitor`` whose attempt direction is the walking direction, not the heading.

    The attempt tracker picks the slot "in front" from ``heading.cos() >= 0``. Walking backward the robot moves away
    from that slot, so a backward crossing never became an attempt: no clean bonus (platform_gap_clean,
    gap_foot_clean) and no clean G in the Top-5. Here the tracker gets "moving toward +x" = heading +x XOR backward.
    Intrusion cost is per slot and already direction-free.
    """

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, strict_contact: bool = False):
        if self._crossing is None:
            self._check_order(env)
            self._check_feet(env, asset_cfg, sensor_cfg)
        robot = env.scene[asset_cfg.name]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
        origins = env.scene.env_origins
        slots, lower = self.geometry.for_envs(env)
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        cmd = env.command_manager.get_term("base_velocity")
        backward = cmd.backward if hasattr(cmd, "backward") else cmd.vel_command_b[:, 0] < 0
        toward_plus_x = (robot.data.heading_w.cos() >= 0) ^ backward
        result = self.tracker.update(
            feet[..., 0] - origins[:, None, 0], feet[..., 2], slots, lower + origins[:, None, 2],
            toward_plus_x, boarding_family_mask(env), self._crossing.last_event, force > CONTACT_FORCE,
        )
        self.cost, self.clean_event = result.cost, result.clean_event
        return torch.zeros(env.num_envs, device=env.device)


def backward_nose_up(env, limit_deg: float = 8.0):
    """V2: per second, nose-up pitch above ``limit_deg`` (rad) on gap tiles while the command is backward.

    MuJoCo (2026-10-23): backing over a gap lifts the front 10-20 deg; failed runs kept rising to 70-90 deg and
    flipped backward. The v2.1 gap pitch term (projected gravity x squared) is ~0.07 at 15 deg.
    nose-up angle = asin(-g_x) with g = projected gravity in the body frame.
    """
    g = env.scene["robot"].data.projected_gravity_b
    nose_up = torch.asin((-g[:, 0]).clamp(-1.0, 1.0))
    cmd = env.command_manager.get_term("base_velocity")
    backward = cmd.backward if hasattr(cmd, "backward") else cmd.vel_command_b[:, 0] < 0
    excess = (nose_up - math.radians(limit_deg)).clamp_min(0.0)
    return excess * (backward & boarding_family_mask(env)).float()


@configclass
class GastRayIrV21Gap50BackV2TeacherCfg(GastRayIrV21Gap50BackTeacherCfg):
    """V2: backward crossings count as attempts (clean bonus, clean G) + backward nose-up penalty on gap tiles."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.platform_gap_monitor.func = MotionDirectionGapMonitor
        self.rewards.backward_nose_up = RewardTermCfg(func=backward_nose_up, weight=-5.0, params={"limit_deg": 8.0})


TASK = "GastRayIrV21Gap50Back"
TASK_ID = "Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0"
TASK_ID_V2 = "Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0"
assert math.isclose(BACK_SPEED_MAX, 1.0)
