"""GAST-Narrow v2.2: v2.1 (Top-1 465) + pacing fix (design: GAST_NARROW_V22_DESIGN.md).

E2 / MuJoCo (2026-10-27) on 465: gap, stairs and stand all met (gap cross >= 96 % in every condition), but
- tight passable passages (0.66 m, m = 0.05) stall: 465 passes 53-94 % (1111 66-97 %), every failure a stand-still;
- MuJoCo narrow_only: two ~1 s bursts of 1.5-2.0 m/s right before the corridor exit and in the L's second leg
  (command 0.5 / 0.35). The one-shot +8 pass bonus still pays for reaching the exit early.
v2.2 changes three reward terms only (terrain, anchor, commands, observations, Top-5 as v2.1):
- narrow_success: the +8 pass bonus is scaled by the pace factor clamp(1 - extra / PACE_EXTRA_M, 0, 1), extra = the
  distance run above |command| + OVERSPEED_MARGIN inside the passage (integral of the overspeed, m). Arriving early
  no longer pays.
- narrow_overspeed: weight -2 -> -1 (the bonus gating carries the pacing; a large per-step cost made standing cheap).
- narrow_stall (new): -1 /s on passable passages before the exit while |command| > STALL_CMD and the planar speed
  stays below STILL_SPEED for more than STALL_S.
"""
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_narrow import mdp_v2
from gd_lab.gast_narrow.state import narrow_state
from gd_lab.gast_narrow.tasks_v21 import OVERSPEED_MARGIN, GastNarrowV21RunnerCfg, GastNarrowV21TeacherCfg

PACE_EXTRA_M = 0.5
OVERSPEED_WEIGHT = -1.0
STALL_CMD, STALL_S, STALL_WEIGHT = 0.2, 1.5, -1.0


def _speeds(env):
    command = env.command_manager.get_term("base_velocity").vel_command_b[:, :2].norm(dim=-1)
    speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
    return command, speed


class NarrowPacedPassBonus(ManagerTermBase):
    """v2 pass bonus (pass event / dt) times the pace factor of the episode's run through the passage."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.extra = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.extra[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env):
        state = narrow_state(env)
        command, speed = _speeds(env)
        inside = state.narrow & ~state.past_exit
        self.extra += (speed - command - OVERSPEED_MARGIN).clamp_min(0.0) * inside.float() * env.step_dt
        pace = (1.0 - self.extra / PACE_EXTRA_M).clamp(0.0, 1.0)
        return mdp_v2.narrow_pass_bonus(env) * pace


class NarrowStallPenalty(ManagerTermBase):
    """1 per second once a passable-passage run has stood still for STALL_S under a moving command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.still = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.still[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env):
        state = narrow_state(env)
        command, speed = _speeds(env)
        passable = state.narrow & ~mdp_v2.narrow_episode(env).blocked & ~state.past_exit
        stuck = passable & (command > STALL_CMD) & (speed < mdp_v2.STILL_SPEED)
        self.still = torch.where(stuck, self.still + env.step_dt, torch.zeros_like(self.still))
        return (self.still > STALL_S).float()


@configclass
class GastNarrowV22TeacherCfg(GastNarrowV21TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        assert self.rewards.narrow_success.func is mdp_v2.narrow_pass_bonus
        self.rewards.narrow_success.func = NarrowPacedPassBonus
        self.rewards.narrow_overspeed.weight = OVERSPEED_WEIGHT
        self.rewards.narrow_stall = RewardTermCfg(func=NarrowStallPenalty, weight=STALL_WEIGHT)


@configclass
class GastNarrowV22RunnerCfg(GastNarrowV21RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/narrow_v22"


TASK = "GastNarrowV22"
TASK_ID = "Gd-GastNarrowV22-Rbq10-Dreamwaq-v0"
SOURCE_TASK_ID = "Gd-GastNarrowV21-Rbq10-Dreamwaq-v0"
