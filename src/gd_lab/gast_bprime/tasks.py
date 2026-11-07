"""Stage B' task: GAST-Narrow v2.5 passages (v25p: no turn / cliff tiles) + gait v3 G5 terms, from the A' model.

Design: docs/hangang/WORKORDER_STAGE_BPRIME_20261101.md. The 3906 / backgap / gast_narrow / gast_gait task code is
unchanged; this class composes them:
- parent GastNarrowV25TeacherCfg (launcher NARROW_TURN=0 NARROW_CLIFF_COLUMNS=0: the 44-column v2.3 terrain with
  backward passages, rear Mid-360, trunk self-occlusion);
- gait v3 terms at the A' weights (apply_gait_v3; ramp off with GAIT_RAMP_STEPS=0), R1 at 3 deg on narrow tiles too;
- gap_nose_up / gap_nose_peak / narrow_nose_up removed (A' has only R1 for the trunk pitch; D1, D3);
- narrow tiles: downward terrain input and R1 / R6 / R15 use the flat floor, walls only through the LiDAR (D4);
- command: v2.5 narrow command + R20 backward gap speeds;
- (11-02, after the bprime_cont margin sweep) balanced pass / block margin draws at the band widths (gast_bprime.margin)
  and the passable-passage stall penalty at 1.0 s / -2 (v2.2: 1.5 s / -1): the continuation learned the stop from the
  width alone (w 0.66 / m 0.05 passed 17.5 %, 602 98.1 %).
"""
from __future__ import annotations

import torch
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_bprime.commands import BPrimeVelocityCommand
from gd_lab.gast_bprime.margin import reset_narrow_bprime
from gd_lab.gast_bprime.terrain_obs import NarrowFloorRaycastIrTerrain
from gd_lab.gast.rayir import RaycastIrTerrain
from gd_lab.gast_gait.tasks import apply_gait_v3
from gd_lab.gast_narrow import mdp_cliff, mdp_v2
from gd_lab.gast_narrow.state import narrow_state
from gd_lab.gast_narrow.tasks_v22 import STALL_CMD, NarrowStallPenalty, _speeds
from gd_lab.gast_narrow.tasks_v25 import GastNarrowV25RunnerCfg, GastNarrowV25TeacherCfg, NarrowStallPenaltyV25

STALL_S, STALL_WEIGHT = 1.0, -2.0   # v2.2: 1.5 s, -1.0


class BPrimeStallPenalty(NarrowStallPenaltyV25):
    """v2.2's stall penalty with STALL_S = 1.0 s, off on cliff tiles."""

    def __call__(self, env):
        state = narrow_state(env)
        command, speed = _speeds(env)
        passable = state.narrow & ~mdp_v2.narrow_episode(env).blocked & ~state.past_exit
        stuck = passable & (command > STALL_CMD) & (speed < mdp_v2.STILL_SPEED)
        self.still = torch.where(stuck, self.still + env.step_dt, torch.zeros_like(self.still))
        return (self.still > STALL_S).float() * (state.kind != mdp_cliff.CLIFF).float()


@configclass
class GastNarrowBPrimeTeacherCfg(GastNarrowV25TeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_gait_v3(self, floor_on_narrow=True)
        self.commands.base_velocity.class_type = BPrimeVelocityCommand  # after apply_gait_v3 (R20 class there)
        self.rewards.gap_nose_up = None
        self.rewards.gap_nose_peak = None
        self.rewards.narrow_nose_up = None
        assert self.observations.terrain.camera_visible.func is RaycastIrTerrain
        self.observations.terrain.camera_visible = ObservationTermCfg(func=NarrowFloorRaycastIrTerrain)
        assert self.events.reset_narrow_v2.func is mdp_v2.reset_narrow_v2
        self.events.reset_narrow_v2.func = reset_narrow_bprime  # same slot: the turn / backward resets follow it
        assert self.rewards.narrow_stall.func is NarrowStallPenaltyV25 and issubclass(NarrowStallPenaltyV25, NarrowStallPenalty)
        self.rewards.narrow_stall.func = BPrimeStallPenalty
        self.rewards.narrow_stall.weight = STALL_WEIGHT


@configclass
class GastNarrowBPrimeRunnerCfg(GastNarrowV25RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/lineage16_bprime"


TASK = "GastNarrowBPrime"
TASK_ID = "Gd-GastNarrowBPrime-Rbq10-Dreamwaq-v0"
