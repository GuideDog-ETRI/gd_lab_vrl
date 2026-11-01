"""Stage B' task: GAST-Narrow v2.5 passages (v25p: no turn / cliff tiles) + gait v3 G5 terms, from the A' model.

Design: docs/hangang/WORKORDER_STAGE_BPRIME_20261101.md. The 3906 / backgap / gast_narrow / gast_gait task code is
unchanged; this class composes them:
- parent GastNarrowV25TeacherCfg (launcher NARROW_TURN=0 NARROW_CLIFF_COLUMNS=0: the 44-column v2.3 terrain with
  backward passages, rear Mid-360, trunk self-occlusion);
- gait v3 terms at the A' weights (apply_gait_v3; ramp off with GAIT_RAMP_STEPS=0), R1 at 3 deg on narrow tiles too;
- gap_nose_up / gap_nose_peak / narrow_nose_up removed (A' has only R1 for the trunk pitch; D1, D3);
- narrow tiles: downward terrain input and R1 / R6 / R15 use the flat floor, walls only through the LiDAR (D4);
- command: v2.5 narrow command + R20 backward gap speeds.
"""
from __future__ import annotations

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_bprime.commands import BPrimeVelocityCommand
from gd_lab.gast_bprime.terrain_obs import NarrowFloorRaycastIrTerrain
from gd_lab.gast.rayir import RaycastIrTerrain
from gd_lab.gast_gait.tasks import apply_gait_v3
from gd_lab.gast_narrow.tasks_v25 import GastNarrowV25RunnerCfg, GastNarrowV25TeacherCfg


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


@configclass
class GastNarrowBPrimeRunnerCfg(GastNarrowV25RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/lineage16_bprime"


TASK = "GastNarrowBPrime"
TASK_ID = "Gd-GastNarrowBPrime-Rbq10-Dreamwaq-v0"
