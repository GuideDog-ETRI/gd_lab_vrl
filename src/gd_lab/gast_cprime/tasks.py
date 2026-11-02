"""Stage C' task: B' + the v2.5 cliff decks (front / back, 0.3..0.8 m/s) + the support_end actor input.

Design: docs/design/GAST_NARROW_V26_CLIFF_DESIGN.md. Launcher: NARROW_TURN=0 NARROW_CLIFF_COLUMNS=4 (44 + 4 columns).
Everything else is B' (gast_bprime.tasks): gait v3 G5 terms, LiDAR-only passages, balanced margin draws, R20. The
v2.5 cliff machinery (mdp_cliff: zone, cliff_stand, hold / breach / fall, base_height over the void, stall / centering /
tracking off, own curriculum, critic slots 7 / 8) is inherited through B'; the B' floor handling of the terrain input
skips the decks (gast_bprime.terrain_obs.passage_mask) while R1 / R6 / R15 keep the deck top as the floor (the deck is
flat and the env origin sits on it).
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from gd_lab.gast_bprime.commands import BPrimeVelocityCommand
from gd_lab.gast_bprime.tasks import GastNarrowBPrimeRunnerCfg, GastNarrowBPrimeTeacherCfg
from gd_lab.gast_cprime.cliff import reset_cliff_cprime
from gd_lab.gast_cprime.commands import CPrimeVelocityCommand
from gd_lab.gast_cprime.model import SUPPORT_GROUP
from gd_lab.gast_cprime.support import SupportEnd
from gd_lab.gast_narrow import mdp_cliff, tasks_v25

CLIFF_COLUMNS = 4


@configclass
class NarrowSupportEndGroup(ObservationGroupCfg):
    """Policy input [N, 1]: gast_cprime.support.SupportEnd."""
    support_end = ObservationTermCfg(func=SupportEnd)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GastNarrowCPrimeTeacherCfg(GastNarrowBPrimeTeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        if tasks_v25.TURN_ON or int(tasks_v25.CLIFF_COLUMNS) != CLIFF_COLUMNS:
            raise ValueError(f"stage C' needs NARROW_TURN=0 NARROW_CLIFF_COLUMNS={CLIFF_COLUMNS} "
                             f"(got turn={tasks_v25.TURN_ON}, cliff columns={tasks_v25.CLIFF_COLUMNS})")
        assert "narrow_cliff" in self.scene.terrain.terrain_generator.sub_terrains
        assert self.events.reset_cliff.func is mdp_cliff.reset_cliff
        self.events.reset_cliff.func = reset_cliff_cprime
        assert self.commands.base_velocity.class_type is BPrimeVelocityCommand
        self.commands.base_velocity.class_type = CPrimeVelocityCommand  # world-frame cliff command (11-02 loophole)
        setattr(self.observations, SUPPORT_GROUP, NarrowSupportEndGroup())


@configclass
class GastNarrowCPrimeRunnerCfg(GastNarrowBPrimeRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/lineage16_cprime"
        self.policy.class_name = "gd_lab.gast_cprime.model:GastNarrowCPrimeActorCritic"
        self.algorithm.symmetry_cfg.data_augmentation_func = "gd_lab.gast_cprime.model:mirror_narrow_cprime"


TASK = "GastNarrowCPrime"
TASK_ID = "Gd-GastNarrowCPrime-Rbq10-Dreamwaq-v0"
