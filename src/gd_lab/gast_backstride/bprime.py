"""Stage B' + backstride (11-05): the B' passage task (gast_bprime.tasks, unchanged) + the A18L gap terms.

B' (fc7d4e9) composes apply_gait_v3, whose R17 (gap_support: three feet down near a slot) brings the three-point
support back over 2,000 updates. Here the backstride gap terms are added on top (apply_backstride; the launcher sets
the A18L switches: BOTH_DIR=1, R17 off both ways, diagonal trot, step floor 0.50 m, landing margin, backward gaps
<= 0.20 m). Retention anchor: B' (off the narrow tiles) minus the gap steps (BOTH_DIR: any gap step); stand anchor
as B'.

Two stages (11-06): with the B' margin draw (50/50 blocked / pass at 0.62-0.74 m) and the B' stall penalty
(1.0 s, -2) from update 0 the policy never learned to stand (A6000 bprime_stride_cont: stand 1-2.5 %; Hangang job
1222681 cancelled). B' itself had them only after stand was learned. BPRIME_STRIDE_STAGE=1: margin draw off
(BPRIME_MARGIN_BALANCE=0, checked here) and the v2.5 stall (1.5 s, -1) -- stand is learned (A6000 v22stall: 92 %);
BPRIME_STRIDE_STAGE=2 (default): B' as is, from the stage 1 model (A6000 margin: stand 99 %).
"""
from __future__ import annotations

import os

import torch
from isaaclab.utils import configclass

from gd_lab.gast_backstride import terms
from gd_lab.gast_backstride.commands import forward_on_wide_slots
from gd_lab.gast_backstride.tasks import apply_backstride
from gd_lab.gast_bprime import margin
from gd_lab.gast_bprime.commands import BPrimeVelocityCommand
from gd_lab.gast_bprime.tasks import GastNarrowBPrimeRunnerCfg, GastNarrowBPrimeTeacherCfg
from gd_lab.gast_narrow.anchor import NarrowAnchorPPO
from gd_lab.gast_narrow.tasks_v25 import NarrowStallPenaltyV25

STAGE = os.environ.get("BPRIME_STRIDE_STAGE", "2")


class BPrimeNarrowBackGapVelocityCommand(BPrimeVelocityCommand):
    """B' command; reversed gap commands on slots wider than BACKSTRIDE_BACK_GAP_MAX are turned forward."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        forward_on_wide_slots(self, env_ids)


class BPrimeStrideAnchorPPO(NarrowAnchorPPO):
    """NarrowAnchorPPO (retention off the narrow tiles, stand anchor) with the gap steps also left out of retention."""

    def process_env_step(self, obs, rewards, dones, extras):
        mask = terms.back_gap_step_mask()
        if mask is not None:
            buf = getattr(self, "_back_gap_buf", None)
            if buf is None:
                buf = self._back_gap_buf = torch.zeros(self.storage.num_transitions_per_env, mask.shape[0],
                                                       dtype=torch.bool, device=self.device)
            buf[self.storage.step] = mask.to(self.device)  # before super(): storage.step = this transition
        return super().process_env_step(obs, rewards, dones, extras)

    def _retention_samples(self, obs_all):
        buf = getattr(self, "_back_gap_buf", None)
        if buf is None:
            raise RuntimeError("bprime stride retention: no gap mask recorded (stance_spacing_floor term missing?)")
        return super()._retention_samples(obs_all) & ~buf.flatten(0, 1)


@configclass
class GastNarrowBPrimeStrideTeacherCfg(GastNarrowBPrimeTeacherCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_backstride(self, command_cls=BPrimeNarrowBackGapVelocityCommand)
        if STAGE == "1":
            assert not margin.ENABLED, "BPRIME_STRIDE_STAGE=1 needs BPRIME_MARGIN_BALANCE=0"
            self.rewards.narrow_stall.func = NarrowStallPenaltyV25
            self.rewards.narrow_stall.weight = -1.0
        elif STAGE != "2":
            raise ValueError(f"BPRIME_STRIDE_STAGE must be 1 or 2, got {STAGE!r}")


@configclass
class GastNarrowBPrimeStrideRunnerCfg(GastNarrowBPrimeRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "gast/bprime_stride"
        self.algorithm.class_name = "gd_lab.gast_backstride.bprime:BPrimeStrideAnchorPPO"


TASK = "GastNarrowBPrimeStride"
TASK_ID = "Gd-GastNarrowBPrimeStride-Rbq10-Dreamwaq-v0"
