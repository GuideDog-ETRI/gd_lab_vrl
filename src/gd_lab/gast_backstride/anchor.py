"""Retention anchor toward the start model (A') on every sample that is not a backward gap step (any gap step with
BACKSTRIDE_BOTH_DIR=1, A18)."""
from __future__ import annotations

import torch

from gd_lab.gast_backstride import terms
from gd_lab.gast_narrow.anchor import NarrowAnchorPPO


class BackStrideAnchorPPO(NarrowAnchorPPO):
    """NarrowAnchorPPO with the retention mask = not (gap tile and moving backward). Stand anchor off
    (NARROW_STAND_ANCHOR_WEIGHT=0 in the launcher)."""

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
            raise RuntimeError("backstride retention: no back gap mask recorded (stance_spacing_floor term missing?)")
        return ~buf.flatten(0, 1)  # [T, N] -> T-major, as storage.observations.flatten(0, 1)

    def _stand_samples(self, obs_all):
        return None
