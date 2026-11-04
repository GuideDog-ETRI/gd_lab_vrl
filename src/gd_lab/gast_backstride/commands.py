"""Backward gap commands only on narrow slots (A14): the user's subway gaps are narrow (11-04 real-car clip), so
gap tiles whose widest slot exceeds BACKSTRIDE_BACK_GAP_MAX are driven forward instead of in reverse."""
from __future__ import annotations

import os

import torch

from gd_lab.gast_gait.commands import SlowBackGapVelocityCommand
from gd_lab.gast_gait.terms import _GapSlots

BACK_GAP_MAX = float(os.environ.get("BACKSTRIDE_BACK_GAP_MAX", "0"))  # m; 0 = every gap width (6566 / a1993)


class NarrowBackGapVelocityCommand(SlowBackGapVelocityCommand):
    """SlowBackGapVelocityCommand; reversed commands on slots wider than BACK_GAP_MAX are turned forward."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if BACK_GAP_MAX <= 0.0:
            return
        if not hasattr(self, "_slots"):
            self._slots = _GapSlots(self._env)
        if self._slots.geometry is None:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        slots, _ = self._slots.geometry.for_envs(self._env)
        wide = (slots[ids, :, 1] - slots[ids, :, 0]).amax(-1) > BACK_GAP_MAX + 1e-6
        flip = ids[self.backward[ids] & wide]
        if len(flip):
            self.vel_command_b[flip, 0] = self.vel_command_b[flip, 0].abs()
            self.backward[flip] = False
