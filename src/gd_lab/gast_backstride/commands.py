"""Backward gap commands only on narrow slots (A14): the user's subway gaps are narrow (11-04 real-car clip), so
gap tiles whose widest slot exceeds BACKSTRIDE_BACK_GAP_MAX are driven forward instead of in reverse."""
from __future__ import annotations

import os

import torch

from gd_lab.gast_gait.commands import SlowBackGapVelocityCommand
from gd_lab.gast_gait.terms import _GapSlots

BACK_GAP_MAX = float(os.environ.get("BACKSTRIDE_BACK_GAP_MAX", "0"))  # m; 0 = every gap width (6566 / a1993)


def forward_on_wide_slots(command, env_ids) -> None:
    """Turn the reversed commands of env_ids whose gap tile has a slot wider than BACK_GAP_MAX forward."""
    if BACK_GAP_MAX <= 0.0:
        return
    if not hasattr(command, "_slots"):
        command._slots = _GapSlots(command._env)
    if command._slots.geometry is None:
        return
    ids = torch.as_tensor(env_ids, device=command.device, dtype=torch.long)
    slots, _ = command._slots.geometry.for_envs(command._env)
    wide = (slots[ids, :, 1] - slots[ids, :, 0]).amax(-1) > BACK_GAP_MAX + 1e-6
    flip = ids[command.backward[ids] & wide]
    if len(flip):
        command.vel_command_b[flip, 0] = command.vel_command_b[flip, 0].abs()
        command.backward[flip] = False


class NarrowBackGapVelocityCommand(SlowBackGapVelocityCommand):
    """SlowBackGapVelocityCommand; reversed commands on slots wider than BACK_GAP_MAX are turned forward."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        forward_on_wide_slots(self, env_ids)
