"""Gait v3 R20: backward gap commands drawn toward slow speeds.

The 6566 command (gast_backgap.BackGapVelocityCommand) draws the reversed gap-tile speed from the v2.1 forward mix
capped at 1.0 m/s: about 6 % of the backward gap commands are 0.2-0.3 m/s and 60 % are 0.8-1.0 m/s. G4 model_299
stood still in front of the 10 cm and 15 cm backward gaps at -0.3 m/s in MuJoCo (11-01, no foot lift for > 100 s)
and crossed all five gaps at -0.5 m/s; in Isaac it stalled 4.7 % of the -0.3 m/s command time (G3 0.1 %, 6566 0.8 %).
This class redraws every reversed speed from BACK_SPEED_MIX instead (same ceiling ramp); forward commands and the
backward share stay as in 6566.
"""
from __future__ import annotations

import torch

from gd_lab.gast_backgap.tasks import BACK_SPEED_MAX, BackGapVelocityCommand
from gd_lab.mdp.gap_stair_v2_math import speed_ceiling

# (probability, low, high) m/s: 40 % slow 0.2-0.5, 35 % 0.5-0.8, 25 % 0.8-1.0
BACK_SPEED_MIX = ((0.40, 0.2, 0.5), (0.35, 0.5, 0.8), (0.25, 0.8, 1.0))


def sample_back_speed(n: int, ceiling: float, device=None) -> torch.Tensor:
    """[n] backward speeds (positive) from BACK_SPEED_MIX; draws above ``ceiling`` are redrawn in [0.2, ceiling]."""
    u = torch.rand(n, device=device)
    out = torch.empty(n, device=device)
    edge = 0.0
    for p, lo, hi in BACK_SPEED_MIX:
        pick = (u >= edge) & (u < edge + p)
        out[pick] = lo + (hi - lo) * torch.rand(int(pick.sum()), device=device)
        edge += p
    low = BACK_SPEED_MIX[0][1]
    fresh = low + (max(ceiling, low) - low) * torch.rand(n, device=device)
    return torch.where(out > ceiling, fresh, out)


class SlowBackGapVelocityCommand(BackGapVelocityCommand):
    """6566 boarding command; reversed gap-tile speeds from BACK_SPEED_MIX."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        back = ids[self.backward[ids]]
        if not len(back):
            return
        ceiling = min(BACK_SPEED_MAX, speed_ceiling(int(self._env.common_step_counter), self.ramp_steps))
        self.vel_command_b[back, 0] = -sample_back_speed(len(back), ceiling, self.device)
