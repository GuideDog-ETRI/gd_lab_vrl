"""B' command: the v2.5 narrow command (narrow tiles overwritten, backward = False there) + R20 on gap tiles."""
from __future__ import annotations

import torch

from gd_lab.gast_backgap.tasks import BACK_SPEED_MAX
from gd_lab.gast_gait.commands import sample_back_speed
from gd_lab.gast_narrow.tasks_v25 import NarrowV25VelocityCommand
from gd_lab.mdp.gap_stair_v2_math import speed_ceiling


class BPrimeVelocityCommand(NarrowV25VelocityCommand):
    """Reversed gap-tile speeds redrawn from gast_gait.commands.BACK_SPEED_MIX (G5 R20); everything else v2.5."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        back = ids[self.backward[ids]]
        if not len(back):
            return
        ceiling = min(BACK_SPEED_MAX, speed_ceiling(int(self._env.common_step_counter), self.ramp_steps))
        self.vel_command_b[back, 0] = -sample_back_speed(len(back), ceiling, self.device)
