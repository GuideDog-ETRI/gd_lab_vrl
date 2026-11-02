"""C' command: on cliff tiles the velocity command is fixed in the WORLD frame (the spawn heading), not the body frame.

v2.5 gave a constant body-frame command toward one side with wz = 0. A policy that already avoids drops (margin_t1 /
bpfin, trained on gaps with R17 / foothold terms) found the loophole on 11-02 (replay c1_cliff_bpfin: 71 % of the deck
episodes turn more than 90 deg, 0.03 s still inside the zone, no breach, no fall): turn away from the edge and keep
walking "forward" in the body frame across the deck -- full linear tracking, no risk, never inside the zone standing
still. The cliff reward (cliff_stand +1 /s, hold +4) was never sampled (cliff_stand 0 for 57 updates, like v25b).

Here the commanded direction is the spawn heading rotated into the current body frame, so after a turn the command
points back at the edge (walking away is penalised by the tracking term), and wz holds the spawn heading like the
passage command does (gain WZ_GAIN, clamp WZ_MAX). The only ways to score are to stop before the edge or to fall.
"""
from __future__ import annotations

import torch

from gd_lab.gast_bprime.commands import BPrimeVelocityCommand
from gd_lab.gast_cprime.cliff import cliff_yaw0
from gd_lab.gast_narrow import mdp_cliff
from gd_lab.gast_narrow.mdp import WZ_GAIN, WZ_MAX, _wrap


class CPrimeVelocityCommand(BPrimeVelocityCommand):
    def _apply_cliff(self):
        super()._apply_cliff()
        env = self._env
        mask = mdp_cliff._cliff_mask(env)
        if not bool(mask.any()):
            return
        cliff = mdp_cliff.cliff_episode(env)
        yaw = env.scene["robot"].data.heading_w
        rot = _wrap(cliff_yaw0(env) - yaw)                      # spawn frame -> current body frame
        v0 = cliff.dir_vec[cliff.direction] * cliff.speed[:, None]
        c, s = torch.cos(rot), torch.sin(rot)
        vx = c * v0[:, 0] - s * v0[:, 1]
        vy = s * v0[:, 0] + c * v0[:, 1]
        wz = (WZ_GAIN * rot).clamp(-WZ_MAX, WZ_MAX)
        self.vel_command_b[mask, 0] = vx[mask]
        self.vel_command_b[mask, 1] = vy[mask]
        self.vel_command_b[mask, 2] = wz[mask]
