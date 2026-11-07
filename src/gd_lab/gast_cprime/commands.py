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

import os

import torch

from gd_lab.gast_bprime.commands import BPrimeVelocityCommand
from gd_lab.gast_cprime.cliff import cliff_yaw0
from gd_lab.gast_narrow import mdp_cliff
from gd_lab.gast_narrow.mdp import WZ_GAIN, WZ_MAX, _wrap


_DEBUG = os.environ.get("CPRIME_CMD_DEBUG", "0") == "1"
# Staged zone command (tried 11-02 in cprime_t1c after t1b it100 still walked off every deck): inside the cliff zone the
# linear command is scaled by kappa = 0 for the first ZERO_CMD_STEPS env steps (100 steps = 1 update), then ramps to 1
# over CMD_RAMP_STEPS. With a zero command the start model stands, so cliff_stand / hold were sampled from update 0
# (hold 1.00, level 0 -> 11) -- but the policy credited the standing to the zero command, not to the drop: at kappa 0.2
# (update 120) hold fell to 0.79 and at update 150 to 0.17, the same walk-off as t1 / t1b. OFF by default since then
# (0); the stand anchor of gast_cprime.anchor replaces it. CPRIME_ZERO_CMD_STEPS=<n> re-enables it for experiments.
ZERO_CMD_STEPS = int(os.environ.get("CPRIME_ZERO_CMD_STEPS", "0"))
CMD_RAMP_STEPS = int(os.environ.get("CPRIME_CMD_RAMP_STEPS", "10000"))


def zone_command_scale(step: int) -> float:
    if ZERO_CMD_STEPS <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - ZERO_CMD_STEPS) / max(1, CMD_RAMP_STEPS)))


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
        kappa = zone_command_scale(int(env.common_step_counter))
        if kappa < 1.0:
            zone = mdp_cliff.cliff_zone(env)
            scale = torch.where(zone, torch.full_like(vx, kappa), torch.ones_like(vx))
            vx, vy = vx * scale, vy * scale
        self.vel_command_b[mask, 0] = vx[mask]
        self.vel_command_b[mask, 1] = vy[mask]
        self.vel_command_b[mask, 2] = wz[mask]
        if _DEBUG and int(env.common_step_counter) % 100 == 0:
            print(f"[CPRIME_CMD] step={int(env.common_step_counter)} cliff_envs={int(mask.sum())} "
                  f"|rot|max={float(rot[mask].abs().max()):.2f} rad vy|max={float(vy[mask].abs().max()):.2f} "
                  f"wz|max={float(wz[mask].abs().max()):.2f} yaw0[:3]={cliff_yaw0(env)[mask][:3].tolist()} "
                  f"yaw[:3]={yaw[mask][:3].tolist()}", flush=True)
