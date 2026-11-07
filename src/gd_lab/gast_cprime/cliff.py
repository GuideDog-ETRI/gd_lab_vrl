"""C' cliff episodes: v2.5's reset, then the command draw of design D1 / D3 -- front / back only, 0.3..0.8 m/s -- and the
spawn heading the world-frame command (gast_cprime.commands) holds."""
from __future__ import annotations

import os

import torch

from gd_lab.gast_narrow import mdp_cliff, mdp_v2
from gd_lab.gast_narrow.state import narrow_state

DIR_P = (0.5, 0.5, 0.0, 0.0)   # front, back, left, right: the belly cameras see no drop sideways (D1)
SPEED_FB = (0.3, 0.8)          # m/s (v2.5: 0.3..0.6); indoor operation is 0.6..0.8
# t1i/t1j: the stop moves to the edge, the slow-down stays where the drop comes into view. A6000 t1h E2 (11-03): cliff
# hold 100 %, but 0.3 m/s gaps not reached 11-28 % at L14; the L14 replay stops with the base 0.6-0.7 m before the
# slot, where its far side (0.25 m wide) is still outside the 0.8 m input window (measured from the base), i.e. the
# frame equals a cliff end. margin_t1 walks on and crosses (not reached 0-3 %). Standing (cliff_stand, hold,
# should_stand, stand anchor, stall free) pays within STAND_ZONE only; within SLOW_ZONE tracking is off and slowing
# pays (cliff_slow). Both are FOOT margins (the front feet are ~0.31 m ahead of the base): local t1i (0.4 / 0.8)
# still stood with the base 0.7 m out; the far side of an L14 slot needs the base within 0.55 m = feet ~0.2 m.
# SLOW_ZONE 0.5 = base 0.8 m, where the drop first enters the window. t1h rule: 0.6 / 0.6 (+ stall 0.8).
STAND_ZONE = float(os.environ.get("CPRIME_STAND_ZONE", "0.2"))   # m, front-foot margin to the edge
SLOW_ZONE = float(os.environ.get("CPRIME_SLOW_ZONE", "0.5"))     # m, front-foot margin to the edge
mdp_cliff.ZONE = STAND_ZONE  # all v2.5 zone users (hold, cliff_stand, should_stand, tracking off) read the module value


def cliff_yaw0(env) -> torch.Tensor:
    """[N] spawn heading of the current cliff episode (the command's world-frame reference)."""
    yaw0 = getattr(env, "_cprime_yaw0", None)
    if yaw0 is None:
        yaw0 = torch.zeros(env.num_envs, device=env.device)
        env._cprime_yaw0 = yaw0
    return yaw0


def reset_cliff_cprime(env, env_ids):
    """mdp_cliff.reset_cliff (spawn, trackers, v2 blocked off), then the C' direction / speed draw on cliff tiles."""
    mdp_cliff.reset_cliff(env, env_ids)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    ids = ids[narrow_state(env).kind[ids] == mdp_cliff.CLIFF]
    if not len(ids):
        return
    n, d = len(ids), env.device
    cliff = mdp_cliff.cliff_episode(env)
    p = torch.tensor(DIR_P, device=d)
    cliff.direction[ids] = torch.multinomial(p.expand(n, -1), 1).squeeze(-1)
    cliff.speed[ids] = SPEED_FB[0] + (SPEED_FB[1] - SPEED_FB[0]) * torch.rand(n, device=d)
    cliff_yaw0(env)[ids] = env.scene["robot"].data.heading_w[ids]  # pose already written by reset_cliff


def slow_zone(env) -> torch.Tensor:
    """[N] bool: on a cliff tile within SLOW_ZONE of the edge, not breached / fallen (tracking off, slowing pays)."""
    cliff = mdp_cliff.cliff_episode(env)
    monitor = mdp_v2._monitor(env)
    return mdp_cliff._cliff_mask(env) & (cliff.e < SLOW_ZONE) & ~monitor.breached & ~cliff.fell


def track_lin_vel_xy_cprime(env, command_name: str, std_base: float = 0.3, alpha: float = 0.2,
                            cmd_threshold: float = 0.1, vel_frac_low: float = 0.4, vel_frac_high: float = 0.8,
                            asset_cfg=None):
    """v2.5 tracking (off in the stand zone), also off in the slow zone (same parameters as track_lin_vel_xy_cliff)."""
    from gd_lab.gast_narrow import tasks_v25
    track = tasks_v25.track_lin_vel_xy_cliff(env, command_name, std_base=std_base, alpha=alpha,
                                             cmd_threshold=cmd_threshold, vel_frac_low=vel_frac_low,
                                             vel_frac_high=vel_frac_high, asset_cfg=asset_cfg)
    return track * (~slow_zone(env)).float()
