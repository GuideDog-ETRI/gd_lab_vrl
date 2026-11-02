"""C' cliff episodes: v2.5's reset, then the command draw of design D1 / D3 -- front / back only, 0.3..0.8 m/s -- and the
spawn heading the world-frame command (gast_cprime.commands) holds."""
from __future__ import annotations

import torch

from gd_lab.gast_narrow import mdp_cliff
from gd_lab.gast_narrow.state import narrow_state

DIR_P = (0.5, 0.5, 0.0, 0.0)   # front, back, left, right: the belly cameras see no drop sideways (D1)
SPEED_FB = (0.3, 0.8)          # m/s (v2.5: 0.3..0.6); indoor operation is 0.6..0.8


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
