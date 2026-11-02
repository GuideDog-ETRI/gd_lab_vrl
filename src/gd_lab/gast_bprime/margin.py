"""Stage B' (11-02): balanced pass / block draws at the widths where only the clearance margin decides.

mdp_v2.draw_margin draws m uniformly in MARGIN_RANGE and derives blocked = w < ROBOT_WIDTH + 2 m. At w <= 0.62 m every
draw is blocked (the robot cannot pass with m >= 0.05 + BAND), at w = 0.66 m 60 % are, at w = 0.72 m 20 %. The B'
continuation (bprime_cont, it300-599) learned the stop from the width alone: the margin sweep E2 (A6000, 11-02) has
cont_fin passing 17.5 % at w 0.66 / m 0.05 (602: 98.1) and 92.5 % at w 0.72 / m 0.05, while blocked cells hold
96 / 69 %. In the band w in [W_LO, W_HI) both outcomes are possible for some m; there the draw is made 50 / 50 blocked /
passable first and m second, inside the matching sub-range (still outside the +/-BAND dead zone). Dead ends and widths
outside the band keep the v2 draw. Disable with BPRIME_MARGIN_BALANCE=0.
"""
from __future__ import annotations

import os

import torch

from gd_lab.gast_narrow import geometry, mdp_v2
from gd_lab.gast_narrow.mdp_v2 import BAND, MARGIN_RANGE, ROBOT_WIDTH
from gd_lab.gast_narrow.state import narrow_state

ENABLED = os.environ.get("BPRIME_MARGIN_BALANCE", "1") == "1"
P_BLOCKED = 0.5
# passable needs m <= (w - BAND - ROBOT_WIDTH) / 2, blocked needs m >= (w + BAND - ROBOT_WIDTH) / 2, both inside MARGIN_RANGE
W_LO = ROBOT_WIDTH + 2 * MARGIN_RANGE[0] + BAND   # 0.62: below it nothing passes
W_HI = ROBOT_WIDTH + 2 * MARGIN_RANGE[1] - BAND   # 0.74: from it on nothing is blocked


def balanced_margin(width, dead_end, rand=torch.rand):
    """width [n], dead_end [n] bool -> (margin [n], blocked [n]); v2 draw outside the band."""
    m, blocked = mdp_v2.draw_margin(width, dead_end, rand)
    band = ~dead_end & (width >= W_LO) & (width < W_HI)
    if not bool(band.any()):
        return m, blocked
    lo, hi = MARGIN_RANGE
    m_pass_hi = ((width - BAND - ROBOT_WIDTH) / 2).clamp(lo, hi)     # passable: m in [lo, m_pass_hi]
    m_block_lo = ((width + BAND - ROBOT_WIDTH) / 2).clamp(lo, hi)    # blocked:  m in [m_block_lo, hi]
    pick_block = rand(width.shape, device=width.device) < P_BLOCKED
    u = rand(width.shape, device=width.device)
    m_new = torch.where(pick_block, m_block_lo + (hi - m_block_lo) * u, lo + (m_pass_hi - lo) * u)
    m = torch.where(band, m_new, m)
    blocked = dead_end | (width < ROBOT_WIDTH + 2 * m)
    return m, blocked


def reset_narrow_bprime(env, env_ids):
    """reset_narrow_v2, then the balanced redraw on band-width passage tiles (dead ends, turn / cliff tiles untouched)."""
    mdp_v2.reset_narrow_v2(env, env_ids)
    if not ENABLED:
        return
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    state = narrow_state(env)
    episode = mdp_v2.narrow_episode(env)
    passage = state.narrow[ids] & (state.kind[ids] != geometry.KINDS["turn"]) & (state.kind[ids] != geometry.KINDS["cliff"])
    ids = ids[passage]
    if not len(ids):
        return
    width = state.width[ids]
    dead_end = state.kind[ids] == geometry.KINDS["dead_end"]
    m, blocked = balanced_margin(width, dead_end)
    episode.margin[ids] = m
    episode.blocked[ids] = blocked
    episode.slack[ids] = width - (ROBOT_WIDTH + 2 * m)
