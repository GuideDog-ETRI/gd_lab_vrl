"""C' actor input ``support_end`` [N, 1]: how close, along the command direction, the supporting ground ends.

Read off the policy's own terrain input (the 11 x 17 Depth+IR frame the actor already gets, after visibility, noise and
blackout), so the robot's map can produce the same value (design section 2 / 2.1). Grid: index = iy * 17 + ix,
x_body = -0.8 + 0.1 ix (ix 8 under the trunk), y_body = -0.5 + 0.1 iy (iy 5 = centre line); heights are
(scanner_z - z - 0.5) x 5, 0 where the cell is not seen.

Along the centre line outward from the trunk (8 cells, 0.1..0.8 m) a cell SUPPORTS when it is seen and not more than
DROP_JUMP below the last supporting cell (the trunk floor to start): stair steps (<= 0.2 m per step) and a seen far
edge of a gap count as support, a drop seen as the deck-to-floor jump (cliff 1.0 m, gap pit 0.65 m) or an unseen
cell does not. With the first non-supporting cell at distance d and no supporting cell beyond it in the window,
support_end = (0.8 - d) / 0.8 (0.875 one cell ahead, 0 at the window end); 0 when the line is all support, when a
supporting cell lies beyond the break (a gap whose far edge is seen), when the command is lateral or zero (the belly
cameras see nothing sideways, design D1) and when the frame is blacked out.

Measured on the v2.5 decks (replay 11-02, floor tolerance 0.15): on within 0.6 m of a cliff edge 96-100 %, on flat
deck 1-11 %, in front of a 0.26 m gap 22 % (0-0.3 m) / 45 % (0.3-0.6 m). An input, not a stop gate: the policy still
separates the gap (far edge, 0.65 m pit) from the cliff.
"""
from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase

CELLS = 187
NX, NY = 17, 11
CELL, REACH = 0.1, 0.8
DROP_JUMP = 0.35      # m: a cell this much below the last support is a drop (stairs <= 0.2 per step)
CMD_MIN = 0.05        # m/s: below this the command has no direction
CENTRE_ROW = 5


def support_end_from_line(h_line, valid_line, dist, reach: float = REACH, drop_jump: float = DROP_JUMP):
    """h_line [N, K] heights (m, positive = ground lower), valid_line [N, K] seen, dist [K] metres from the trunk;
    the cells ordered outward. Returns [N] in [0, 1]."""
    n, k = h_line.shape
    ref = torch.zeros(n, device=h_line.device, dtype=h_line.dtype)
    support = []
    for i in range(k):
        s = valid_line[:, i] & (h_line[:, i] - ref <= drop_jump)
        ref = torch.where(s, h_line[:, i], ref)
        support.append(s)
    support = torch.stack(support, 1)
    broken = ~support
    first = broken.float().argmax(1)                                   # first non-supporting cell (0 if none)
    order = torch.arange(k, device=h_line.device)
    beyond = (support & (order[None] > first[:, None])).any(1)       # support again past the break: a gap, not an end
    value = ((reach - dist[first]) / reach).clamp(0.0, 1.0)
    return torch.where(broken.any(1) & ~beyond, value, torch.zeros_like(value))


class SupportEnd(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        d = env.device
        self.fwd = torch.tensor([CENTRE_ROW * NX + ix for ix in range(9, 17)], device=d)
        self.back = torch.tensor([CENTRE_ROW * NX + ix for ix in range(7, -1, -1)], device=d)
        self.dist = torch.arange(1, 9, device=d, dtype=torch.float32) * CELL
        self.last = torch.zeros(env.num_envs, device=d)

    def __call__(self, env):
        terrain = env._gast_terrain_term(env)                          # the actor's frame (cached within the step)
        h, valid = terrain[:, :CELLS] / 5.0, terrain[:, CELLS:2 * CELLS] > 0.5
        command = env.command_manager.get_command("base_velocity")
        vx, vy = command[:, 0], command[:, 1]
        axial = (vx.abs() >= CMD_MIN) & (vx.abs() >= vy.abs()) & valid.any(1)
        line = torch.where((vx >= 0)[:, None], self.fwd[None], self.back[None])
        value = support_end_from_line(torch.gather(h, 1, line), torch.gather(valid, 1, line), self.dist)
        self.last = torch.where(axial, value, torch.zeros_like(value))
        return self.last[:, None]
