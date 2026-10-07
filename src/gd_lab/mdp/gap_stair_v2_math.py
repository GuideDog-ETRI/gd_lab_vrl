"""Pure tensor kernels for the v2 gap/stair teacher rewards (no simulator imports; CPU-testable).

Frames: ``x``/``y`` are relative to the tile centre (== env origin); ``z`` is world height.
A foot position is the collision-sphere centre (radius ``FOOT_RADIUS``).
"""

from __future__ import annotations

import math

import torch

FOOT_RADIUS = 0.03


def gap_edge_margin_cost(foot_x: torch.Tensor, slots: torch.Tensor, margin: float) -> torch.Tensor:
    """[N, F] cost in [0, 1] for a foot at ``foot_x`` [N, F] near any edge of ``slots`` [N, S, 2] (lo, hi).

    Both slot edges are drop-offs. A foot centre inside a slot costs 1; on a deck it costs
    ``(margin - d) / margin`` with ``d`` the distance to the nearest slot edge.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    x = foot_x[..., None]  # [N, F, 1]
    lo, hi = slots[:, None, :, 0], slots[:, None, :, 1]  # [N, 1, S]
    inside = ((x > lo) & (x < hi)).any(-1)
    distance = torch.minimum((x - lo).abs(), (x - hi).abs()).amin(-1)
    cost = ((margin - distance) / margin).clamp(0, 1)
    return torch.where(inside, torch.ones_like(cost), cost)


def stair_nose_margin_cost(cheb: torch.Tensor, edges: torch.Tensor, inverted: torch.Tensor,
                           margin: float) -> torch.Tensor:
    """[N, F] cost for feet at Chebyshev radius ``cheb`` [N, F] on pyramid stairs.

    ``edges`` [N, K]: step-edge radii (NaN padded); ``inverted`` [N] bool. Only the drop-off side
    of an edge (the tread just above the nose) is dangerous: on a pyramid (centre high) that is
    ``edge - margin < r <= edge``; on an inverted pyramid (centre low) ``edge <= r < edge + margin``.
    Landing at the foot of a riser is not penalised.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    r = cheb[..., None]  # [N, F, 1]
    e = edges[:, None, :]  # [N, 1, K]
    onto = torch.where(inverted[:, None, None], r - e, e - r)  # distance from the nose onto the upper tread
    valid = torch.isfinite(e) & (onto >= 0) & (onto < margin)
    cost = torch.where(valid, (margin - onto) / margin, torch.zeros_like(onto))
    return cost.amax(-1)


def pyramid_step_edges(size: float, border_width: float, platform_width: float, step_width: float) -> list[float]:
    """Edge radii of IsaacLab-style pyramid stairs (and gd_lab nosing stairs), outermost first.

    Mirrors ``num_steps = (size - 2 border - platform) // (2 step) + 1``; step ``k`` spans the
    square ring whose outer half-size is ``size/2 - border - k * step``; the platform edge closes it.
    """
    num_steps = int((size - 2 * border_width - platform_width) // (2 * step_width) + 1)
    outer = size / 2 - border_width
    return [outer - k * step_width for k in range(num_steps + 1)]


def nosed_edges(edges: list[float], nose_depth: float, inverted: bool) -> list[float]:
    """Drop-off radii with nosing lips (gd_lab nosing stairs): every pyramid lip protrudes outward by
    ``nose_depth``; inverted lips protrude inward at every edge except the outermost (no lip there)."""
    if not nose_depth:
        return list(edges)
    if inverted:
        return [edges[0]] + [e - nose_depth for e in edges[1:]]
    return [e + nose_depth for e in edges]


def slot_low_or_contact_cost(foot_x: torch.Tensor, foot_z: torch.Tensor, contact: torch.Tensor, slots: torch.Tensor,
                             upper_decks: torch.Tensor, clearance: float = 0.02) -> torch.Tensor:
    """[N] mean over feet: the foot is over a slot (+- radius) and either touching or lower than the
    HIGHER adjacent deck + radius + clearance. Catches shallow probing, edge scraping and dipping."""
    x = foot_x[..., None]
    lo, hi = slots[:, None, :, 0] - FOOT_RADIUS, slots[:, None, :, 1] + FOOT_RADIUS
    over = (x > lo) & (x < hi)  # [N, F, S]
    low = foot_z[..., None] < upper_decks[:, None, :] + FOOT_RADIUS + clearance
    bad = (over & (low | contact[..., None])).any(-1)
    return bad.float().mean(-1)


def chebyshev_frame(xy: torch.Tensor):
    """Chebyshev radius [N] and the unit outward direction [N, 2] of the square ring through ``xy`` [N, 2]."""
    ax, ay = xy[:, 0].abs(), xy[:, 1].abs()
    along_x = ax >= ay
    outward = torch.zeros_like(xy)
    outward[:, 0] = torch.where(along_x, torch.sign(xy[:, 0]), 0.0)
    outward[:, 1] = torch.where(along_x, 0.0, torch.sign(xy[:, 1]))
    outward = torch.where(outward.abs().sum(-1, keepdim=True) > 0, outward, torch.tensor([1.0, 0.0], device=xy.device))
    return torch.maximum(ax, ay), outward


def stair_push_modes(uphill: torch.Tensor, heading: torch.Tensor, velocity: torch.Tensor, command_vx: torch.Tensor,
                     min_command: float = 0.15, min_speed: float = 0.1, max_angle_deg: float = 45.0):
    """(ascending, descending) [N] bool: walking FORWARD (command and body heading) up or down the slope.

    ``uphill``/``heading``/``velocity`` are horizontal unit/xy vectors [N, 2]."""
    cos_max = math.cos(math.radians(max_angle_deg))
    along = (heading * uphill).sum(-1)
    speed = (velocity * uphill).sum(-1)
    forward = command_vx > min_command
    ascending = forward & (along > cos_max) & (speed > min_speed)
    descending = forward & (along < -cos_max) & (speed < -min_speed)
    return ascending, descending


def downhill_force(uphill: torch.Tensor, magnitude: torch.Tensor, below_horizontal_rad: torch.Tensor) -> torch.Tensor:
    """[N, 3] world force pointing downhill (``-uphill``), tilted ``below_horizontal_rad`` under the horizon.

    Ascending: a person below pulls the hip handle back AND down (nose-up moment, front feet lift).
    Descending: a push from behind is downhill too, sampled from slightly upward to downward."""
    down = -uphill
    c, s = torch.cos(below_horizontal_rad), torch.sin(below_horizontal_rad)
    force = torch.stack((down[:, 0] * c, down[:, 1] * c, -s), -1)
    return force * magnitude[:, None]


def front_lift_cost(front_contact: torch.Tensor, nose_up_rate: torch.Tensor, rate_threshold: float = 1.0) -> torch.Tensor:
    """[N] in [0, 1]: both front feet airborne (trot never does this), plus nose-up pitch rate above
    ``rate_threshold`` rad/s (excess, capped)."""
    both_air = (~front_contact).all(-1).float()
    excess = (nose_up_rate - rate_threshold).clamp(min=0)
    return (both_air + excess).clamp(max=1.0)


# --- v2.1 speeds (used by gd_lab.mdp.gap_stair_v21) ---------------------------------------------------
# (probability, low, high) m/s: usual 0.8-1.0, a slow tail and a fast tail up to the 1.2 m/s maximum.
V21_SPEED_MIX = ((0.25, 0.2, 0.8), (0.60, 0.8, 1.0), (0.15, 1.0, 1.2))
V21_SPEED_FLOOR_CEILING = (0.6, 1.2)


def sample_v21_speed(n: int, ceiling: float, device=None, generator=None) -> torch.Tensor:
    """[n] forward speeds from V21_SPEED_MIX, clipped to the current ceiling (pure torch)."""
    u = torch.rand(n, device=device, generator=generator)
    out = torch.empty(n, device=device)
    edge = 0.0
    for p, lo, hi in V21_SPEED_MIX:
        pick = (u >= edge) & (u < edge + p)
        out[pick] = lo + (hi - lo) * torch.rand(int(pick.sum()), device=device, generator=generator)
        edge += p
    return out.clamp(max=ceiling)


def speed_ceiling(step: int, ramp_steps: int) -> float:
    lo, hi = V21_SPEED_FLOOR_CEILING
    if ramp_steps <= 0:
        return hi
    return lo + (hi - lo) * min(1.0, step / ramp_steps)
