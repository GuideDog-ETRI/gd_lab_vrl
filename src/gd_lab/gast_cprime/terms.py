"""C' reward terms."""
from __future__ import annotations

import torch

from gd_lab.gast_cprime import anchor
from gd_lab.gast_cprime.cliff import slow_zone
from gd_lab.gast_narrow import mdp_cliff

V_REF = 0.6   # m/s: zone entry speeds are 0.3..0.8, so slowing down pays from the first step in the zone


def cliff_slow(env, v_ref: float = V_REF):
    """Per second in the slow zone (gast_cprime.cliff.SLOW_ZONE): 1 - speed / v_ref (clamped to [0, 1]). v2.5's
    cliff_stand kernel exp(-(v/0.1)^2) is ~0 above 0.25 m/s, so a policy that walks into the zone at 0.3..0.8 m/s gets
    no gradient toward slowing down and only meets the breach (-4): cprime_t1b it100 walked off every deck in 3.9 s
    exactly like the untrained start model. This term pays for every bit of deceleration; cliff_stand and the hold
    bonus still pay for standing still (t1i: in the stand zone only; standing between the stand and the slow zone pays
    the stall cost, so creeping on wins there)."""
    speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
    # The C' stand anchor (gast_cprime.anchor) pulls exactly the stand-zone samples; the retention anchor skips the
    # cliff decks.
    anchor.set_cliff_zone_mask(mdp_cliff.cliff_zone(env), mdp_cliff._cliff_mask(env))
    return (1.0 - speed / v_ref).clamp(0.0, 1.0) * slow_zone(env).float()
