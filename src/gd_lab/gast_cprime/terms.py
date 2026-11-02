"""C' reward terms."""
from __future__ import annotations

import torch

from gd_lab.gast_cprime import anchor
from gd_lab.gast_narrow import mdp_cliff

V_REF = 0.6   # m/s: zone entry speeds are 0.3..0.8, so slowing down pays from the first step in the zone


def cliff_slow(env, v_ref: float = V_REF):
    """Per second in the cliff zone: 1 - speed / v_ref (clamped to [0, 1]). v2.5's cliff_stand kernel exp(-(v/0.1)^2)
    is ~0 above 0.25 m/s, so a policy that walks into the zone at 0.3..0.8 m/s gets no gradient toward slowing down and
    only meets the breach (-4): cprime_t1b it100 walked off every deck in 3.9 s exactly like the untrained start model.
    This term pays for every bit of deceleration; cliff_stand and the hold bonus still pay for standing still."""
    speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
    zone = mdp_cliff.cliff_zone(env)
    anchor.set_cliff_zone_mask(zone)  # the C' stand anchor (gast_cprime.anchor) pulls exactly these samples
    return (1.0 - speed / v_ref).clamp(0.0, 1.0) * zone.float()
