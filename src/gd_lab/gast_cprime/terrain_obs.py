"""C' terrain input: the B' term (NarrowFloorRaycastIrTerrain) with no camera blackout on the cliff decks.

Why (t1g, 11-03): the blackout schedule (bivt: start 0.1 %/step, 1-6 s; 5 % of the episodes blind from the reset;
~20 % of the walking frames) left 38.5 % of the cliff-zone frames without any terrain input. A blind robot cannot see a
cliff, yet walking on blind paid the breach: the policy learned "stand when blind" and stood on flat ground too (t1g
150: 10-20 % of drop-free episodes, 47 % of those stops right after a blackout; margin_t1 0 %). Decision A (user,
11-03): on a camera loss the deploy safety layer slows / stops the robot; the policy keeps walking blind elsewhere and
learns the cliff stop only from frames that show the edge. CPRIME_CLIFF_BLACKOUT=1 restores the schedule on the decks.
"""
from __future__ import annotations

import os

from gd_lab.gast_bprime.terrain_obs import NarrowFloorRaycastIrTerrain
from gd_lab.mdp.gap_stair_v2 import _column
from gd_lab.mdp.terrain_families import family_column_masks

CLIFF_FAMILY = "narrow_cliff"


def cliff_blackout_on() -> bool:
    return os.environ.get("CPRIME_CLIFF_BLACKOUT", "0") == "1"


class CPrimeRaycastIrTerrain(NarrowFloorRaycastIrTerrain):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cliff_blackout_on():
            return
        schedule, mask = self.blackout, self.blackout.mask

        def no_cliff_blackout(step):
            out = mask(step)
            cols = family_column_masks(env).get(CLIFF_FAMILY)
            return out if cols is None else out & ~(_column(env)[:, None] == cols[None]).any(1)

        schedule.mask = no_cliff_blackout
