"""Stage C' PPO: the v2.1 retention anchor (B' source on non-narrow samples) + the v2.4c stand anchor on the CLIFF ZONE.

Why (A6000 cprime_t1c, 11-02): the staged zero command (gast_cprime.commands) made the start model stand in the zone
from update 0 (cliff hold 1.00, level 0 -> 11), but the policy credited the standing to the zero command, not to the
drop in its terrain input / support_end: as soon as the command ramped back (kappa 0.2 at update 120) the hold fell to
0.79, then 0.17 at update 150 -- the same walk-off as t1 / t1b. A zero-command device cannot teach "ignore the command
when the support ends" because in every standing sample there is no command to ignore.

The stand anchor of v2.4c (gast_narrow.anchor, run v24h: in-place-turn stop 0.2 -> 64-78 %) gives that gradient
directly: on the rollout samples inside the cliff zone the policy's deterministic action is pulled toward the action of
the FROZEN start policy (NARROW_STAND_ANCHOR_CKPT, default NARROW_SOURCE = the B' model) on the same observation with
the velocity command (all history frames) and the support_end input zeroed -- "support ends ahead = behave as the start
policy does under a zero command" (it stands). Unlike v2.4e's zone extension (v24i: passable and blocked corridors
differ only by the margin input, PPO pushed the other way), the cliff zone is unambiguous in the inputs (the terrain
input shows the drop, support_end > 0, nothing to pass) and PPO agrees with the pull there (tracking off, cliff_stand /
cliff_slow / hold +4 pay for standing, breach -4), so the two do not fight. Gap tiles are protected by the retention
anchor (non-narrow samples follow the B' source), gap samples are never in the zone mask (privileged cliff state), and
the gait v3 terms are untouched. The zone mask is handed over each env step by gast_cprime.terms.cliff_slow.

Env: NARROW_STAND_ANCHOR_WEIGHT (default 1.0, 0 = off), NARROW_STAND_ANCHOR_CKPT (default NARROW_SOURCE).
"""
from __future__ import annotations

import os

import torch

from gd_lab.gast_cprime.model import SUPPORT_GROUP
from gd_lab.gast_narrow.anchor import NarrowAnchorPPO

_cliff_zone_mask = None  # [num_envs] bool, this env step (set by gast_cprime.terms.cliff_slow)


def set_cliff_zone_mask(mask: torch.Tensor) -> None:
    global _cliff_zone_mask
    _cliff_zone_mask = mask


class CPrimeAnchorPPO(NarrowAnchorPPO):
    def process_env_step(self, obs, rewards, dones, extras):
        # Before super(): storage.step is still the index of the transition this step completes.
        if self._stand_anchor_wanted() and _cliff_zone_mask is not None:
            buf = getattr(self, "_cliff_buf", None)
            if buf is None:
                buf = self._cliff_buf = torch.zeros(self.storage.num_transitions_per_env, _cliff_zone_mask.shape[0],
                                                    dtype=torch.bool, device=self.device)
            buf[self.storage.step] = _cliff_zone_mask.to(self.device)
        return super().process_env_step(obs, rewards, dones, extras)

    @staticmethod
    def _stand_anchor_wanted() -> bool:
        return float(os.environ.get("NARROW_STAND_ANCHOR_WEIGHT", "1.0")) > 0

    def _stand_samples(self, obs_all):
        if self._stand_weight <= 0:
            return None
        buf = getattr(self, "_cliff_buf", None)
        if buf is None:
            raise RuntimeError("C' stand anchor: no cliff zone mask recorded (cliff_slow term missing?)")
        return buf.flatten(0, 1)  # [T, N] -> T-major, as storage.observations.flatten(0, 1)

    def _stand_target_obs(self, obs):
        out = super()._stand_target_obs(obs)  # velocity command zeroed in every history frame
        if SUPPORT_GROUP in out.keys():
            out[SUPPORT_GROUP] = torch.zeros_like(obs[SUPPORT_GROUP])
        return out
