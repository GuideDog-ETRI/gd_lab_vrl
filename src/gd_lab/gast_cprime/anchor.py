"""Stage C' PPO: the v2.1 retention anchor + the v2.4c stand anchor on the CLIFF ZONE.

Stand anchor (t1d, 11-02). Why (A6000 cprime_t1c): the staged zero command made the start model stand in the zone from
update 0 (cliff hold 1.00, level 0 -> 11), but the policy credited the standing to the zero command, not to the drop in
its terrain input / support_end: as soon as the command ramped back (kappa 0.2 at update 120) the hold fell to 0.79,
then 0.17 at update 150 -- the same walk-off as t1 / t1b. A zero-command device cannot teach "ignore the command when
the support ends" because in every standing sample there is no command to ignore.

The stand anchor of v2.4c (gast_narrow.anchor, run v24h: in-place-turn stop 0.2 -> 64-78 %) gives that gradient
directly: on the rollout samples inside the cliff zone the policy's deterministic action is pulled toward the action of
the FROZEN start policy (NARROW_STAND_ANCHOR_CKPT, default NARROW_SOURCE = the B' model) on the same observation with
the velocity command (all history frames) and the support_end input zeroed -- "support ends ahead = behave as the start
policy does under a zero command" (it stands). Unlike v2.4e's zone extension (v24i: passable and blocked corridors
differ only by the margin input, PPO pushed the other way), the cliff zone is unambiguous in the inputs (the terrain
input shows the drop, support_end > 0, nothing to pass) and PPO agrees with the pull there (tracking off, cliff_stand /
cliff_slow / hold +4 pay for standing, breach -4), so the two do not fight. The zone mask is handed over each env step
by gast_cprime.terms.cliff_slow. Result t1d it150: cliff hold 0.50-0.74 with breach 0-0.03 (t1c: 0.17 / 0.83).

Retention on the passages (t1e, CPRIME_RETAIN_PASSAGES, default 1). Why (t1d it100-150): the tight-passage pass rate
fell 0.92 (B' margin_t1) -> 0.55-0.80 while the blocked hold rose 0.84 -> 0.92 -- the v24i pattern: the extra "stand"
pressure leaked into the passable tight corridors, which differ from the blocked ones only by the margin input. B'
already solved the passages and C' does not teach anything new there, so the retention anchor (frozen B' source) now
covers every sample that is not on a cliff deck -- gap / stairs / slopes as before PLUS corridors, doors, bends, dead
ends. Cliff decks keep plain PPO + the stand anchor. The cliff-tile mask is handed over with the zone mask.

Gap focus (t1f, CPRIME_GAP_FOCUS_WEIGHT, default 1.0). Why (E2 t1d model_299): the stand anchor spread from the cliff
zone to the gaps -- support_end fires 22-45 % in front of a gap and the terrain input shows a drop there too: gap
crossing 39.5 % (margin_t1 94.4 %), not reached 41 % forward / 73 % backward. The retention anchor already covered the
gap samples, but as a mean over all off-cliff samples (stairs, slopes, rough, passages) the few gap-approach samples
weigh little. The focus term is the same retention loss (same frozen target, real command) averaged over the gap-tile
samples before the crossing only, with its own weight: the counter-pair of the stand anchor ("cliff: stand, gap: go").

Env: NARROW_STAND_ANCHOR_WEIGHT (default 1.0, 0 = off), NARROW_STAND_ANCHOR_CKPT (default NARROW_SOURCE),
CPRIME_RETAIN_PASSAGES (default 1; 0 = the v2.1 rule, narrow tiles excluded from the retention anchor),
CPRIME_GAP_FOCUS_WEIGHT (default 1.0; 0 = off), CPRIME_STAND_MIN_SUPPORT (default 0.2: stand anchor only on zone samples
with support_end above it; 0 = whole zone).
"""
from __future__ import annotations

import os

import torch

from gd_lab.gast_cprime.model import SUPPORT_GROUP
from gd_lab.gast_narrow.anchor import NarrowAnchorPPO

_cliff_zone_mask = None  # [num_envs] bool, this env step (set by gast_cprime.terms.cliff_slow)
_cliff_tile_mask = None  # [num_envs] bool, this env step: on a cliff deck (same setter)
_gap_approach_mask = None  # [num_envs] bool, this env step: gap tile before the crossing (gast_cprime.tasks stall term)


def set_gap_approach_mask(mask: torch.Tensor) -> None:
    global _gap_approach_mask
    _gap_approach_mask = mask


def set_cliff_zone_mask(mask: torch.Tensor, tile: torch.Tensor | None = None) -> None:
    global _cliff_zone_mask, _cliff_tile_mask
    _cliff_zone_mask = mask
    if tile is not None:
        _cliff_tile_mask = tile


def retain_passages() -> bool:
    return os.environ.get("CPRIME_RETAIN_PASSAGES", "1") == "1"


STAND_MIN_SUPPORT = float(os.environ.get("CPRIME_STAND_MIN_SUPPORT", "0.2"))  # 0 = the t1d rule (whole zone)


def gap_focus_weight() -> float:
    return float(os.environ.get("CPRIME_GAP_FOCUS_WEIGHT", "1.0"))


class CPrimeAnchorPPO(NarrowAnchorPPO):
    def _record(self, name: str, mask) -> None:
        buf = getattr(self, name, None)
        if buf is None:
            buf = torch.zeros(self.storage.num_transitions_per_env, mask.shape[0], dtype=torch.bool, device=self.device)
            setattr(self, name, buf)
        buf[self.storage.step] = mask.to(self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        # Before super(): storage.step is still the index of the transition this step completes.
        if self._stand_anchor_wanted() and _cliff_zone_mask is not None:
            self._record("_cliff_buf", _cliff_zone_mask)
        if retain_passages() and _cliff_tile_mask is not None:
            self._record("_cliff_tile_buf", _cliff_tile_mask)
        if gap_focus_weight() > 0 and _gap_approach_mask is not None:
            self._record("_gap_buf", _gap_approach_mask)
        return super().process_env_step(obs, rewards, dones, extras)

    @staticmethod
    def _stand_anchor_wanted() -> bool:
        return float(os.environ.get("NARROW_STAND_ANCHOR_WEIGHT", "1.0")) > 0

    def _retention_samples(self, obs_all):
        if not retain_passages():
            return super()._retention_samples(obs_all)
        buf = getattr(self, "_cliff_tile_buf", None)
        if buf is None:
            raise RuntimeError("C' passage retention: no cliff tile mask recorded (cliff_slow term missing?)")
        return ~buf.flatten(0, 1)  # every sample off the cliff decks; [T, N] -> T-major like storage.observations

    def _focus_samples(self, obs_all):
        if gap_focus_weight() <= 0:
            return None, 0.0
        buf = getattr(self, "_gap_buf", None)
        if buf is None:
            raise RuntimeError("C' gap focus: no gap approach mask recorded (stall term missing?)")
        return buf.flatten(0, 1), gap_focus_weight()

    def _stand_samples(self, obs_all):
        if self._stand_weight <= 0:
            return None
        buf = getattr(self, "_cliff_buf", None)
        if buf is None:
            raise RuntimeError("C' stand anchor: no cliff zone mask recorded (cliff_slow term missing?)")
        zone = buf.flatten(0, 1)  # [T, N] -> T-major, as storage.observations.flatten(0, 1)
        if STAND_MIN_SUPPORT > 0:
            # t1g: only where the actor's own input shows the end of support (support_end is read off the actor frame
            # after visibility / blackout). Replay t1f model_59: in the zone the actor frame showed the drop on 18 %
            # of the frames (38 % blacked out, 43 % flat) -- the anchor taught "stand" on inputs that look like
            # walking, so the policy also stood on flat ground (25-50 % of drop-free episodes, margin_t1 0 %).
            zone = zone & (obs_all[SUPPORT_GROUP][..., 0] > STAND_MIN_SUPPORT)
        return zone

    def _stand_target_obs(self, obs):
        out = super()._stand_target_obs(obs)  # velocity command zeroed in every history frame
        if SUPPORT_GROUP in out.keys():
            out[SUPPORT_GROUP] = torch.zeros_like(obs[SUPPORT_GROUP])
        return out
