"""C' actor-critic: the v2.4c network + 1 actor input (support_end), zero column at warm start (the B' source acts
unchanged until the column is learned); the critic is unchanged (narrow_priv already carries the edge margin)."""
from __future__ import annotations

import torch

from gd_lab.gast_narrow.model import GastNarrowV24ActorCritic, _widen_first_linear, mirror_narrow_v24

SUPPORT_GROUP = "narrow_support_end"


class GastNarrowCPrimeActorCritic(GastNarrowV24ActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _widen_first_linear(self.actor, 1)

    def _actor_input(self, obs, *, inference: bool):
        return torch.cat((super()._actor_input(obs, inference=inference), obs[SUPPORT_GROUP]), dim=-1)


def mirror_narrow_cprime(obs=None, actions=None, env=None):
    base = obs.exclude(SUPPORT_GROUP) if obs is not None else None
    augmented, action_aug = mirror_narrow_v24(base, actions, env)
    if obs is not None:
        support = obs[SUPPORT_GROUP]
        augmented[SUPPORT_GROUP] = torch.cat((support, support), 0)  # centre-line value, no side
    return augmented, action_aug
