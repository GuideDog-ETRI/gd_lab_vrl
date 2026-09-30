"""VRL teacher whose terrain latent is zero when no terrain cell is valid.

A missing camera stream (all cells invalid) therefore always reaches the actor as
the same zero latent -- the input under which, with terrain blackouts during
training, the actor learns to walk blind. Deployment must send zeros too.
"""

from __future__ import annotations

import torch
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic
from tensordict import TensorDict


class DreamwaqVrlGatedActorCritic(DreamwaqVrlActorCritic):
    def terrain_latent(self, obs: TensorDict) -> torch.Tensor:
        terrain = self.normalized_height_scan(obs)
        cells = terrain.shape[-1] // 2
        any_valid = (terrain[..., cells:] > 0).any(-1, keepdim=True)
        return self.terrain_encoder(terrain) * any_valid
