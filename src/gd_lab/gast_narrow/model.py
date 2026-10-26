"""6566 Ray+IR GAST teacher + head-LiDAR obstacle encoder (additive, zero-initialised actor/critic columns).

actor input  = [6566 actor input (frames, CENet code, terrain latent 32), lidar latent 16]
critic input = [critic_obs_normalizer(critic group), narrow_privileged 6]   (critic group unchanged)
At warm start the new actor/critic input columns are zero, so the policy and value equal 6566 exactly.
"""
from __future__ import annotations

import torch
from torch import nn
from isaaclab.utils import configclass

from gd_lab.gast.teacher import GastRayIrActorCritic, GastRayIrRunnerCfg, mirror_gast
from gd_lab.gast_narrow import lidar

LIDAR_LATENT = 16
PRIV_DIM = 6
LIDAR_GROUP, PRIV_GROUP = "narrow_lidar", "narrow_priv"
LIDAR_PREFIX = "lidar_encoder."


class LidarEncoder(nn.Module):
    def __init__(self, latent=LIDAR_LATENT):
        super().__init__()
        self.cnn = nn.Sequential(
            # Small channel counts: PPO stores 1024 x 100 maps and runs mirrored minibatches of ~51k (GPU memory).
            nn.Conv2d(lidar.FRAMES, 8, 5, stride=2, padding=2), nn.ELU(),    # 15 x 24
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ELU(),              # 8 x 12
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),             # 4 x 6
            nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32 * 4 * 6, 64), nn.ELU(), nn.Linear(64, latent))

    def forward(self, flat):
        return self.head(self.cnn(flat.reshape(-1, lidar.FRAMES, lidar.NY, lidar.NX)))


def _widen_first_linear(mlp: nn.Sequential, extra: int) -> None:
    first = mlp[0]
    grown = nn.Linear(first.in_features + extra, first.out_features)
    with torch.no_grad():
        grown.weight.zero_()
        grown.weight[:, :first.in_features] = first.weight
        grown.bias.copy_(first.bias)
    mlp[0] = grown


class GastNarrowActorCritic(GastRayIrActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lidar_encoder = LidarEncoder()
        _widen_first_linear(self.actor, LIDAR_LATENT)
        _widen_first_linear(self.critic, PRIV_DIM)

    def lidar_latent(self, obs):
        return self.lidar_encoder(obs[LIDAR_GROUP])

    def _actor_input(self, obs, *, inference: bool):
        return torch.cat((super()._actor_input(obs, inference=inference), self.lidar_latent(obs)), dim=-1)

    def evaluate(self, obs, **kwargs):
        critic = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(torch.cat((critic, obs[PRIV_GROUP]), dim=-1))


def mirror_narrow(obs=None, actions=None, env=None):
    base = obs.exclude(LIDAR_GROUP, PRIV_GROUP) if obs is not None else None
    augmented, action_aug = mirror_gast(base, actions, env)
    if obs is not None:
        grid = obs[LIDAR_GROUP]
        mirrored = grid.reshape(-1, lidar.FRAMES, lidar.NY, lidar.NX).flip(-2).reshape_as(grid)
        augmented[LIDAR_GROUP] = torch.cat((grid, mirrored), 0)
        priv = obs[PRIV_GROUP]
        # [d_left, d_right, lateral, heading error, progress, flag] -> swap sides, negate lateral and heading.
        flipped = torch.stack((priv[..., 1], priv[..., 0], -priv[..., 2], -priv[..., 3], priv[..., 4], priv[..., 5]),
                              -1)
        augmented[PRIV_GROUP] = torch.cat((priv, flipped), 0)
    return augmented, action_aug


@configclass
class GastNarrowRunnerCfg(GastRayIrRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "gd_lab.gast_narrow.model:GastNarrowActorCritic"
        self.algorithm.symmetry_cfg.data_augmentation_func = "gd_lab.gast_narrow.model:mirror_narrow"
        self.experiment_name = "gast/narrow"
