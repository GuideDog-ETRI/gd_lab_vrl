"""Unchanged CENet + terrain denoising autoencoder and PPO."""
import torch
from torch import nn
from isaaclab.utils import configclass
from gd_lab.teachers.cvtt.actor_critic import DreamwaqVrlActorCritic
from gd_lab.teachers.cvtt.agent_cfg import DreamwaqVrlRunnerCfg
from gd_lab.rl.ppo import DreamwaqPPO
from gd_lab.gast.geometry import reconstruction_loss
from gd_lab.gast.temporal import TemporalTerrainEncoder
from gd_lab.rl.distributed import average_gradients, update_normalizer
from gd_lab.methods.dreamwaq.vrl_symmetry import mirror_vrl_observations


class GastActorCritic(DreamwaqVrlActorCritic):
    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            update_normalizer(self.actor_obs_normalizer, self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            update_normalizer(self.critic_obs_normalizer, self.get_critic_obs(obs))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terrain_encoder = TemporalTerrainEncoder()
        self.terrain_decoder = nn.Sequential(nn.Linear(32, 128), nn.ELU(), nn.Linear(128, 187*6))

    def terrain_latent(self, obs):
        terrain = self.normalized_height_scan(obs)
        latent = self.terrain_encoder(obs['gast_history'])
        # Exact zero terrain contribution during no-scan segments.
        available = (terrain[:, 187:].sum(-1, keepdim=True) > 0).to(latent.dtype)
        return latent * available


def mirror_gast(obs=None, actions=None, env=None):
    base = obs.exclude('gast_clean', 'gast_history') if obs is not None else None
    augmented, action_aug = mirror_vrl_observations(base, actions, env)
    if obs is not None:
        clean = obs['gast_clean']
        augmented['gast_clean'] = torch.cat((clean, clean.reshape(-1, 4, 11, 17).flip(-2).reshape_as(clean)), 0)
        history = obs['gast_history'].reshape(-1, 8, 375)
        mirrored = history.clone()
        mirrored[..., :374] = history[..., :374].reshape(-1, 8, 2, 11, 17).flip(-2).reshape(-1, 8, 374)
        augmented['gast_history'] = torch.cat((history.flatten(1), mirrored.flatten(1)), 0)
    return augmented, action_aug


class GastPPO(DreamwaqPPO):
    def update(self):
        metrics = super().update()
        terrain = self.storage.observations['gast_history'].flatten(0, 1)
        clean = self.storage.observations['gast_clean'].flatten(0, 1)
        # A bounded auxiliary pass, using the SAME Adam state saved by the runner.
        idx = torch.randperm(terrain.shape[0], device=terrain.device)[:2048]
        encoded = self.policy.terrain_encoder(terrain[idx])
        pred = self.policy.terrain_decoder(encoded).reshape(-1, 187, 6)
        loss = reconstruction_loss(pred, clean[idx])
        self.optimizer.zero_grad()
        (.1 * loss).backward()
        average_gradients(self.policy.parameters())
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        metrics['gast_reconstruction'] = loss.item()
        return metrics


@configclass
class GastRunnerCfg(DreamwaqVrlRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = 'gd_lab.gast.teacher:GastActorCritic'
        self.algorithm.class_name = 'gd_lab.gast.teacher:GastPPO'
        self.algorithm.symmetry_cfg.data_augmentation_func = 'gd_lab.gast.teacher:mirror_gast'
        self.logger = 'tensorboard'
        self.experiment_name = 'gast/arm4'
