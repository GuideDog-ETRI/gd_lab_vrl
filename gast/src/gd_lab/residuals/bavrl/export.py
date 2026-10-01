"""BAVRL simulation export; explicit packed residual state, never a CVTT actor."""

import copy

import torch
from torch import nn

from gd_lab.deploy.export_vrl import _time_major_to_term_major_index


class BAVRLActorExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval()
        t = self.model.teacher
        self.register_buffer("reorder", _time_major_to_term_major_index([3, 3, 3, 12, 12, 12, 1], 5))
        self.register_buffer("older", t.frames_idx[t.one_step_obs_dim:].clone())

    def forward(self, direct_obs, cenet_obs, terrain_latent):
        t = self.model.teacher
        history = t.actor_obs_normalizer(cenet_obs[:, self.reorder])
        latest = (direct_obs-t.actor_obs_normalizer._mean[:, t.latest_idx]) / (
            t.actor_obs_normalizer._std[:, t.latest_idx]+t.actor_obs_normalizer.eps)
        code = t.cenet.get_code(t.cenet(history, deterministic=True))
        features = torch.cat((latest, history[:, self.older], code), -1)
        base = t.actor(features)
        latent, previous = terrain_latent[:, :32], terrain_latent[:, 32:44]
        age = terrain_latent[:, 44:45]
        good = (terrain_latent[:, 45] > .5) & (age[:, 0] >= 0) & (age[:, 0] <= self.model.config.stale_seconds)
        mean = self.model.residual(torch.cat((features, latent, previous,
                                             (age/self.model.config.stale_seconds).clamp(0, 1)), -1))
        action, delta = self.model.compose(base, mean, previous, good)
        return action, code, delta


class BAVRLVisionExport(nn.Module):
    architecture = "grid_attention_v1"
    num_cameras = 4
    gru_hidden_dim = 64

    def __init__(self, model):
        super().__init__()
        self.vision = copy.deepcopy(model.vision).cpu().eval()
        self.stale = model.config.stale_seconds

    def forward(self, frames, hidden):
        # Shared camera thread supplies seconds; training uses age/stale limit.
        memory = torch.cat((hidden[:, :63], (hidden[:, 63:64]/self.stale).clamp(0, 1)), -1)
        return self.vision(frames, memory)
