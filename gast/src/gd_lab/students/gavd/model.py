"""Small grid-query student. No Isaac dependency; legacy student is unchanged.

The 187 queries have robot-grid identities, not exact gravity-aligned BEV
reprojection (no live pose input). Invalid/far rays remain tokens: a missing
depth return can be a gap. Hidden slot 63 carries normalized delivery age on
input, and is zero on output; the caller sets it before each sensor update.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gd_lab.core.camera_contract import DEFAULT_CAMERA_PROFILE, load_camera_contract


def camera_ray_buffers(contract):
    """(rays[1,4,144,3], mounts[1,4,1,3]) of the 9x16 token grid, in the trunk frame, from a camera contract."""
    py, px = torch.meshgrid((torch.arange(9) + .5) * 45 / 9,
                            (torch.arange(16) + .5) * 80 / 16, indexing="ij")
    # Optical depth is z, not radial range. Convert OpenGL rays to trunk.
    ray = torch.stack(((px - 40) / contract.fx, -(py - 22.5) / contract.fy, -torch.ones_like(px)), -1)
    rays = []
    for quat in contract.quaternions_opengl:
        q = F.normalize(torch.tensor(quat, dtype=torch.float32), dim=0)  # profiles may list ints
        v = q[1:].expand_as(ray)
        rays.append(ray + 2 * (q[0] * torch.cross(v, ray, dim=-1)
                               + torch.cross(v, torch.cross(v, ray, dim=-1), dim=-1)))
    return torch.stack(rays).reshape(1, 4, 144, 3), torch.tensor(contract.positions, dtype=torch.float32).reshape(1, 4, 1, 3)


def _verify_after_load(module, incompatible_keys):
    module.verify_camera_geometry()


class GridAttentionStudent(nn.Module):
    architecture = "grid_attention_v1"
    num_cameras = 4
    gru_hidden_dim = 64
    latent_dim = 32
    age_slot = 63

    def __init__(self, camera_profile=DEFAULT_CAMERA_PROFILE, dim=32):
        super().__init__()
        self.camera_profile = camera_profile
        self.dim = dim
        c = load_camera_contract(camera_profile)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, dim, 3, stride=2, padding=1), nn.ELU(),
        )
        # Pool to 9x16: substantially more spatial tokens than legacy 3x5.
        self.camera_embedding = nn.Embedding(4, dim)
        self.geometry = nn.Linear(8, dim)  # xyz, ray xyz, valid, far/unknown
        ys, xs = torch.meshgrid(torch.linspace(-.5, .5, 11), torch.linspace(-.8, .8, 17), indexing="ij")
        self.register_buffer("grid_xy", torch.stack((xs, ys), -1).reshape(187, 2))
        self.query = nn.Linear(2, dim)
        self.query_identity = nn.Parameter(torch.randn(187, dim) * .02)
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ELU(), nn.Linear(dim * 2, dim))
        self.spatial_head = nn.Linear(dim, 5)  # height, validity, up, down, local support proxy
        self.fuse = nn.Linear(187 * dim, 63)
        self.gru = nn.GRUCell(64, 63)
        self.head = nn.Sequential(nn.Linear(63, 32), nn.Tanh())
        self.hazard_head = nn.Sequential(nn.Linear(64, 16), nn.ELU(), nn.Linear(16, 1))
        rays, mounts = camera_ray_buffers(c)
        # Persistent buffers: a checkpoint carries its camera geometry, so every load is verified. A post-load
        # hook (not a load_state_dict override) also fires when a parent module (e.g. BAVRL) is loaded.
        self.register_buffer("rays", rays)
        self.register_buffer("mounts", mounts)
        self.register_load_state_dict_post_hook(_verify_after_load)

    def verify_camera_geometry(self, atol=1e-5):
        """Fail if the loaded ray/mount buffers are not those of ``self.camera_profile``.

        A legacy checkpoint loaded into a model built for another profile would otherwise silently
        restore the old camera geometry (and bake it into an ONNX export).
        """
        rays, mounts = camera_ray_buffers(load_camera_contract(self.camera_profile))
        for name, expected in (("rays", rays), ("mounts", mounts)):
            actual = getattr(self, name)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise RuntimeError(f"student {name} buffer is malformed for camera_profile={self.camera_profile!r}")
            if not torch.allclose(actual.detach().cpu().to(expected.dtype), expected, atol=atol, rtol=0):
                raise RuntimeError(
                    f"student {name} geometry does not match camera_profile={self.camera_profile!r}: the checkpoint "
                    "was trained with another camera calibration; build the model with that profile instead"
                )


    def init_hidden(self, num_envs, device):
        return torch.zeros(num_envs, 64, device=device)

    def encode(self, frames, hidden):
        n = frames.shape[0]
        safe = torch.nan_to_num(frames, nan=1., posinf=1., neginf=0.).clamp(0, 1)
        images = safe.reshape(-1, 2, 45, 80)
        features = F.interpolate(self.cnn(images), size=(9, 16), mode="bilinear", align_corners=False)
        features = features.flatten(2).transpose(1, 2).reshape(n, 4, 144, self.dim)
        depth = F.interpolate(images[:, :1], size=(9, 16), mode="nearest").reshape(n, 4, 144, 1)
        valid = ((depth > 0) & (depth < .999)).to(depth.dtype)
        rays = self.rays.expand(n, -1, -1, -1)
        xyz = self.mounts + rays * (depth * 4.85 + .15)
        geometry = torch.cat((xyz / 5., F.normalize(rays, dim=-1), valid, 1-valid), -1)
        tokens = features + self.geometry(geometry) + self.camera_embedding.weight[None, :, None]
        tokens = tokens.reshape(n, 576, self.dim)
        queries = self.query(self.grid_xy) + self.query_identity
        weights = torch.softmax(queries[None] @ self.key(tokens).transpose(1, 2) / math.sqrt(self.dim), -1)
        grid = self.norm(queries[None] + weights @ self.value(tokens))
        grid = self.norm(grid + self.ff(grid))
        spatial = self.spatial_head(grid)
        age = hidden[:, 63:64].clamp(0, 1)
        memory = self.gru(torch.cat((torch.tanh(self.fuse(grid.flatten(1))), age), -1), hidden[:, :63])
        new_hidden = torch.cat((memory, torch.zeros_like(age)), -1)
        return self.head(memory), new_hidden, spatial

    def forward(self, frames, hidden):
        latent, memory, _ = self.encode(frames, hidden)
        return latent, memory


def spatial_targets(terrain, threshold=.04):
    """Visible-only geometry labels. Support means locally level, NOT guaranteed safe foothold."""
    h = terrain[:, :187].reshape(-1, 11, 17) / 5.
    valid = terrain[:, 187:].reshape(-1, 11, 17).bool() & torch.isfinite(h)
    # Observation h is scanner_z-ground_z-offset: negate delta for ground rise.
    delta = -(h[:, :, 1:] - h[:, :, :-1])
    pair = valid[:, :, 1:] & valid[:, :, :-1]
    up = F.pad((delta > threshold).float(), (0, 1))
    down = F.pad((delta < -threshold).float(), (0, 1))
    edge_valid = F.pad(pair, (0, 1))
    dx = F.pad((h[:, :, 1:] - h[:, :, :-1]).abs(), (0, 1))
    dy = F.pad((h[:, 1:, :] - h[:, :-1, :]).abs(), (0, 0, 0, 1))
    support_mask = edge_valid & F.pad(valid[:, 1:, :] & valid[:, :-1, :], (0, 0, 0, 1))
    support = ((dx < threshold) & (dy < threshold)).float()
    target = torch.stack((torch.nan_to_num(h), valid.float(), up, down, support), -1).flatten(1, 2)
    mask = torch.stack((valid, torch.ones_like(valid), edge_valid, edge_valid, support_mask), -1).flatten(1, 2)
    return target, mask


def spatial_loss(prediction, terrain):
    target, mask = spatial_targets(terrain)
    errors = torch.cat((F.smooth_l1_loss(prediction[..., :1], target[..., :1], reduction="none"),
                        F.binary_cross_entropy_with_logits(prediction[..., 1:], target[..., 1:], reduction="none")), -1)
    # Rare edges are weighted; invisible height/edges never become invented ground truth.
    weight = torch.ones_like(errors)
    weight[..., 2:4] = 1 + 4 * target[..., 2:4]
    return ((errors * mask * weight).sum((0, 1)) / (mask * weight).sum((0, 1)).clamp_min(1)).mean()
