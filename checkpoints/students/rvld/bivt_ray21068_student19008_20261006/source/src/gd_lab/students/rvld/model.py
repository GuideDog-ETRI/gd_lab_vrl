"""Vision-RL stage 3: CNN-GRU perception student.

Distills ``DreamwaqVrlActorCritic.terrain_encoder`` (the frozen, privileged
height_scan teacher trained in stage 1) into a module that reproduces the
same latent from the 4 real belly cameras (depth + IR) trained in stage 2 --
matching APT-RL's teacher-student split (Fig. 2iii) and gd_rbq10_deploy's own
async low-rate perception thread (this module is meant to run at the
cameras' own slower update rate, not every policy step).
"""

from __future__ import annotations

import torch
from torch import nn


def height_discontinuity_metres(
    height_scan: torch.Tensor,
    scale: float,
    grid_shape=(11, 17),
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Maximum visible adjacent height jump in metres."""
    if scale == 0 or min(grid_shape) < 2:
        raise ValueError("height scan scale must be nonzero and grid axes >= 2")
    grid = height_scan.reshape(height_scan.shape[0], *grid_shape) / scale
    rows = (grid[:, 1:, :] - grid[:, :-1, :]).abs()
    cols = (grid[:, :, 1:] - grid[:, :, :-1]).abs()
    if valid_mask is not None:
        valid = valid_mask.reshape(valid_mask.shape[0], *grid_shape).bool()
        rows = torch.where(valid[:, 1:, :] & valid[:, :-1, :], rows, 0.0)
        cols = torch.where(valid[:, :, 1:] & valid[:, :, :-1], cols, 0.0)
    rows = torch.where(torch.isfinite(rows), rows, 0.0)
    cols = torch.where(torch.isfinite(cols), cols, 0.0)
    return torch.maximum(rows.amax(dim=(1, 2)), cols.amax(dim=(1, 2)))


class _CameraCNN(nn.Module):
    """Shared per-camera trunk: (depth, ir) 2-channel image -> flat feature vector.

    Weights are shared across all 4 cameras (one small trunk, not four) --
    the cameras are the same physical sensor model (D430) at different
    mounts, so there is no reason to spend separate parameters per view.
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 32) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 2026-09-19: (2,2) -> (5,3). At 80x45 input (16:9, matching the
            # real D430's streamed aspect) the 3 stride-2 convs above reduce
            # to 10x6, where (at ~1.5m viewing distance, D430's real H87/V58
            # deg FOV) one cell covers roughly 28x28cm -- i.e. a 22cm
            # subway-gap-scale hazard already occupies about ONE cell here.
            # Pooling all the way to (2,2) averages that one cell together
            # with ~14 neighboring (mostly safe) cells, nearly erasing it.
            # (5,3) is exactly a 2x2 box-average of the native 10x6 map: it
            # keeps some noise-smoothing (still useful against per-pixel
            # sensor noise and the hotspot patches in CameraNoiseCfg) while
            # only merging the hazard cell with ~4 neighbors instead of ~14.
            # Cost is negligible either way (a few hundred more MACs in the
            # following Linear, well inside the ~80ms perception budget).
            # (H, W), not (W, H) -- at 80x45 input the three stride-2 convs leave
            # 6x10, so this halves each axis exactly. A non-integer ratio still
            # trains fine but makes the graph un-exportable: ONNX rejects
            # adaptive_avg_pool2d whose output is not a factor of its input.
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(32 * 3 * 5, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class CameraPerceptionEncoder(nn.Module):
    """4-camera depth+IR CNN -> fused GRU -> terrain latent (student).

    Stateless between calls by design: the caller owns the GRU ``hidden``
    tensor and is responsible for (a) only calling ``forward`` on steps where
    the camera sensors actually refreshed, and (b) zeroing the rows of
    ``hidden`` belonging to envs/episodes that just reset. This mirrors
    gd_rbq10_deploy's separate-thread perception model, where the hidden
    state is the only thing that carries across calls at the perception
    thread's own (lower) rate.
    """

    def __init__(
        self,
        num_cameras: int = 4,
        in_channels_per_camera: int = 2,
        cnn_feature_dim: int = 32,
        gru_hidden_dim: int = 64,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.gru_hidden_dim = gru_hidden_dim
        self.latent_dim = latent_dim
        self.camera_cnn = _CameraCNN(in_channels_per_camera, cnn_feature_dim)
        self.fuse = nn.Sequential(nn.Linear(cnn_feature_dim * num_cameras, gru_hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(gru_hidden_dim, gru_hidden_dim)
        self.head = nn.Sequential(nn.Linear(gru_hidden_dim, latent_dim), nn.Tanh())

        # 2026-09-18: auxiliary "hazard proximity" head -- trained (by
        # train_perception.py, outside forward()) to regress the largest
        # adjacent-cell height discontinuity in the teacher's own privileged
        # height_scan, i.e. "how close/severe is the nearest foot-safety edge
        # (stair nosing, platform gap) right now." Ground truth comes from
        # sim state, not from these noisy camera frames, so this loss keeps
        # pulling the shared CNN+GRU trunk toward hazard-relevant features
        # even on training steps where the camera frames are badly corrupted
        # (see CameraNoiseCfg's hotspot fields) and the direct latent-MSE
        # signal alone might not be enough. Deliberately NOT called from
        # forward() -- see this module's own docstring plus
        # deploy/export_student_vrl.py -- so it never enters the traced/
        # exported graph and gd_rbq10_deploy needs zero changes.
        self.hazard_head = nn.Sequential(nn.Linear(gru_hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def init_hidden(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(num_envs, self.gru_hidden_dim, device=device)

    def forward(self, frames: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``frames``: (num_envs, num_cameras, in_channels, H, W). Returns (latent, new_hidden)."""
        num_envs, num_cameras = frames.shape[:2]
        flat = frames.reshape(num_envs * num_cameras, *frames.shape[2:])
        feats = self.camera_cnn(flat).reshape(num_envs, num_cameras * self.camera_cnn.out_dim)
        fused = self.fuse(feats)
        new_hidden = self.gru(fused, hidden)
        latent = self.head(new_hidden)
        return latent, new_hidden
