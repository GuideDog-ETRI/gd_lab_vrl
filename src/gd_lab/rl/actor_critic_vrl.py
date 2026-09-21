"""Stage-1 vision teacher: the blind DreamWaQ actor-critic plus a privileged
terrain encoder.

Kept in its own module, as a subclass, so ``actor_critic.py`` stays byte-identical
to upstream and rebasing onto gd_lab's main costs nothing.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.networks import MLP
from tensordict import TensorDict

from .actor_critic import DreamwaqActorCritic


class HeightScanCNN(nn.Module):
    """height_scan (a flat GridPatternCfg raycast grid) -> terrain latent.

    height_scan is not an unordered feature vector: it is a regular grid (this
    robot's scanner is GridPatternCfg(resolution=0.1, size=[1.6,1.0]),
    ordering="xy" -> 17 points along x, 11 along y, flattened y-major/x-minor,
    17*11=187). A plain MLP has to learn spatial adjacency from scratch; a CNN
    gets it for free and picks up local patterns -- a step edge, a gap -- more
    data-efficiently.

    This costs nothing at deploy time: the terrain encoder only ever runs while
    the teacher trains, because privileged height_scan does not exist on
    hardware. The camera student replaces this whole module, so its
    architecture is free to differ from the student's own CNN.
    """

    def __init__(self, grid_shape: tuple[int, int], latent_dim: int, hidden_channels: tuple[int, int] = (8, 16)):
        super().__init__()
        self.grid_shape = grid_shape
        c1, c2 = hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            # No striding before this: 187 cells is already small, unlike the
            # camera images, so there is little to gain and real resolution to
            # lose. The pool only trims the flatten width for the head.
            nn.AdaptiveAvgPool2d((grid_shape[0] // 2, grid_shape[1] // 2)),
            nn.Flatten(),
        )
        pooled = (grid_shape[0] // 2) * (grid_shape[1] // 2)
        self.head = nn.Sequential(nn.Linear(c2 * pooled, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.Tanh())

    def forward(self, height_scan: torch.Tensor) -> torch.Tensor:
        expected = 2 * self.grid_shape[0] * self.grid_shape[1]
        if height_scan.shape[-1] != expected:
            raise ValueError(f"terrain encoder expects {expected} values, got {height_scan.shape[-1]}")
        grid = height_scan.reshape(height_scan.shape[0], 2, *self.grid_shape)
        return self.head(self.net(grid))


class DreamwaqVrlActorCritic(DreamwaqActorCritic):
    """Actor input = cat(K newest frames, CENet code, terrain latent).

    Everything except the terrain latent is inherited unchanged. The latent is
    produced from the privileged height_scan block of the critic observation
    group, located by the spec-derived ``height_scan_start`` plus the grid size.

    Unlike the CENet code, the terrain latent is **not** detached: the teacher's
    PPO gradient trains the terrain encoder, which is the point. Once trained,
    the encoder's output becomes the regression target the camera student learns
    to predict from images, and the student then substitutes for it at deploy.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        height_scan_start: int,
        terrain_latent_dim: int = 32,
        terrain_encoder_hidden_channels: tuple[int, int] = (8, 16),
        height_scan_grid_shape: tuple[int, int] = (11, 17),
        **kwargs: Any,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        critic_obs_dim = obs[obs_groups["critic"][0]].shape[-1]
        rows, cols = height_scan_grid_shape
        stop = height_scan_start + rows * cols
        if height_scan_start < 0 or min(rows, cols) < 2 or stop > critic_obs_dim:
            raise ValueError(
                f"height_scan_grid_shape {height_scan_grid_shape} needs critic obs [{height_scan_start}:{stop}] "
                f"but the group is only {critic_obs_dim} wide"
            )
        self._height_scan_slice = slice(height_scan_start, stop)
        self.terrain_latent_dim = terrain_latent_dim

        self.terrain_encoder = HeightScanCNN(
            height_scan_grid_shape, terrain_latent_dim, hidden_channels=tuple(terrain_encoder_hidden_channels)
        )

        # Widen the actor the base class just built: it sized itself for
        # (frames + code) and has no terrain latent in it.
        self.actor = MLP(
            self.one_step_obs_dim * self.actor_history_steps + self.cenet.code_dim + terrain_latent_dim,
            num_actions,
            kwargs.get("actor_hidden_dims", [512, 256, 128]),
            kwargs.get("activation", "elu"),
        )

    def normalized_height_scan(self, obs: TensorDict) -> torch.Tensor:
        """height_scan slice of the critic group, scaled by the critic normalizer's own stats.

        Mirrors ``normalize_latest``'s approach (reuse an existing normalizer's
        per-feature statistics for a sub-slice) but on the critic side, since the
        terrain encoder reads privileged critic data. Feeding the raw slice
        instead would hand the CNN a differently-scaled input than every other
        consumer of that group sees.
        """
        terrain = obs.get("terrain")
        if terrain is None:
            raise KeyError("VRL observation group 'terrain' is required")
        expected = 2 * (self._height_scan_slice.stop - self._height_scan_slice.start)
        if terrain.shape[-1] != expected:
            raise ValueError(f"VRL terrain group must have width {expected}, got {terrain.shape[-1]}")
        return terrain

    def terrain_latent(self, obs: TensorDict) -> torch.Tensor:
        """Privileged terrain latent. Also the student's distillation target."""
        return self.terrain_encoder(self.normalized_height_scan(obs))

    def _actor_input(self, obs: TensorDict, *, inference: bool) -> torch.Tensor:
        # PPO recomputes log probabilities from stored observations, not stored
        # VAE noise. Use posterior means so the same policy has the same mean
        # on rollout and replay. The auxiliary VAE loss still samples z.
        base = super()._actor_input(obs, inference=True)
        if self._should_inject_gt(inference):
            base = base.clone()
            start = self.one_step_obs_dim * self.actor_history_steps
            velocity = self.velocity_target(obs)
            base[..., start : start + velocity.shape[-1]] = velocity
        return torch.cat((base, self.terrain_latent(obs)), dim=-1)

    def act_with_terrain_latent(self, obs: TensorDict, terrain_latent: torch.Tensor) -> torch.Tensor:
        """Student inference using the same K-frame/CENet path as the teacher."""
        base = super()._actor_input(obs, inference=True)
        return self.actor(torch.cat((base, terrain_latent), dim=-1))
