"""Training-only auxiliary map head for the CNN-GRU RVLD student."""

import torch
from torch import nn


class TerrainAlignmentHead(nn.Module):
    """Decode the student GRU state into the teacher's 11x17 terrain grid.

    Output channels match GAVD: normalized height, visibility logit, upward edge,
    downward edge and local-support proxy. This head is auxiliary-only and is
    not part of the deployed latent/action graph.
    """

    def __init__(self, hidden_dim=64, grid_shape=(11, 17), channels=5):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.channels = int(channels)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ELU(),
            nn.Linear(128, self.grid_shape[0] * self.grid_shape[1] * self.channels),
        )

    def forward(self, hidden):
        return self.decoder(hidden).reshape(hidden.shape[0], -1, self.channels)
