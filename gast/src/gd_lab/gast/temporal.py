"""Pure Torch encoder; temporal history is explicit PPO observation."""
import torch
from torch import nn

HISTORY_OFFSETS = (48, 32, 16, 8, 4, 2, 1, 0)

class TemporalTerrainEncoder(nn.Module):
    grid_shape = (11, 17)
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ELU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ELU(), nn.AdaptiveAvgPool2d((5, 8)),
            nn.Flatten(), nn.Linear(640, 64), nn.ELU())
        self.age = nn.Linear(2, 64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
    def forward(self, history):
        seq = history.reshape(-1, 8, 375)
        features = self.cnn(seq[..., :374].reshape(-1, 2, 11, 17)).reshape(-1, 8, 64)
        features = features + self.age(torch.stack((seq[..., -1]/5, seq[..., 187:374].mean(-1)), -1))
        mixed, _ = self.attention(features, features, features, need_weights=False)
        _, hidden = self.gru(self.norm(features + mixed))
        return self.head(hidden[0])
