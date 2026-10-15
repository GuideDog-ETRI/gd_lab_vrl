"""Pure Torch encoder; temporal history is explicit PPO observation."""
import os
from contextlib import contextmanager
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

HISTORY_OFFSETS = (48, 32, 16, 8, 4, 2, 1, 0)


@contextmanager
def attention_backend(math_only):
    """GAST_MATH_SDPA=1 forces the math SDPA kernel (same formula) to isolate fused-kernel backward faults."""
    if not math_only:
        yield
        return
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        yield


class TemporalTerrainEncoder(nn.Module):
    grid_shape = (11, 17)
    def __init__(self, channels=2):
        super().__init__()
        # channels=2: [height, valid] (GAST v2.1); 3: + IR (UPDATE134B Ray+IR teacher).
        self.channels = channels
        self.execution_chunk_size = 8192
        # Diagnosis switches for the 8016/8148 non-finite backward; both off = unchanged behaviour.
        self.math_sdpa = os.getenv("GAST_MATH_SDPA", "0") == "1"
        self.disable_cudnn_gru = os.getenv("GAST_NO_CUDNN_GRU", "0") == "1"
        self.cnn = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.ELU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ELU(), nn.AdaptiveAvgPool2d((5, 8)),
            nn.Flatten(), nn.Linear(640, 64), nn.ELU())
        self.age = nn.Linear(2, 64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
    def forward(self, history):
        # Symmetry doubles PPO batches beyond the CUDA attention grid limit.
        # Samples are independent: split only execution, not PPO minibatches.
        # Recompute chunk activations during backward to bound memory use.
        if history.shape[0] > self.execution_chunk_size:
            return torch.cat([checkpoint(self._encode, chunk, use_reentrant=False)
                              if torch.is_grad_enabled() else self._encode(chunk)
                              for chunk in history.split(self.execution_chunk_size)], 0)
        return self._encode(history)

    def _encode(self, history):
        width = 187 * self.channels
        seq = history.reshape(-1, 8, width + 1)
        features = self.cnn(seq[..., :width].reshape(-1, self.channels, 11, 17)).reshape(-1, 8, 64)
        features = features + self.age(torch.stack((seq[..., -1]/5, seq[..., 187:374].mean(-1)), -1))
        with attention_backend(self.math_sdpa):
            mixed, _ = self.attention(features, features, features, need_weights=False)
        normalized = self.norm(features + mixed)
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled and not self.disable_cudnn_gru):
            _, hidden = self.gru(normalized)
        return self.head(hidden[0])
