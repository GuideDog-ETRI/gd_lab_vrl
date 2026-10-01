"""Per-env terrain blackout segments (pure torch, simulator-free)."""

from __future__ import annotations

import torch


class BlackoutSchedule:
    """Per-env blackout segments of ``duration_steps`` = (min, max) policy steps.

    ``start_prob`` is the per-step chance that a non-blacked-out env starts one;
    ``episode_prob`` blacks out a whole episode from its reset.
    """

    def __init__(self, num_envs: int, device, start_prob: float, duration_steps: tuple[int, int],
                 episode_prob: float):
        if not 0 <= start_prob <= 1 or not 0 <= episode_prob <= 1:
            raise ValueError("blackout probabilities must be in [0, 1]")
        low, high = duration_steps
        if not 1 <= low <= high:
            raise ValueError("duration_steps must satisfy 1 <= min <= max")
        self.start_prob, self.duration, self.episode_prob = start_prob, (low, high), episode_prob
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.whole_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_step = -1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.remaining[ids] = 0
        count = self.whole_episode[ids].numel()
        self.whole_episode[ids] = torch.rand(count, device=self.remaining.device) < self.episode_prob

    def mask(self, step: int) -> torch.Tensor:
        """True where the terrain group is blacked out; advances once per policy step."""
        if step != self.last_step:
            self.last_step = step
            self.remaining.sub_(1).clamp_(min=0)
            start = (self.remaining == 0) & (torch.rand_like(self.remaining, dtype=torch.float) < self.start_prob)
            lengths = torch.randint(self.duration[0], self.duration[1] + 1, self.remaining.shape,
                                    device=self.remaining.device)
            self.remaining[start] = lengths[start]
        return self.whole_episode | (self.remaining > 0)
