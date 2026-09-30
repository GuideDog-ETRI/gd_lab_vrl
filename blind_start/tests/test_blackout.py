"""Blackout schedule and zero-latent gating (CPU only, no simulator)."""

from __future__ import annotations

import torch
from gd_lab_blind_start.actor_critic import DreamwaqVrlGatedActorCritic
from gd_lab_blind_start.blackout import BlackoutSchedule
from tensordict import TensorDict

from test_blind_init import COMMON, GROUPS, _obs


def _schedule(**kwargs):
    return BlackoutSchedule(**kwargs)


def test_blackout_segments_advance_once_per_step_and_expire():
    torch.manual_seed(0)
    s = _schedule(num_envs=64, device="cpu", start_prob=1.0, duration_steps=(3, 3), episode_prob=0.0)
    s.reset()
    first = s.mask(1)
    assert first.all()
    assert torch.equal(s.mask(1), first)  # same step: no advance
    assert s.mask(2).all() and s.mask(3).all()
    s.start_prob = 0.0
    assert not s.mask(4).any()


def test_episode_blackout_is_resampled_on_reset():
    s = _schedule(num_envs=8, device="cpu", start_prob=0.0, duration_steps=(1, 1), episode_prob=1.0)
    s.reset()
    assert s.mask(1).all()
    s.episode_prob = 0.0
    s.reset(torch.tensor([0, 1]))
    assert s.mask(2).tolist() == [False, False] + [True] * 6


def test_gated_latent_is_zero_only_without_valid_cells():
    torch.manual_seed(0)
    policy = DreamwaqVrlGatedActorCritic(
        _obs(), GROUPS, 12, height_scan_start=110, height_scan_grid_shape=(11, 17),
        terrain_latent_dim=32, **COMMON,
    ).eval()
    obs = _obs(3)
    terrain = torch.randn(3, 374)
    terrain[:, 187:] = 1.0
    terrain[0] = 0.0  # blacked out
    obs = TensorDict({**obs.to_dict(), "terrain": terrain}, batch_size=[3])
    latent = policy.terrain_latent(obs)
    assert torch.count_nonzero(latent[0]) == 0
    assert torch.count_nonzero(latent[1:]) > 0
