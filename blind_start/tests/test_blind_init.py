"""Blind -> VRL warm start: identical actions at iteration 0, strict shape checks."""

from __future__ import annotations

import pytest
import torch
from gd_lab.rl.actor_critic import DreamwaqActorCritic
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic
from gd_lab_blind_start.blind_init import blind_to_vrl_state_dict, load_blind_checkpoint
from tensordict import TensorDict

TERM_DIMS = [3, 3, 3, 12, 12, 12, 1]
HISTORY = 5
COMMON = dict(
    actor_obs_normalization=True, critic_obs_normalization=True,
    actor_hidden_dims=[64, 32], critic_hidden_dims=[64, 32], activation="elu",
    init_noise_std=1.0, noise_std_type="scalar", history_length=HISTORY,
    policy_term_dims=TERM_DIMS, velocity_target_slice=(45, 48), actor_history_steps=4,
    cenet_encoder_hidden_dims=[32, 16], cenet_decoder_hidden_dims=[16, 32],
)
GROUPS = {"policy": ["policy"], "critic": ["critic"]}


def _obs(n: int = 4) -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(n, sum(TERM_DIMS) * HISTORY), "critic": torch.randn(n, 298),
         "terrain": torch.randn(n, 374)},
        batch_size=[n],
    )


def _pair():
    torch.manual_seed(0)
    blind = DreamwaqActorCritic(_obs(), GROUPS, 12, **COMMON).eval()
    vrl = DreamwaqVrlActorCritic(
        _obs(), GROUPS, 12, height_scan_start=110, height_scan_grid_shape=(11, 17),
        terrain_latent_dim=32, **COMMON,
    ).eval()
    return blind, vrl


def test_warm_start_reproduces_blind_actions(tmp_path):
    blind, vrl = _pair()
    obs = _obs()
    blind.update_normalization(obs)
    path = tmp_path / "model_1.pt"
    torch.save({"model_state_dict": blind.state_dict(), "infos": {"gd_lab": {"learning_rate": 3e-4}}}, path)

    infos = load_blind_checkpoint(vrl, str(path))

    assert infos["learning_rate"] == 3e-4
    with torch.no_grad():
        assert torch.allclose(vrl.act_inference(obs), blind.act_inference(obs), atol=1e-6)
        assert torch.allclose(vrl.evaluate(obs), blind.evaluate(obs), atol=1e-6)
    assert torch.count_nonzero(vrl.actor[0].weight[:, -32:]) == 0


def test_rejects_mismatched_blind_layout():
    blind, vrl = _pair()
    state = blind.state_dict()
    state["critic.0.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        blind_to_vrl_state_dict(state, vrl.state_dict())
