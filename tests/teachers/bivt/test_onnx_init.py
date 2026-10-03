from __future__ import annotations

import torch
import torch.nn.functional as F

from gd_lab.teachers.bivt import onnx_init


def _mlp(x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    for index in (0, 2, 4, 6):
        x = F.linear(x, state[f"actor.{index}.weight"], state[f"actor.{index}.bias"])
        if index != 6:
            x = F.elu(x)
    return x


def test_onnx_actor_direct_and_code_weights_are_repacked(monkeypatch, tmp_path):
    torch.manual_seed(7)
    source = {
        "actor.0.weight": torch.randn(512, 65),
        "actor.0.bias": torch.randn(512),
        "actor.2.weight": torch.randn(256, 512),
        "actor.2.bias": torch.randn(256),
        "actor.4.weight": torch.randn(128, 256),
        "actor.4.bias": torch.randn(128),
        "actor.6.weight": torch.randn(12, 128),
        "actor.6.bias": torch.randn(12),
        "cenet_encoder.0.weight": torch.randn(128, 230),
        "cenet_encoder.0.bias": torch.randn(128),
        "cenet_encoder.2.weight": torch.randn(64, 128),
        "cenet_encoder.2.bias": torch.randn(64),
        "cenet_mean_velocity.weight": torch.randn(3, 64),
        "cenet_mean_velocity.bias": torch.randn(3),
        "cenet_mean_latent.weight": torch.randn(16, 64),
        "cenet_mean_latent.bias": torch.randn(16),
        "normalizer._mean": torch.randn(1, 230),
        "onnx::Div_124": torch.full((1, 230), 2.0),
    }
    target = {
        "actor.0.weight": torch.randn(512, 235),
        "actor.0.bias": torch.zeros(512),
        "actor.2.weight": torch.zeros(256, 512),
        "actor.2.bias": torch.zeros(256),
        "actor.4.weight": torch.zeros(128, 256),
        "actor.4.bias": torch.zeros(128),
        "actor.6.weight": torch.zeros(12, 128),
        "actor.6.bias": torch.zeros(12),
        "cenet.encoder.0.weight": torch.zeros(128, 230),
        "cenet.encoder.0.bias": torch.zeros(128),
        "cenet.encoder.2.weight": torch.zeros(64, 128),
        "cenet.encoder.2.bias": torch.zeros(64),
        "cenet.mean_velocity.weight": torch.zeros(3, 64),
        "cenet.mean_velocity.bias": torch.zeros(3),
        "cenet.mean_latent.weight": torch.zeros(16, 64),
        "cenet.mean_latent.bias": torch.zeros(16),
        "actor_obs_normalizer._mean": torch.zeros(1, 230),
        "actor_obs_normalizer._std": torch.ones(1, 230),
        "actor_obs_normalizer._var": torch.ones(1, 230),
        "actor_obs_normalizer.count": torch.zeros(1),
        "terrain_encoder.head.2.weight": torch.zeros(32, 64),
    }
    graph = {
        "inputs": {"direct_obs": [0, 46], "cenet_obs": [0, 230]},
        "outputs": {"actions": [0, 12], "z_t": [0, 19]},
    }
    monkeypatch.setattr(onnx_init, "_onnx_initializers", lambda _path: (source, graph))

    onnx_path = tmp_path / "fixture.onnx"
    onnx_path.write_bytes(b"test fixture")
    state, _ = onnx_init.onnx_to_bivt_state_dict(onnx_path, target)
    mapped = state["actor.0.weight"]
    assert torch.equal(mapped[:, :46], source["actor.0.weight"][:, :46])
    assert torch.count_nonzero(mapped[:, 46:184]) == 0
    assert torch.equal(mapped[:, 184:203], source["actor.0.weight"][:, 46:65])
    assert torch.count_nonzero(mapped[:, 203:]) == 0

    direct, code = torch.randn(5, 46), torch.randn(5, 19)
    other_frames, terrain = torch.randn(5, 138), torch.randn(5, 32)
    target_input = torch.cat((direct, other_frames, code, terrain), dim=-1)
    expected = _mlp(torch.cat((direct, code), dim=-1), source)
    actual = _mlp(target_input, state)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
