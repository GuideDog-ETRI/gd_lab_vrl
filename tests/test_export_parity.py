"""The exported deploy module must match act_inference bit-for-bit."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from tensordict import TensorDict

from gd_lab.deploy.export import DreamwaqDeployPolicy, export_policy
from gd_lab.rl import DreamwaqActorCritic

NUM_ENVS = 16
TERM_DIMS = [3, 3, 3, 12, 12, 12, 1]
ONE_STEP = sum(TERM_DIMS)
POLICY_DIM = ONE_STEP * 5
CRITIC_DIM = 298


def _make_policy() -> DreamwaqActorCritic:
    obs = TensorDict(
        {"policy": torch.randn(NUM_ENVS, POLICY_DIM), "critic": torch.randn(NUM_ENVS, CRITIC_DIM)},
        batch_size=[NUM_ENVS],
    )
    policy = DreamwaqActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        12,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        history_length=5,
        policy_term_dims=list(TERM_DIMS),
        velocity_target_slice=(45, 48),
        cenet_encoder_hidden_dims=[32, 16],
        cenet_decoder_hidden_dims=[16, 32],
    )
    # Drive the normalizer to non-trivial statistics so parity is meaningful.
    policy.train()
    for _ in range(5):
        sample = TensorDict(
            {
                "policy": torch.randn(64, POLICY_DIM) * 3.0 + 1.0,
                "critic": torch.randn(64, CRITIC_DIM),
            },
            batch_size=[64],
        )
        policy.update_normalization(sample)
    policy.eval()
    return policy


def test_deploy_module_matches_act_inference():
    policy = _make_policy()
    raw = torch.randn(NUM_ENVS, POLICY_DIM)
    obs = TensorDict({"policy": raw, "critic": torch.randn(NUM_ENVS, CRITIC_DIM)}, batch_size=[NUM_ENVS])
    expected = policy.act_inference(obs)
    actions, velocity = DreamwaqDeployPolicy(policy)(raw)
    assert torch.allclose(actions, expected, atol=0.0)
    assert velocity.shape == (NUM_ENVS, 3)


def test_jit_and_onnx_export(tmp_path):
    policy = _make_policy()
    jit_path, onnx_path = export_policy(policy, str(tmp_path))

    raw = torch.randn(4, POLICY_DIM)
    obs = TensorDict({"policy": raw, "critic": torch.randn(4, CRITIC_DIM)}, batch_size=[4])
    expected = policy.act_inference(obs)

    jit_actions, _ = torch.jit.load(jit_path)(raw)
    assert torch.allclose(jit_actions, expected, atol=1e-6)

    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime not installed")
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    (ort_actions, _) = session.run(None, {"obs_history": raw.numpy()})
    assert torch.allclose(torch.from_numpy(ort_actions), expected, atol=1e-5)
