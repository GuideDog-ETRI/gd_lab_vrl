"""The offline entry point must use the saved architecture without importing IsaacSim."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from tensordict import TensorDict

from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC, HISTORY_LENGTH, POLICY_OBS_DIM, VELOCITY_TARGET_SLICE
from gd_lab.rl.actor_critic import DreamwaqActorCritic


@pytest.mark.parametrize("moved_checkpoint", [False, True])
def test_offline_export_uses_saved_policy_config(tmp_path, moved_checkpoint):
    policy_cfg = dict(
        history_length=HISTORY_LENGTH,
        policy_term_dims=[term.dim for term in DREAMWAQ_SPEC.policy.terms],
        velocity_target_slice=VELOCITY_TARGET_SLICE,
        actor_history_steps=3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        cenet_latent_dim=4,
        cenet_encoder_hidden_dims=[16],
        cenet_decoder_hidden_dims=[16],
    )
    obs = TensorDict(
        {"policy": torch.randn(2, POLICY_OBS_DIM), "critic": torch.randn(2, DREAMWAQ_SPEC.critic.resolve(height_scan=187).total)},
        batch_size=[2],
    )
    policy = DreamwaqActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **policy_cfg).eval()
    checkpoint = tmp_path / "model_2.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    agent_path = tmp_path / ("saved_agent.yaml" if moved_checkpoint else "params/agent.yaml")
    agent_path.parent.mkdir(exist_ok=True)
    # Match IsaacLab's tuple tag for velocity_target_slice.
    agent_path.write_text(yaml.dump({"policy": {"class_name": "gd_lab.rl.actor_critic:DreamwaqActorCritic", **policy_cfg}}))
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / "scripts/export.py"), str(checkpoint), "--no-metadata"]
    if moved_checkpoint:
        command += ["--agent-config", str(agent_path)]
    result = subprocess.run(
        command, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")},
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "exported/policy.onnx").is_file()
    exported = torch.jit.load(str(tmp_path / "exported/policy.pt"))
    torch.testing.assert_close(exported(obs["policy"])[0], policy.act_inference(obs), atol=1e-6, rtol=1e-5)
