"""The exported deploy module must match act_inference bit-for-bit."""

from __future__ import annotations

import importlib.util
import json

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


def _deploy_context() -> dict:
    """The shape ``capture_context`` produces for the blind DreamWaQ task."""
    legs = ("FL", "FR", "HL", "HR")
    joint_names = [f"{leg}_{j}" for j in ("HIP", "THIGH", "KNEE") for leg in legs]
    default_pos = [0.0] * 4 + [0.76] * 4 + [-1.45] * 4
    kp = [88.1367] * 8 + [102.2177] * 4
    kd = [1.9919] * 8 + [1.9932] * 4
    terms = [
        ("base_ang_vel", "angular_velocity", 3, [{"op": "scale", "values": [0.25] * 3}]),
        ("projected_gravity", "gravity", 3, []),
        ("velocity_commands", "command", 3, [{"op": "scale", "values": [2.0, 2.0, 0.25]}]),
        ("joint_pos", "joint_position_rel", 12, []),
        ("joint_vel", "joint_velocity", 12, [{"op": "scale", "values": [0.05] * 12}]),
        ("actions", "previous_action", 12, []),
        ("payload", "payload", 1, [{"op": "scale", "values": [0.2]}]),
    ]
    return {
        "schema_version": 1,
        "robot": "RBQ10",
        "policy_dt": 0.02,
        "joint_names": joint_names,
        "default_joint_pos": default_pos,
        "action": {
            "type": "joint_position",
            "scale": [0.25] * 12,
            "offset": default_pos,
            "clip": None,
            "previous_action": "clipped",
            "soft_margin_deg": 5.0,
            "gains": {"kp": kp, "kd": kd},
        },
        "history_initialization": "repeat_first",
        "terms": [
            {"name": n, "source": s, "size": d, "history": 5, "transforms": t} for n, s, d, t in terms
        ],
        "actor_history_steps": 4,
        "provenance": {"policy_class": "DreamwaqActorCritic", "source": "test"},
    }


def test_onnx_metadata_matches_deploy_parser(tmp_path):
    """Assert what gd_rbq10_deploy pilot/src/PolicyRuntime.cpp requires of the embedded contract."""
    import onnx

    policy = _make_policy()
    _, onnx_path = export_policy(policy, str(tmp_path), deploy_context=_deploy_context())

    props = {p.key: p.value for p in onnx.load(onnx_path).metadata_props}
    spec = json.loads(props["camel.policy.v1"])
    assert json.loads((tmp_path / "deploy.json").read_text()) == spec

    assert spec["schema_version"] == 1 and spec["robot"] == "RBQ10"
    assert spec["normalization"] == "in_graph"
    assert spec["history_initialization"] in ("zeros", "repeat_first")
    ticks = spec["policy_dt"] / 0.002  # RlWalker runs at 500 Hz
    assert 0.002 <= spec["policy_dt"] <= 0.1 and abs(ticks - round(ticks)) < 1e-5
    expected_joints = {f"{leg}_{j}" for leg in ("FL", "FR", "HL", "HR") for j in ("HIP", "THIGH", "KNEE")}
    assert len(spec["joint_names"]) == 12 and set(spec["joint_names"]) == expected_joints
    assert len(spec["default_joint_pos"]) == 12

    action = spec["action"]
    assert action["type"] == "joint_position" and action["previous_action"] == "clipped"
    assert "clip" in action and action["clip"] is None
    assert len(action["scale"]) == 12 and len(action["offset"]) == 12
    assert all(0 <= v <= 200 for v in action["gains"]["kp"]) and len(action["gains"]["kp"]) == 12
    assert all(0 <= v <= 10 for v in action["gains"]["kd"]) and len(action["gains"]["kd"]) == 12

    fixed = {
        "angular_velocity": 3,
        "gravity": 3,
        "command": 3,
        "joint_position_rel": 12,
        "joint_velocity": 12,
        "previous_action": 12,
        "payload": 1,
    }
    for t in spec["terms"]:
        assert t["size"] == fixed[t["source"]]
        assert all(tr["op"] in ("clip", "scale", "offset") for tr in t["transforms"])
    assert len({t["name"] for t in spec["terms"]}) == len(spec["terms"])

    (inp,) = spec["inputs"]
    assert inp["dtype"] == "float32" and inp["role"] == "observation" and inp["layout"] == "term_major"
    assert inp["shape"] == [1, sum(t["size"] * t["history"] for t in spec["terms"])] == [1, POLICY_DIM]
    assert inp["terms"] == [t["name"] for t in spec["terms"]]
    roles = [o["role"] for o in spec["outputs"]]
    assert roles.count("action") == 1
    assert next(o for o in spec["outputs"] if o["role"] == "action")["shape"] == [1, 12]
    assert spec["states"] == []
