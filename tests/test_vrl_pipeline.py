"""CPU checks for the privileged teacher and camera student deployment path."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from gd_lab.deploy.export_student_vrl import export_student_vrl
from gd_lab.deploy.export_vrl import DreamwaqVrlDeployPolicy, export_policy_vrl
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic
from gd_lab.rl.perception import CameraPerceptionEncoder, height_discontinuity_metres

NUM_ENVS = 4
TERM_DIMS = [3, 3, 3, 12, 12, 12, 1]
ONE_STEP = sum(TERM_DIMS)
HISTORY = 5
POLICY_DIM = ONE_STEP * HISTORY
CRITIC_DIM = 298
HEIGHT_SCAN_START = 110
TERRAIN_DIM = 32


def _obs(num_envs: int = NUM_ENVS) -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(num_envs, POLICY_DIM), "critic": torch.randn(num_envs, CRITIC_DIM), "terrain": torch.randn(num_envs, 374)},
        batch_size=[num_envs],
    )


def _teacher(production: bool = False) -> DreamwaqVrlActorCritic:
    return DreamwaqVrlActorCritic(
        _obs(),
        {"policy": ["policy"], "critic": ["critic"]},
        12,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128] if production else [64, 32],
        critic_hidden_dims=[512, 256, 128] if production else [64, 32],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        history_length=HISTORY,
        policy_term_dims=TERM_DIMS,
        velocity_target_slice=(45, 48),
        actor_history_steps=4,
        cenet_encoder_hidden_dims=[128, 64] if production else [32, 16],
        cenet_decoder_hidden_dims=[64, 128] if production else [16, 32],
        height_scan_start=HEIGHT_SCAN_START,
        height_scan_grid_shape=(11, 17),
        terrain_latent_dim=TERRAIN_DIM,
    ).eval()


def _term_major_to_time_major(term_major: torch.Tensor) -> torch.Tensor:
    chunks = []
    offsets = []
    cursor = 0
    for dim in TERM_DIMS:
        offsets.append(cursor)
        cursor += dim * HISTORY
    for t in range(HISTORY):
        for dim, offset in zip(TERM_DIMS, offsets, strict=True):
            chunks.append(term_major[..., offset + t * dim : offset + (t + 1) * dim])
    return torch.cat(chunks, dim=-1)


def test_vrl_actor_uses_four_frames_and_external_latent():
    teacher = _teacher()
    obs = _obs()
    latent = torch.randn(NUM_ENVS, TERRAIN_DIM)
    expected_input = torch.cat((super(DreamwaqVrlActorCritic, teacher)._actor_input(obs, inference=True), latent), -1)
    assert expected_input.shape == (NUM_ENVS, 235)
    assert torch.equal(teacher.act_with_terrain_latent(obs, latent), teacher.actor(expected_input))


def test_vrl_deploy_matches_external_latent_inference():
    teacher = _teacher()
    term_major = torch.randn(NUM_ENVS, POLICY_DIM)
    obs = TensorDict(
        {"policy": term_major, "critic": torch.randn(NUM_ENVS, CRITIC_DIM), "terrain": torch.randn(NUM_ENVS, 374)}, batch_size=[NUM_ENVS]
    )
    direct = term_major[..., teacher.latest_idx]
    time_major = _term_major_to_time_major(term_major)
    latent = torch.randn(NUM_ENVS, TERRAIN_DIM)
    expected = teacher.act_with_terrain_latent(obs, latent)
    actual, code = DreamwaqVrlDeployPolicy(teacher, TERM_DIMS)(direct, time_major, latent)
    assert torch.allclose(actual, expected, atol=0.0)
    assert code.shape == (NUM_ENVS, 19)


def test_camera_student_bptt_and_export(tmp_path):
    student = CameraPerceptionEncoder().train()
    hidden = student.init_hidden(2, torch.device("cpu"))
    loss = torch.zeros(())
    for _ in range(3):
        latent, hidden = student(torch.randn(2, 4, 2, 45, 80), hidden)
        loss = loss + latent.square().mean() + student.hazard_head(hidden).square().mean()
    loss.backward()
    assert student.gru.weight_hh.grad is not None and torch.isfinite(student.gru.weight_hh.grad).all()

    actor_path = tmp_path / "policy_vrl.onnx"
    _, onnx_path = export_student_vrl(student, str(actor_path))
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime not installed")
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    outputs = session.run(
        None,
        {
            "frames": torch.zeros(1, 4, 2, 45, 80).numpy(),
            "hidden_in": torch.zeros(1, 64).numpy(),
        },
    )
    assert outputs[0].shape == (1, 32)
    assert outputs[1].shape == (1, 64)
    hidden_pt = torch.randn(1, 64)
    hidden_onnx = hidden_pt.numpy()
    for _ in range(3):
        frames = torch.rand(1, 4, 2, 45, 80)
        with torch.no_grad():
            latent_pt, hidden_pt = student(frames, hidden_pt)
        latent_onnx, hidden_onnx = session.run(None, {"frames": frames.numpy(), "hidden_in": hidden_onnx})
        assert torch.allclose(latent_pt, torch.from_numpy(latent_onnx), atol=1e-5)
        assert torch.allclose(hidden_pt, torch.from_numpy(hidden_onnx), atol=1e-5)


def test_vrl_ppo_replay_is_deterministic_and_gradient_routes_are_separate():
    teacher, obs = _teacher(), _obs()
    teacher.cenet.inject_gt.zero_()
    first = teacher._actor_input(obs, inference=False)
    second = teacher._actor_input(obs, inference=False)
    assert torch.equal(first, second)
    teacher.actor(first).square().mean().backward()
    assert all(p.grad is None for p in teacher.cenet.parameters())
    grads = [p.grad for p in teacher.terrain_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    teacher.cenet.inject_gt.fill_(1)
    injected = teacher._actor_input(obs, inference=False)
    assert torch.equal(injected[:, 184:187], teacher.velocity_target(obs))
    assert torch.equal(teacher._actor_input(obs, inference=True), first)


def test_hazard_target_undoes_observation_scale():
    grid = torch.zeros(2, 11, 17)
    grid[0, :, 8:] = 0.65 * 5
    result = height_discontinuity_metres(grid.flatten(1), 5)
    assert torch.allclose(result, torch.tensor([0.65, 0.0]))
    with pytest.raises(ValueError):
        height_discontinuity_metres(grid.flatten(1), 0)


@pytest.mark.parametrize("symmetry", [False, True])
def test_vrl_actual_ppo_aux_update_and_resume(tmp_path, symmetry):
    from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC
    from gd_lab.rl import DreamwaqRunner

    spec = importlib.util.spec_from_file_location("vrl_dummy_helpers", Path(__file__).with_name("test_rl_dreamwaq.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    config = helpers._train_cfg()
    config["policy"].update(
        class_name="gd_lab.rl.actor_critic_vrl:DreamwaqVrlActorCritic",
        height_scan_start=110, height_scan_grid_shape=(11, 17), terrain_latent_dim=32,
    )
    if symmetry:
        config["algorithm"]["symmetry_cfg"] = {
            "use_data_augmentation": True, "use_mirror_loss": False, "mirror_loss_coeff": 0.0,
            "data_augmentation_func": "gd_lab.methods.dreamwaq.vrl_symmetry:mirror_vrl_observations",
        }

    def environment():
        env = helpers._DummyEnv()
        env.unwrapped = env
        env.cfg = SimpleNamespace(scene=SimpleNamespace(height_scanner=SimpleNamespace(
            pattern_cfg=SimpleNamespace(size=(1.6, 1.0), resolution=0.1),
        )))
        joints = [f"{leg}_{joint}" for joint in ("HIP", "THIGH", "KNEE") for leg in ("FL", "FR", "RL", "RR")]
        env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_names=joints))}
        env.observation_manager = SimpleNamespace(
            active_terms={group: [term.name for term in spec.terms] for group, spec in DREAMWAQ_SPEC.groups.items()},
            group_obs_dim={"policy": (230,), "critic": (298,)},
        )
        return env

    runner = DreamwaqRunner(environment(), copy.deepcopy(config), log_dir=str(tmp_path / "run"), device="cpu")
    before = next(runner.alg.policy.terrain_encoder.parameters()).detach().clone()
    runner.learn(num_learning_iterations=2)
    assert not torch.equal(before, next(runner.alg.policy.terrain_encoder.parameters()))
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    path = str(tmp_path / "teacher.pt")
    runner.save(path)
    fresh = DreamwaqRunner(environment(), copy.deepcopy(config), log_dir=str(tmp_path / "resume"), device="cpu")
    fresh.load(path)
    obs = _obs()
    assert torch.equal(runner.alg.policy.act_inference(obs), fresh.alg.policy.act_inference(obs))
    fresh.learn(num_learning_iterations=1)


def test_production_layer_tensor_shapes():
    teacher = _teacher(production=True)
    seen = {}

    def capture(name):
        def hook(module, inputs, output):
            seen[name] = tuple(output.shape)
        return hook

    handles = [m.register_forward_hook(capture(n)) for n, m in teacher.named_modules()
               if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d))]
    obs = _obs(2)
    assert teacher.act_inference(obs).shape == (2, 12)
    assert teacher.evaluate(obs).shape == (2, 1)
    expected = {
        "cenet.encoder.0": (2, 128), "cenet.encoder.2": (2, 64),
        "cenet.mean_latent": (2, 16), "cenet.logvar_latent": (2, 16), "cenet.mean_velocity": (2, 3),
        "cenet.decoder.0": (2, 64), "cenet.decoder.2": (2, 128), "cenet.decoder.4": (2, 46),
        "terrain_encoder.net.0": (2, 8, 11, 17), "terrain_encoder.net.2": (2, 16, 11, 17),
        "terrain_encoder.net.4": (2, 16, 5, 8), "terrain_encoder.head.0": (2, 64),
        "terrain_encoder.head.2": (2, 32),
        "actor.0": (2, 512), "actor.2": (2, 256), "actor.4": (2, 128), "actor.6": (2, 12),
        "critic.0": (2, 512), "critic.2": (2, 256), "critic.4": (2, 128), "critic.6": (2, 1),
    }
    for name, shape in expected.items():
        assert seen[name] == shape, name
    for handle in handles:
        handle.remove()

    student = CameraPerceptionEncoder()
    seen.clear()
    handles = [m.register_forward_hook(capture(n)) for n, m in student.named_modules()
               if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d, torch.nn.GRUCell))]
    latent, hidden = student(torch.zeros(2, 4, 2, 45, 80), torch.zeros(2, 64))
    student.hazard_head(hidden)
    expected = {
        "camera_cnn.net.0": (8, 16, 23, 40), "camera_cnn.net.2": (8, 32, 12, 20),
        "camera_cnn.net.4": (8, 32, 6, 10), "camera_cnn.net.6": (8, 32, 3, 5),
        "camera_cnn.proj": (8, 32), "fuse.0": (2, 64), "gru": (2, 64), "head.0": (2, 32),
        "hazard_head.0": (2, 16), "hazard_head.2": (2, 1),
    }
    assert latent.shape == (2, 32)
    for name, shape in expected.items():
        assert seen[name] == shape, name
    for handle in handles:
        handle.remove()


def test_vrl_actor_onnx_matches_torch_for_dynamic_batches(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    teacher = _teacher()
    jit_path, onnx_path = export_policy_vrl(teacher, str(tmp_path), TERM_DIMS, 32)
    session = ort.InferenceSession(onnx_path)
    jit = torch.jit.load(jit_path)
    module = DreamwaqVrlDeployPolicy(teacher, TERM_DIMS)
    for batch in (1, 3):
        args = (torch.randn(batch, 46), torch.randn(batch, 230), torch.randn(batch, 32))
        with torch.no_grad():
            expected = module(*args)
            scripted = jit(*args)
        actual = session.run(None, {key: value.numpy() for key, value in zip(
            ("direct_obs", "cenet_obs", "terrain_latent"), args, strict=True,
        )})
        for pt, ts, onnx in zip(expected, scripted, actual, strict=True):
            assert torch.allclose(pt, ts, atol=1e-6)
            assert torch.allclose(pt, torch.from_numpy(onnx), atol=1e-5)


def test_hazard_target_uses_visibility_mask():
    grid = torch.zeros(1, 11, 17)
    grid[:, :, 8:] = 3.25
    visible = torch.ones_like(grid, dtype=torch.bool)
    visible[:, :, 8:] = False
    assert torch.equal(height_discontinuity_metres(grid.flatten(1), 5, valid_mask=visible), torch.zeros(1))


def test_camera_projection_round_trip_with_rotated_mount():
    import math

    from gd_lab.core.camera_geometry import camera_visible_points, depth_pixels_world

    quat = torch.tensor([[[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]]])
    position = torch.zeros(1, 1, 3)
    intrinsic = torch.tensor([[[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]])
    depth = torch.ones(1, 1, 5, 5)
    point = depth_pixels_world(depth, position, quat, intrinsic)[:, :, 2, 2]
    assert camera_visible_points(point, position[:, 0], quat[:, 0], intrinsic[:, 0], depth[:, 0]).item()
