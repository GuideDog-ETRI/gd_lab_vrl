"""Regression checks for directional ghosts, reset cadence and the noise switch."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from gd_lab.core.camera_contract import CAMERA_NAMES, load_camera_contract
from gd_lab.core.camera_geometry import calibrated_resample, camera_rotation_matrix, mounted_camera_world_poses
from gd_lab.core.camera_timing import camera_refresh_mask
from gd_lab.mdp.camera_noise import augment_student_camera_frames
from gd_lab.mdp.platform_gap_noise import PlatformGapDepthGhost, PlatformGapDepthGhostCfg


def _ghost_fixture(depth_value=4.0):
    frames = torch.zeros(2, 4, 2, 9, 12)
    frames[:, :, 0] = (depth_value - 0.15) / 4.85
    frames[:, :, 1] = 0.37
    depth = torch.full((2, 4, 9, 12), depth_value)
    pos = torch.zeros(2, 4, 3)
    pos[..., 0], pos[..., 2] = 1.5, -depth_value
    rot = torch.zeros(2, 4, 4)
    rot[..., 0] = 1
    k = torch.eye(3).expand(2, 4, 3, 3).clone()
    k[..., 0, 0] = k[..., 1, 1] = 100
    k[..., 0, 2], k[..., 1, 2] = 6, 4
    return frames, (frames.clone(), depth, pos, rot, k)


def _forced_ghost(frames, mode):
    ghost = PlatformGapDepthGhost(PlatformGapDepthGhostCfg(probability=0.0))
    ghost._ensure_state(frames.shape, frames.device)
    ghost.remaining.fill_(3)
    ghost.mode.fill_(mode)
    ghost.center[..., 0], ghost.center[..., 1] = 6, 4
    ghost.radius.fill_(20)
    ghost.hole_depth.fill_(2.5)
    ghost.return_offset.fill_(0.25)
    return ghost


@pytest.mark.parametrize("depth", [0.2, 1.0, 4.0, 4.99])
@pytest.mark.parametrize("mode", [0, 1])
def test_ghost_direction_uses_clean_depth_and_preserves_teacher_ir_and_other_envs(depth, mode):
    frames, snapshot = _ghost_fixture(depth)
    clean = tuple(x.clone() for x in snapshot)
    # Deliberately contradict the clean input, as preceding generic noise can.
    frames[:, :, 0] = 0.0 if mode == 0 else 1.0
    original = frames.clone()
    ghost = _forced_ghost(frames, mode)
    out = ghost(frames, snapshot, torch.zeros(2, 3), torch.tensor([True, False]))
    metric = out[0, :, 0] * 4.85 + 0.15
    assert (metric > depth).all() if mode == 0 else (metric < depth).all()
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    assert torch.equal(out[:, :, 1], original[:, :, 1])
    assert torch.equal(out[1], original[1])
    assert torch.equal(frames, original)
    assert all(torch.equal(a, b) for a, b in zip(snapshot, clean, strict=True))


def test_ghost_rejects_far_sentinel_and_non_gap_pixels():
    frames, snapshot = _ghost_fixture(5.0)
    ghost = _forced_ghost(frames, 1)
    assert torch.equal(ghost(frames, snapshot, torch.zeros(2, 3), torch.ones(2, dtype=torch.bool)), frames)
    frames, snapshot = _ghost_fixture(1.0)
    snapshot[2][..., 0] = 4.0
    assert torch.equal(ghost(frames, snapshot, torch.zeros(2, 3), torch.ones(2, dtype=torch.bool)), frames)


def test_ghost_lifetime_parameters_and_partial_reset():
    frames, snapshot = _ghost_fixture(1.0)
    ghost = PlatformGapDepthGhost(PlatformGapDepthGhostCfg(probability=1.0, lifetime_steps=(3, 3)))
    active = torch.ones(2, dtype=torch.bool)
    ghost(frames, snapshot, torch.zeros(2, 3), active)
    names = ("center", "mode", "radius", "hole_depth", "return_offset")
    params = [getattr(ghost, key).clone() for key in names]
    for remaining in (1, 0):
        ghost(frames, snapshot, torch.zeros(2, 3), active)
        assert (ghost.remaining == remaining).all()
        assert all(torch.equal(getattr(ghost, key), value) for key, value in zip(names, params, strict=True))
    ghost(frames, snapshot, torch.zeros(2, 3), active)
    ghost.reset(torch.tensor([0]))
    assert not ghost.remaining[0].any() and (ghost.remaining[1] == 2).all()
    ghost.reset()
    assert not ghost.remaining.any()


def test_noise_off_bypasses_both_generic_and_gap_noise():
    frames, snapshot = _ghost_fixture()
    def forbidden(*args):
        raise AssertionError("noise disabled but gap ghost was invoked")
    result = augment_student_camera_frames(frames, snapshot, None, None, None, forbidden)
    assert result is frames


@pytest.mark.parametrize("profile", ["vendor_legacy", "vendor_new"])
def test_camera_mount_uses_live_translated_rotated_trunk(profile):
    contract = load_camera_contract(profile)
    root_pos = torch.tensor([[8., -56., 1.3], [-12., 30., 0.6]])
    root_quat = torch.tensor([[1., 0., 0., 0.], [0.70710678, 0., 0., 0.70710678]])
    pos, ros = mounted_camera_world_poses(root_pos, root_quat, contract)
    offsets = root_pos.new_tensor(contract.positions)
    expected_rotated = torch.stack((-offsets[:, 1], offsets[:, 0], offsets[:, 2]), -1)
    assert torch.allclose(pos[0], root_pos[0] + offsets)
    assert torch.allclose(pos[1], root_pos[1] + expected_rotated)
    gl = camera_rotation_matrix(root_pos.new_tensor(contract.quaternions_opengl))
    root = camera_rotation_matrix(root_quat)
    ros_conversion = torch.diag(torch.tensor([1., -1., -1.]))
    assert torch.allclose(camera_rotation_matrix(ros), root[:, None] @ gl[None] @ ros_conversion, atol=1e-6)


def test_canonical_snapshot_does_not_read_stale_usd_pose():
    path = Path(__file__).parents[1] / "src/gd_lab/mdp/camera_observations.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(torch=torch, CAMERA_NAMES=CAMERA_NAMES, calibrated_resample=calibrated_resample,
              mounted_camera_world_poses=mounted_camera_world_poses)
    exec(compile(tree, str(path), "exec"), ns)
    contract = load_camera_contract()
    body_pos = torch.tensor([[[0., 0., 0.], [8., -56., 1.3]]])
    body_quat = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    scene = {"robot": NS(body_names=["other", "trunk"], data=NS(body_pos_w=body_pos, body_quat_w=body_quat))}
    k = torch.tensor([[[contract.fx, 0., 40.], [0., contract.fx, 24.], [0., 0., 1.]]])
    for name in CAMERA_NAMES:
        # No pose fields: canonical projection must use live articulation data.
        data = NS(intrinsic_matrices=k, output={
            "distance_to_image_plane": torch.ones(1, 48, 80, 1),
            "rgb": torch.full((1, 48, 80, 3), 128, dtype=torch.uint8),
        })
        scene[name] = NS(data=data)
    snapshot = ns["canonical_camera_snapshot"](scene, contract)
    expected_pos, expected_quat = mounted_camera_world_poses(body_pos[:, 1], body_quat[:, 1], contract)
    assert snapshot[0].shape == (1, 4, 2, 45, 80)
    assert torch.allclose(snapshot[2], expected_pos) and torch.allclose(snapshot[3], expected_quat)
    assert torch.isfinite(snapshot[0]).all()


@pytest.mark.parametrize("reset_step", [1, 2, 3])
def test_partial_reset_rejoins_global_render_clock(reset_step):
    last = torch.full((2,), -1)
    calls = []
    for step in range(10):
        if step == reset_step:
            last[0] = -1
        refresh = camera_refresh_mask(last, step, 4)
        if refresh.any():
            assert step % 4 == 0 or step == reset_step
            calls.append((step, refresh.tolist()))
        last[refresh] = step
        assert not camera_refresh_mask(last, step, 4).any()
    assert calls == [(0, [True, True]), (reset_step, [True, False]), (4, [True, True]), (8, [True, True])]


def test_actual_observation_adapter_keeps_images_and_targets_on_same_clock():
    # Exercise production state/buffer code without starting Kit.
    path = Path(__file__).parents[1] / "src/gd_lab/mdp/camera_observations.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scan = NS(pos_w=torch.zeros(2, 3), ray_hits_w=torch.zeros(2, 187, 3))
    env = NS(num_envs=2, device="cpu", cfg=NS(camera_profile="test"), common_step_counter=0,
             scene={"height_scanner": NS(data=scan)})
    captures = []
    def capture(*args):
        step = env.common_step_counter
        assert step in (0, 1, 4, 8)
        captures.append(step)
        scan.ray_hits_w[..., 2] = -0.5 - step / 100
        return tuple(torch.full(shape, float(step)) for shape in
                     ((2, 4, 2, 3, 3), (2, 4, 3, 3), (2, 4, 3), (2, 4, 4), (2, 4, 3, 3)))
    ns = dict(torch=torch, ManagerTermBase=type("Base", (), {"__init__": lambda self, cfg, env: None}),
              camera_refresh_mask=camera_refresh_mask, canonical_camera_snapshot=capture,
              load_camera_contract=lambda profile: NS(period_steps=4, depth_clip=(0.15, 5.0)),
              camera_visible_points=lambda points, *args: torch.ones(points.shape[:2], dtype=torch.bool))
    exec(compile(tree, str(path), "exec"), ns)
    term = ns["CameraVisibleTerrain"](None, env)
    for step in range(10):
        env.common_step_counter = step
        if step == 1:
            term.reset(torch.tensor([0]))
        obs = term(env)
        stamps = env._vrl_camera_snapshot_steps
        assert torch.equal(env._vrl_camera_snapshot[0][:, 0, 0, 0, 0], stamps.float())
        assert torch.allclose(obs[:, 0], stamps.float() / 20, atol=1e-6)
    assert captures == [0, 1, 4, 8]
