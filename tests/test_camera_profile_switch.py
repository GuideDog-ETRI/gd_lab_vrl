"""Ground-view camera calibration = RBQ SDK values (claude_handoff/CLAUDE_CAMERA_SWITCH_PROPOSAL.md).

CPU only. Visibility checks use the repository's own raycast_visibility with an analytic flat
floor, a level trunk at 0.52 m and NO leg/trunk occlusion (an upper bound, fixed conditions).
"""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gd_lab.core import camera_contract as cc
from gd_lab.core.camera_geometry import mounted_camera_world_poses
from gd_lab.students.gavd.model import GridAttentionStudent, camera_ray_buffers

ROOT = Path(__file__).parents[1]

# rbq_simulator/rbq_mujoco/resources/model/rbq/rbq.xml, BT0..BT3 bodies: identical in RBQ v1.19.47, v1.20.0,
# public main and the nightly builds 41f6fac6 (2026-09-09) and 5974087c (2026-10-04). quat = wxyz; MuJoCo
# cameras look along -z with +y up (OpenGL convention).
SDK_BT = (
    ((0.364, 0, -0.024919), (0, -0.1736482, 0, 0.9848078)),
    ((0.26097, 0, -0.04582), (0, 0.1218693, 0, 0.9925462)),
    ((-0.19515, 0.0065, -0.0465), (0, 0, 0, 1)),
    ((-0.352082, -0.000011, -0.018938), (0.9848078, 0, 0.1736482, 0)),
)
# Official manual, Coordinate Frames (TF) > Ground-view cameras (D430): base->camera rotation, ROS optical axes.
MANUAL_R = (
    ((-0.939693, 0, 0.342020), (0, 1, 0), (-0.342020, 0, -0.939693)),
    ((-0.970296, 0, -0.241922), (0, 1, 0), (0.241922, 0, -0.970296)),
    ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    ((0.861341, -0.0909518, -0.499820), (-0.105042, -0.9944680, -5.69736e-05), (-0.497050, 0.0525513, -0.866129)),
)


def _rotation(quat):
    q = torch.tensor(quat, dtype=torch.float64)
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], dtype=torch.float64)


def _elevations_deg(contract):
    """Optical-axis elevation of each camera in the trunk frame (negative = looking down)."""
    return [math.degrees(math.asin(float(-_rotation(q)[2, 2]))) for q in contract.quaternions_opengl]


def test_default_profile_is_the_sdk_calibration():
    assert cc.DEFAULT_CAMERA_PROFILE == "vendor_new"
    assert cc.load_camera_contract().profile == "vendor_new"
    assert GridAttentionStudent().camera_profile == "vendor_new"


def test_vendor_new_is_exactly_the_sdk_rbq_xml():
    c = cc.load_camera_contract("vendor_new")
    assert tuple(tuple(p) for p in c.positions) == tuple(tuple(map(float, p)) for p, _ in SDK_BT)
    assert tuple(tuple(q) for q in c.quaternions_opengl) == tuple(tuple(map(float, q)) for _, q in SDK_BT)


def test_ground_view_cameras_look_down_and_legacy_does_not():
    assert all(e < -45 for e in _elevations_deg(cc.load_camera_contract("vendor_new")))
    legacy = _elevations_deg(cc.load_camera_contract("vendor_legacy"))
    assert max(legacy) > 0  # cam0/cam3 look upward: the reason legacy is load/evaluate-only


def test_sdk_rotations_agree_with_the_manual_except_the_documented_cam3_gap():
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))  # ROS optical -> OpenGL camera
    c = cc.load_camera_contract("vendor_new")
    angles = []
    for quat, manual in zip(c.quaternions_opengl, MANUAL_R, strict=True):
        relative = (torch.tensor(manual, dtype=torch.float64) @ flip).T @ _rotation(quat)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, (float(torch.trace(relative)) - 1) / 2)))))
    assert max(angles[:3]) < 0.1
    assert 10 < angles[3] < 13  # cam3: SDK -70 deg vs manual -60 deg (+ small roll/yaw), asked to Rainbow


def _visibility(profile):
    spec = importlib.util.spec_from_file_location("rv", ROOT / "src/gd_lab/teachers/bivt/raycast_visibility.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    c = cc.load_camera_contract(profile)

    def flat_floor(starts, directions, max_dist):
        t = -starts[:, 2] / directions[:, 2]
        return torch.where((directions[:, 2] < 0) & (t > 0) & (t < max_dist), t, torch.full_like(t, float("inf")))

    ys, xs = torch.meshgrid(torch.linspace(-.5, .5, 11), torch.linspace(-.8, .8, 17), indexing="ij")
    points = torch.stack((xs, ys, torch.zeros_like(xs)), -1).reshape(1, 187, 3)
    pos, rot = mounted_camera_world_poses(torch.tensor([[0.0, 0.0, 0.52]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), c)
    k = rv.contract_intrinsic(c, "cpu")
    visible = torch.zeros(1, 187, dtype=torch.bool)
    for cam in range(4):
        visible |= rv.raycast_visible_points(points, pos[:, cam], rot[:, cam], k, (c.height, c.width), c.depth_clip,
                                             flat_floor)
    grid = visible.reshape(11, 17)
    hind = grid[:, (xs[0] > -0.45) & (xs[0] < -0.15)]
    return visible.float().mean().item(), hind.float().mean().item()


def test_teacher_grid_visibility_regression_on_flat_ground():
    union, hind = _visibility("vendor_new")
    assert union > 0.5 and hind > 0.5  # 55.6% of the 11x17 grid, 64% of the hind-foot band (upper bound)
    legacy_union, legacy_hind = _visibility("vendor_legacy")
    assert legacy_union < 0.2 and legacy_hind == 0.0


def test_legacy_training_needs_an_explicit_opt_in():
    with pytest.raises(ValueError):
        cc.require_training_camera_profile("vendor_legacy", environ={})
    cc.require_training_camera_profile("vendor_legacy", environ={cc.ALLOW_LEGACY_CAMERA_ENV: "1"})
    cc.require_training_camera_profile("vendor_new", environ={})


def test_manifest_records_frame_convention_and_source():
    manifest = cc.load_camera_contract("vendor_new").manifest()
    assert manifest["transform"] == "camera_to_trunk" and "opengl" in manifest["camera_convention"]
    assert "5974087c" in manifest["source"] and "814e6d4d" in cc.load_camera_contract("vendor_legacy").manifest()["source"]


def test_student_geometry_follows_the_profile_and_every_load_is_verified():
    new, legacy = GridAttentionStudent("vendor_new"), GridAttentionStudent("vendor_legacy")
    assert not torch.allclose(new.rays, legacy.rays)
    with pytest.raises(RuntimeError):
        GridAttentionStudent("vendor_new").load_state_dict(legacy.state_dict())  # legacy checkpoint -> new model
    GridAttentionStudent("vendor_new").load_state_dict(new.state_dict())
    GridAttentionStudent("vendor_legacy").load_state_dict(legacy.state_dict())  # explicit legacy evaluation
    broken = new.state_dict()
    broken["mounts"] = broken["mounts"] * float("nan")
    with pytest.raises(RuntimeError):
        GridAttentionStudent("vendor_new").load_state_dict(broken)


def test_ray_geometry_includes_the_intrinsics_not_only_the_mounts():
    new, legacy = cc.load_camera_contract("vendor_new"), cc.load_camera_contract("vendor_legacy")
    assert new.sensor_size_m != legacy.sensor_size_m  # the profiles also differ in sensor size -> fx/fy
    same_mounts_other_sensor = replace(new, sensor_size_m=legacy.sensor_size_m)
    assert not torch.allclose(camera_ray_buffers(new)[0], camera_ray_buffers(same_mounts_other_sensor)[0])


@pytest.mark.parametrize("script", ["scripts/distill_student.py", "scripts/train_bivt.py",
                                    "archive/cvtt/scripts/train_cvtt.py", "archive/bavrl/scripts/train_bavrl.py"])
def test_training_entry_points_refuse_a_legacy_start(script):
    source = (ROOT / script).read_text()
    assert "require_training_camera_profile(env_cfg.camera_profile)" in source


def test_distillation_resume_checks_the_saved_camera_contract():
    source = (ROOT / "scripts/distill_student.py").read_text()
    resume = source[source.index("if args_cli.student_resume:"):source.index("camera_noise_cfg =")]
    assert 'check_checkpoint_camera_contract(saved.get(\'camera_contract\'), camera_contract, purpose="resume")' in resume
    assert resume.index("check_checkpoint_camera_contract") < resume.index("student.load_state_dict")


def test_a_parent_module_load_is_verified_too():
    """BAVRL-style nesting: PyTorch only calls the child's _load_from_state_dict, so an override would be skipped."""
    class Parent(torch.nn.Module):
        def __init__(self, profile):
            super().__init__()
            self.vision = GridAttentionStudent(profile)

    with pytest.raises(RuntimeError):
        Parent("vendor_new").load_state_dict(Parent("vendor_legacy").state_dict())
    Parent("vendor_legacy").load_state_dict(Parent("vendor_legacy").state_dict())


def test_bavrl_checkpoint_with_another_camera_geometry_is_refused():
    pytest.importorskip("yaml")
    from gd_lab.residuals.bavrl import BAVRL, ResidualConfig
    from tests.test_bavrl import make_model

    new = make_model()
    assert new.vision.camera_profile == "vendor_new"  # the default followed the SDK switch
    legacy = BAVRL(new.teacher, ResidualConfig(camera_profile="vendor_legacy"))
    with pytest.raises(RuntimeError):
        new.load_state_dict(legacy.state_dict(), strict=True)


def test_contract_identity_ignores_metadata_but_not_geometry_order_or_timing():
    current = cc.camera_contract_for_policy("vendor_new", 0.01)
    old_style = {k: v for k, v in current.manifest().items() if k not in ("transform", "camera_convention", "source")}
    assert cc.same_camera_contract(old_style, current)  # a checkpoint saved before the metadata fields still matches
    for key, value in (("quaternions_opengl", cc.load_camera_contract("vendor_legacy").quaternions_opengl),
                       ("camera_names", list(reversed(cc.CAMERA_NAMES))), ("depth_semantics", "radial_metres"),
                       ("period_steps", current.period_steps + 1)):
        assert not cc.same_camera_contract({**old_style, key: value}, current)
    assert cc.same_camera_contract({**old_style, "period_steps": 99}, current, timing=False)


def test_evaluation_scripts_do_not_compare_whole_manifests():
    play = (ROOT / "scripts/play_student.py").read_text()
    assert 'check_checkpoint_camera_contract(ckpt.get("camera_contract"), camera_contract, purpose="evaluation")' in play
    assert "[WARN] Legacy student checkpoint has no camera calibration metadata" not in play  # no silent continue
    assert "same_camera_contract(" in (ROOT / "scripts/compare_arm4_stand.py").read_text()


# ---------------------------------------------------------------- executed fail-closed checks
def test_checkpoint_contract_check_refuses_missing_and_mismatched_contracts():
    current = cc.camera_contract_for_policy("vendor_new", 0.01)
    with pytest.raises(ValueError, match="no camera contract"):
        cc.check_checkpoint_camera_contract(None, current, purpose="evaluation", environ={})
    assert cc.check_checkpoint_camera_contract(None, current, environ={cc.ALLOW_MISSING_CAMERA_CONTRACT_ENV: "1"}) is None
    with pytest.raises(ValueError, match="no camera contract"):  # the legacy-training opt-in is NOT enough
        cc.check_checkpoint_camera_contract(None, current, environ={cc.ALLOW_LEGACY_CAMERA_ENV: "1"})
    legacy = cc.camera_contract_for_policy("vendor_legacy", 0.01).manifest()
    with pytest.raises(ValueError, match="differs"):
        cc.check_checkpoint_camera_contract(legacy, current, environ={cc.ALLOW_MISSING_CAMERA_CONTRACT_ENV: "1",
                                                                       cc.ALLOW_LEGACY_CAMERA_ENV: "1"})
    assert cc.check_checkpoint_camera_contract(current.manifest(), current, environ={}) is not None


def _export_module():
    spec = importlib.util.spec_from_file_location("export_student_under_test", ROOT / "scripts/export_student.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rvld_checkpoint(profile="vendor_new", **override):
    from gd_lab.students.rvld.model import CameraPerceptionEncoder

    ckpt = {"model": CameraPerceptionEncoder().state_dict(), "student_arch": "cnn_gru",
            "camera_contract": cc.camera_contract_for_policy(profile, 0.01).manifest(), "iteration": 1}
    ckpt.update(override)
    return ckpt


def _gavd_checkpoint(profile="vendor_new", config_profile=None, **override):
    ckpt = {"model": GridAttentionStudent(profile).state_dict(), "student_arch": "grid_attention_v1",
            "student_config": {"camera_profile": config_profile or profile},
            "camera_contract": cc.camera_contract_for_policy(profile, 0.01).manifest(), "iteration": 1}
    ckpt.update(override)
    return ckpt


def test_export_validation_accepts_consistent_rvld_and_gavd_checkpoints():
    validate = _export_module().validate_export_checkpoint
    for ckpt in (_rvld_checkpoint(), _gavd_checkpoint(), _gavd_checkpoint("vendor_legacy")):
        assert validate(ckpt, 80, 45, 32, environ={})["profile"] == ckpt["camera_contract"]["profile"]


@pytest.mark.parametrize("case", ["rvld_missing", "gavd_missing", "unknown_profile", "tampered_geometry",
                                  "image_size", "gavd_config_profile", "rvld_latent_size"])
def test_export_validation_refuses_missing_or_inconsistent_contracts(case):
    validate = _export_module().validate_export_checkpoint
    tampered = cc.camera_contract_for_policy("vendor_new", 0.01).manifest()
    tampered["quaternions_opengl"] = cc.load_camera_contract("vendor_legacy").quaternions_opengl  # same name, other geometry
    ckpt, size, latent = {
        "rvld_missing": (_rvld_checkpoint(camera_contract=None), (80, 45), 32),
        "gavd_missing": (_gavd_checkpoint(camera_contract=None), (80, 45), 32),
        "unknown_profile": (_rvld_checkpoint(camera_contract={**tampered, "profile": "vendor_typo"}), (80, 45), 32),
        "tampered_geometry": (_rvld_checkpoint(camera_contract=tampered), (80, 45), 32),
        "image_size": (_rvld_checkpoint(), (640, 360), 32),
        "gavd_config_profile": (_gavd_checkpoint("vendor_new", config_profile="vendor_legacy"), (80, 45), 32),
        "rvld_latent_size": (_rvld_checkpoint(), (80, 45), 16),
    }[case]
    with pytest.raises(ValueError):
        validate(ckpt, *size, latent, environ={})


def test_export_entry_point_refuses_before_writing_anything(tmp_path, monkeypatch):
    module = _export_module()
    checkpoint = tmp_path / "perception_1.pt"
    torch.save(_rvld_checkpoint(camera_contract=None), checkpoint)
    actor = tmp_path / "policy.onnx"
    monkeypatch.setattr("sys.argv", ["export_student.py", str(checkpoint), "--actor-onnx", str(actor)])
    monkeypatch.delenv(cc.ALLOW_MISSING_CAMERA_CONTRACT_ENV, raising=False)
    with pytest.raises(ValueError, match="no camera contract"):
        module.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["perception_1.pt"]  # nothing exported


@pytest.mark.parametrize("case", ["gavd_missing", "gavd_config_profile", "gavd_tampered_geometry"])
def test_export_entry_point_refuses_gavd_checkpoints(tmp_path, monkeypatch, case):
    tampered = cc.camera_contract_for_policy("vendor_new", 0.01).manifest()
    tampered["positions"] = cc.load_camera_contract("vendor_legacy").positions
    ckpt = {"gavd_missing": _gavd_checkpoint(camera_contract=None),
            "gavd_config_profile": _gavd_checkpoint("vendor_new", config_profile="vendor_legacy"),
            "gavd_tampered_geometry": _gavd_checkpoint(camera_contract=tampered)}[case]
    checkpoint = tmp_path / "perception_1.pt"
    torch.save(ckpt, checkpoint)
    monkeypatch.setattr("sys.argv", ["export_student.py", str(checkpoint), "--actor-onnx", str(tmp_path / "policy.onnx")])
    monkeypatch.delenv(cc.ALLOW_MISSING_CAMERA_CONTRACT_ENV, raising=False)
    with pytest.raises(ValueError):
        _export_module().main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["perception_1.pt"]


# ---------------------------------------------------------------- server entry-path smoke kit (CPU part)
def test_negative_smoke_fixtures_reach_and_fail_only_the_camera_check(tmp_path):
    import hashlib
    import subprocess
    import sys

    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"stand-in teacher bytes")
    out = tmp_path / "fixtures"
    subprocess.run([sys.executable, str(ROOT / "tools/camera_negative_smoke/make_fixtures.py"), "main", str(teacher), str(out)],
                   check=True, cwd=ROOT, capture_output=True)
    sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
    legacy_run = cc.camera_contract_for_policy("vendor_legacy", 0.01)
    for name, arch in (("rvld", "cnn_gru"), ("gavd", "grid_attention_v1")):
        for case in ("missing", "mismatch", "match"):
            saved = torch.load(out / f"{name}_{case}.pt", map_location="cpu", weights_only=False)
            # the checks distill_student.py runs before the camera check all pass
            assert saved["teacher_sha256"] == sha and saved["student_arch"] == arch
            assert saved["student_alignment_contract"] == 2
            if case == "match":
                cc.check_checkpoint_camera_contract(saved["camera_contract"], legacy_run, purpose="resume", environ={})
                if name == "gavd":
                    GridAttentionStudent(saved["student_config"]["camera_profile"]).load_state_dict(saved["model"])
            else:
                with pytest.raises(ValueError):
                    cc.check_checkpoint_camera_contract(saved.get("camera_contract"), legacy_run, purpose="resume",
                                                        environ={cc.ALLOW_LEGACY_CAMERA_ENV: "1"})


def test_negative_smoke_runner_expects_each_refusal_message():
    text = (ROOT / "tools/camera_negative_smoke/run.sh").read_text()
    assert "unset GD_LAB_ALLOW_MISSING_CAMERA_CONTRACT" in text and "GD_LAB_ALLOW_LEGACY_CAMERA=1" in text
    for message in ("checkpoint has no camera contract", "camera contract differs from the one required",
                    "[INFO] Student/optimizer resumed at 1;", "Resume camera calibration/timing contract differs"):
        assert message in text
    source = (ROOT / "scripts/distill_student.py").read_text() + (ROOT / "src/gd_lab/core/camera_contract.py").read_text()
    for message in ("checkpoint has no camera contract", "camera contract differs from the one required",
                    "Student/optimizer resumed at"):
        assert message in source  # the runner greps for strings the code really prints
    for script in ("scripts/gast/train_student.py", "scripts/gast/train_student_live.py"):
        gast = (ROOT / script).read_text()
        assert "Resume camera calibration/timing contract differs from this run" in gast
        assert "[INFO] Student/optimizer resumed at {start_iteration};" in gast
    play = (ROOT / "scripts/play_student.py").read_text()
    assert "[INFO] Student checkpoint:" in play and "import gd_lab.teachers.bivt.student_task" in play
    for case in ("train_student_live.py", "scripts/play_student.py", "--experiment_name camsmoke --load_run teacher"):
        assert case in text  # evaluation and the GAST live trainer are part of the kit
    assert 'realpath "${1:-' in text and "exit 101" in text and "exit 102" in text  # setup errors stop the run
    assert "exit 3" not in text  # setup codes never collide with a failed-case count
    assert '[ "$kind" = refuse ] && [ "$status" -ne 0 ]' in text  # refusal also needs a failing exit status


def test_exported_student_onnx_records_its_camera_profile(tmp_path):
    onnx = pytest.importorskip("onnx")
    from gd_lab.deploy.export_student_vrl import export_student_vrl
    from gd_lab.students.gavd.model import GridAttentionStudent

    def props(path):
        return {p.key: p.value for p in onnx.load(path).metadata_props}

    student = GridAttentionStudent("vendor_new").eval()
    _, path = export_student_vrl(student, str(tmp_path / "new.onnx"), camera_profile="vendor_new")
    assert props(path)["camel.camera_profile"] == "vendor_new"
    _, path = export_student_vrl(student, str(tmp_path / "unverified.onnx"), camera_profile=None)
    assert "camel.camera_profile" not in props(path)  # deploy runtime then treats it as legacy-only
    with pytest.raises(ValueError, match="not 'vendor_legacy'"):
        export_student_vrl(student, str(tmp_path / "wrong.onnx"), camera_profile="vendor_legacy")
    with pytest.raises(ValueError):
        export_student_vrl(student, str(tmp_path / "bogus.onnx"), camera_profile="vendor_bogus")
    with pytest.raises(TypeError):  # the profile must always be stated
        export_student_vrl(student, str(tmp_path / "implicit.onnx"))
