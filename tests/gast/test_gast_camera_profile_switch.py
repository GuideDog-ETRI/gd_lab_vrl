"""GAST paths of the SDK camera switch (GAST code lives in the shared src/gd_lab)."""

from pathlib import Path

import pytest
import torch

from gd_lab.core import camera_contract as cc
from gd_lab.gast.student import GastStudent

GAST = Path(__file__).parents[2]  # repository root (GAST used to be a separate gast/ tree)


def test_gast_imports_the_shared_package():
    assert Path(cc.__file__).resolve().is_relative_to((GAST / "src").resolve())


def test_gast_defaults_to_the_sdk_profile_and_guards_legacy_training():
    assert cc.DEFAULT_CAMERA_PROFILE == "vendor_new"
    assert GastStudent().camera_profile == "vendor_new"
    with pytest.raises(ValueError):
        cc.require_training_camera_profile("vendor_legacy", environ={})


def test_gast_student_refuses_a_checkpoint_with_another_camera_geometry():
    legacy = GastStudent("vendor_legacy").state_dict()
    with pytest.raises(RuntimeError):
        GastStudent("vendor_new").load_state_dict(legacy)
    GastStudent("vendor_legacy").load_state_dict(legacy)  # explicit legacy evaluation still works
    new = GastStudent("vendor_new")
    assert not torch.allclose(new.rays, GastStudent("vendor_legacy").rays)


@pytest.mark.parametrize("script", ["scripts/gast/train_student.py", "scripts/gast/train_student_live.py"])
def test_gast_trainers_guard_legacy_and_compare_calibration_signatures(script):
    source = (GAST / script).read_text()
    assert "require_training_camera_profile(env_cfg.camera_profile)" in source
    assert "check_checkpoint_camera_contract(saved.get('camera_contract'), camera_contract, purpose='resume')" in source


def test_gast_student_nested_load_is_verified():
    class Parent(torch.nn.Module):
        def __init__(self, profile):
            super().__init__()
            self.student = GastStudent(profile)

    with pytest.raises(RuntimeError):
        Parent("vendor_new").load_state_dict(Parent("vendor_legacy").state_dict())


def test_gast_contract_identity_keeps_timing():
    current = cc.camera_contract_for_policy("vendor_new", 0.01)
    assert not cc.same_camera_contract({**current.manifest(), "policy_dt": 0.02}, current)


def test_gast_contract_check_refuses_missing_contracts_by_default():
    current = cc.camera_contract_for_policy("vendor_new", 0.01)
    with pytest.raises(ValueError):
        cc.check_checkpoint_camera_contract(None, current, purpose="resume", environ={})
    with pytest.raises(ValueError):
        cc.check_checkpoint_camera_contract(cc.camera_contract_for_policy("vendor_legacy", 0.01).manifest(), current,
                                            purpose="resume", environ={})


def test_gast_uses_the_shared_package_helper_and_rank_agnostic_finite_check():
    """train_student.py imports gd_lab.students.package; hazard labels are 1-D per env."""
    from gd_lab.students.package import find_teacher_env_yaml  # noqa: F401  (ImportError broke GAST launch)
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/gast/train_student.py").read_text()
    assert "hazard_label.flatten(1)" not in source and "rows_finite(hazard_label)" in source
    namespace = {"torch": torch}
    start = source.index("def rows_finite(")
    exec(source[start:source.index("\n\n\n", start)], namespace)
    rows = namespace["rows_finite"]
    assert rows(torch.tensor([1.0, float("nan"), 2.0])).tolist() == [True, False, True]
    assert rows(torch.ones(2, 3, 4)).tolist() == [True, True]
