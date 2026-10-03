from pathlib import Path

from gd_lab.students.package import find_teacher_env_yaml


def test_finds_staged_teacher_params(tmp_path):
    launch = tmp_path / "run.launch"
    launch.mkdir()
    checkpoint = launch / "teacher.pt"
    checkpoint.touch()
    metadata = launch / "teacher_params" / "env.yaml"
    metadata.parent.mkdir()
    metadata.write_text("camera_profile: vendor_new\n")

    assert find_teacher_env_yaml(checkpoint) == metadata


def test_finds_native_params_and_rejects_missing(tmp_path):
    package = tmp_path / "teacher"
    (package / "params").mkdir(parents=True)
    checkpoint = package / "model.pt"
    checkpoint.touch()
    metadata = package / "params" / "env.yaml"
    metadata.write_text("camera_profile: vendor_legacy\n")

    assert find_teacher_env_yaml(checkpoint) == metadata
    assert find_teacher_env_yaml(tmp_path / "missing.pt") is None
