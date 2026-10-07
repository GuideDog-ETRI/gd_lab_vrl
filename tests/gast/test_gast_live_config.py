import json

import pytest

from gd_lab.gast.live_config import LIVE_DEFAULTS, StudentLiveConfig


def test_live_config_creates_defaults_and_reloads_allowed_values(tmp_path):
    path = tmp_path / "student_live.json"
    live = StudentLiveConfig(path, {"learning_rate": 5e-4})
    assert json.loads(path.read_text()) == {**LIVE_DEFAULTS, "learning_rate": 5e-4}

    path.write_text(json.dumps({**LIVE_DEFAULTS, "learning_rate": 2e-4,
                                "checkpoint_interval": 500}))
    settings, changed, warning = live.poll()
    assert warning is None
    assert settings["learning_rate"] == 2e-4
    assert changed == {"learning_rate": 2e-4, "checkpoint_interval": 500}


def test_live_config_rejects_invalid_edit_and_keeps_last_valid_settings(tmp_path):
    path = tmp_path / "student_live.json"
    live = StudentLiveConfig(path, LIVE_DEFAULTS)
    path.write_text('{"learning_rate": 0}')
    settings, changed, warning = live.poll()
    assert warning and "learning_rate must be positive" in warning
    assert settings == LIVE_DEFAULTS
    assert changed == {}


@pytest.mark.parametrize("setting,value", [
    ("num_envs", 1024),
    ("iterations", 20000),
    ("teacher_checkpoint", "another.pt"),
    ("camera_delay_ms", [0, 100]),
    ("hazard_loss_coef", 0.9),
])
def test_live_config_refuses_startup_contract_changes(tmp_path, setting, value):
    live = StudentLiveConfig(tmp_path / "student_live.json", LIVE_DEFAULTS)
    live.path.write_text(json.dumps({**LIVE_DEFAULTS, setting: value}))
    settings, changed, warning = live.poll()
    assert warning and "unsupported live setting" in warning
    assert settings == LIVE_DEFAULTS
    assert changed == {}
