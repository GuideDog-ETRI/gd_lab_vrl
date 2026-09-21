from __future__ import annotations

import gymnasium as gym
import pytest

import gd_lab  # noqa: F401  (registers the tasks)
from gd_lab.core import registry


def test_registered_blind_and_vision_ids():
    assert registry.all_ids() == [
        "Gd-Blind-Rbq10-Dreamwaq-v0",
        "Gd-Blind-Rbq10-Dreamwaq-Play-v0",
        "Gd-Blind-Rbq10-Dreamwaq-Gamepad-v0",
        "Gd-Vrl-Rbq10-Dreamwaq-v0",
        "Gd-Vrl-Rbq10-Dreamwaq-Play-v0",
        "Gd-Vrl-Rbq10-Dreamwaq-Vision-v0",
        "Gd-Vrl-Rbq10-Dreamwaq-VisionPlay-v0",
    ]


def test_ids_resolve_in_gym_registry():
    for task_id in registry.all_ids():
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].startswith("gd_lab.tasks.")
        assert spec.kwargs["rsl_rl_cfg_entry_point"].startswith("gd_lab.agents.")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        registry.register_task(
            task="Blind", robot="Rbq10", method="Dreamwaq", env_cfg="x:y", agent_cfg="x:y"
        )


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        registry.make_task_id(task="Blind", robot="Rbq10", method="Dreamwaq", mode="Heavy")
