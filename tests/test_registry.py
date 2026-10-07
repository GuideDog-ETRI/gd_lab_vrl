from __future__ import annotations

import gymnasium as gym
import pytest

import gd_lab  # noqa: F401  (registers the tasks)
import gd_lab.teachers.bivt  # noqa: F401 (registers optional teacher variants)
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
        "Gd-VrlBlindStart-Rbq10-Dreamwaq-v0",
        "Gd-VrlBlindStart-Rbq10-Dreamwaq-Vision-v0",
        "Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0",
        # opt-in gap fine-tuning arms; every ID above is unchanged and keeps its position
        "Gd-VrlGapFinetuneBaselineRaycast-Rbq10-Dreamwaq-v0",
        "Gd-VrlGapFinetuneIntrusionRaycast-Rbq10-Dreamwaq-v0",
        "Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0",
        # v2 (2026-10-06): Clean + gap foothold margin / slot probing + stair hip-handle disturbance
        "Gd-VrlGapFinetuneCleanV2Raycast-Rbq10-Dreamwaq-v0",
        # v2.1 (2026-10-07): v2 + speeds up to 1.2 m/s, hind-hop / overspeed / stair-stall costs
        "Gd-VrlGapFinetuneCleanV21Raycast-Rbq10-Dreamwaq-v0",
    ]


def test_ids_resolve_in_gym_registry():
    for task_id in registry.all_ids():
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].startswith(("gd_lab.tasks.", "gd_lab.teachers."))
        assert spec.kwargs["rsl_rl_cfg_entry_point"].startswith(("gd_lab.agents.", "gd_lab.teachers."))


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        registry.register_task(
            task="Blind", robot="Rbq10", method="Dreamwaq", env_cfg="x:y", agent_cfg="x:y"
        )


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        registry.make_task_id(task="Blind", robot="Rbq10", method="Dreamwaq", mode="Heavy")
