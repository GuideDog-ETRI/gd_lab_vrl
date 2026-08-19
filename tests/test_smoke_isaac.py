"""Simulator smoke test: build every registered task and run one learning iteration.

Requires IsaacSim + IsaacLab; run on the training machine with
``GD_LAB_ISAAC_TESTS=1 pytest tests/test_smoke_isaac.py``.
"""

from __future__ import annotations

import importlib
import os

import pytest

if os.environ.get("GD_LAB_ISAAC_TESTS") != "1":
    pytest.skip("set GD_LAB_ISAAC_TESTS=1 to run simulator tests", allow_module_level=True)
pytest.importorskip("isaaclab")

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import gd_lab  # noqa: F401, E402
from gd_lab.core import registry  # noqa: E402
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking  # noqa: E402
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC  # noqa: E402


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


# The Gamepad variant needs a physical device; smoke-test the other two.
_SMOKE_IDS = [tid for tid in registry.all_ids() if "Gamepad" not in tid]


@pytest.mark.parametrize("task_id", _SMOKE_IDS)
def test_constructs_and_learns(task_id, tmp_path):
    env_cfg = parse_env_cfg(task_id, num_envs=2)
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.logger = "tensorboard"  # no external logging service in the smoke test
    # A fresh USD stage per test; reusing the previous test's stage hangs env creation.
    omni.usd.get_context().new_stage()
    env = gym.make(task_id, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)

    # Obs contract: the live groups must match the spec exactly.
    om = env.unwrapped.observation_manager
    for group, spec in DREAMWAQ_SPEC.groups.items():
        scanner = env_cfg.scene.height_scanner.pattern_cfg
        rays = (round(scanner.size[0] / scanner.resolution) + 1) * (round(scanner.size[1] / scanner.resolution) + 1)
        resolved = spec.resolve(**{t.name: rays for t in spec.terms if t.dim is None})
        assert list(om.active_terms[group]) == [t.name for t in resolved.terms]
        assert om.group_obs_dim[group][0] == resolved.total

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_class = _resolve(agent_cfg.class_name)
    # rsl_rl requires a real log_dir (store_code_state does not accept None).
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=str(tmp_path / "log"), device=agent_cfg.device)
    runner.learn(num_learning_iterations=1)
    env.close()
