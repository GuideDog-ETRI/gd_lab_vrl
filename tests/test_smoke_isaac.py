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
from hydra import compose, initialize  # noqa: E402
from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces  # noqa: E402
from isaaclab.utils import replace_strings_with_slices  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import register_task_to_hydra  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import gd_lab  # noqa: F401, E402
from gd_lab.core import registry  # noqa: E402
from gd_lab.core.experiments import training_arm_overrides  # noqa: E402
from gd_lab.deploy.metadata import capture_context  # noqa: E402
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking  # noqa: E402
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC  # noqa: E402


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


# The Gamepad variant needs a physical device; smoke-test the other two.
_SMOKE_IDS = [tid for tid in registry.all_ids() if "Gamepad" not in tid]


def _compose_arm(task_id, arm, extra=()):
    env_cfg, agent_cfg = register_task_to_hydra(task_id, "rsl_rl_cfg_entry_point")
    overrides = training_arm_overrides(arm, play="Play" in task_id or "Gamepad" in task_id)
    with initialize(config_path=None, version_base="1.3"):
        cfg = compose(config_name=task_id, overrides=overrides + list(extra))
    values = replace_strings_with_slices(OmegaConf.to_container(cfg, resolve=True))
    env_cfg.from_dict(values["env"])
    env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
    agent_cfg.from_dict(values["agent"])
    return env_cfg, agent_cfg


@pytest.mark.parametrize("task_id", registry.all_ids())
@pytest.mark.parametrize("arm, hz", [("1", 50), ("2", 100), ("3", 50), ("4", 100)])
def test_arm_config_matches_training_conditions(task_id, arm, hz):
    base = parse_env_cfg(task_id)
    env_cfg, agent_cfg = _compose_arm(task_id, arm)
    assert env_cfg.sim.dt == 0.005
    assert env_cfg.decimation * env_cfg.sim.dt == pytest.approx(1 / hz)
    assert env_cfg.sim.render_interval == env_cfg.decimation
    assert env_cfg.scene.contact_forces.update_period == env_cfg.sim.dt
    for name in (
        "height_scanner", "height_scanner_feet_fl", "height_scanner_feet_fr",
        "height_scanner_feet_rl", "height_scanner_feet_rr",
    ):
        assert getattr(env_cfg.scene, name).update_period == pytest.approx(1 / hz)
    assert env_cfg.curriculum.pulse_prob_schedule is None
    if "Gamepad" not in task_id:
        assert env_cfg.commands.base_velocity.pulse_prob == 0.0
    else:
        assert env_cfg.commands.to_dict() == base.commands.to_dict()
    for group, kp in (("rbq_hip", 123.39), ("rbq_knee", 127.77)):
        gains = env_cfg.scene.robot.actuators[group]
        defaults = base.scene.robot.actuators[group]
        assert gains.stiffness == (defaults.stiffness if arm in {"1", "2"} else kp)
        assert gains.damping == (defaults.damping if arm in {"1", "2"} else 2.4)
    payload = env_cfg.events.randomize_payload
    assert payload is not None and payload.mode == "reset"
    assert payload.params["payload_masses"] == (0.0, 6.0)
    assert payload.params["payload_probabilities"] == (0.2, 0.8)
    assert env_cfg.events.to_dict() == base.events.to_dict()
    assert agent_cfg.experiment_name == f"blind_rbq10_dreamwaq/arm_{arm}"
    # Loading an arm must not mutate the recipe used by the next run.
    assert parse_env_cfg(task_id).to_dict() == base.to_dict()


def test_arm_allows_explicit_cli_overrides():
    task_id = registry.make_task_id(task="Blind", robot="Rbq10", method="Dreamwaq")
    env_cfg, agent_cfg = _compose_arm(task_id, "4", [
        "env.scene.robot.actuators.rbq_hip.stiffness=120.0",
        "agent.experiment_name=custom_arm_run",
    ])
    assert env_cfg.scene.robot.actuators["rbq_hip"].stiffness == 120.0
    assert agent_cfg.experiment_name == "custom_arm_run"


_ARM_RUNS = [(registry.make_task_id(task="Blind", robot="Rbq10", method="Dreamwaq"), str(i)) for i in range(1, 5)]


@pytest.mark.parametrize("task_id, arm", [(tid, None) for tid in _SMOKE_IDS] + _ARM_RUNS)
def test_constructs_and_learns(task_id, arm, tmp_path):
    if arm is None:
        env_cfg = parse_env_cfg(task_id)
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    else:
        env_cfg, agent_cfg = _compose_arm(task_id, arm)
    env_cfg.scene.num_envs = 2
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
    runner.deploy_context = capture_context(env, runner.alg.policy)
    assert len(runner.deploy_context["action"]["offset"]) == env.num_actions
    assert runner.deploy_context["policy_dt"] == env.unwrapped.step_dt
    payload_term = next(t for t in runner.deploy_context["terms"] if t["source"] == "payload")
    assert len(payload_term["transforms"]) == 1
    assert payload_term["transforms"][0]["op"] == "scale"
    assert payload_term["transforms"][0]["values"] == pytest.approx([0.2])
    runner.learn(num_learning_iterations=1)
    env.close()
