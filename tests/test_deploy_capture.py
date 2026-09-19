"""Deployment metadata must describe one robot even when training many environments."""

from types import SimpleNamespace as Namespace

import pytest
import torch

from gd_lab.deploy.metadata import capture_context


@pytest.fixture
def capture_inputs():
    joints = [f"{leg}_{joint}" for joint in ("HIP", "THIGH", "KNEE") for leg in ("FL", "FR", "HL", "HR")]
    defaults = torch.tensor([0.0] * 4 + [0.76] * 4 + [-1.45] * 4).repeat(32, 1)
    robot = Namespace(
        joint_names=joints,
        data=Namespace(default_joint_pos=defaults, default_joint_vel=torch.zeros_like(defaults)),
        cfg=Namespace(actuators={
            "hip": Namespace(joint_names_expr=[".*_HIP", ".*_THIGH"], stiffness=88.1367, damping=1.9919),
            "knee": Namespace(joint_names_expr=[".*_KNEE"], stiffness=102.2177, damping=1.9932),
        }),
    )
    action = Namespace(
        _joint_names=joints, _scale=0.25, _offset=defaults.clone(), cfg=Namespace(clip=None, soft_margin_deg=5.0),
    )
    observation = Namespace(
        func=Namespace(__name__="payload_mass"), history_length=5, flatten_history_dim=True, clip=None, scale=0.2,
    )
    env = Namespace(
        scene={"robot": robot}, step_dt=0.02,
        action_manager=Namespace(_terms={"joint_pos": action}),
        observation_manager=Namespace(
            group_obs_concatenate={"policy": True}, active_terms={"policy": ["payload"]},
            group_obs_term_dim={"policy": [(5,)]}, _group_obs_term_cfgs={"policy": [observation]},
        ),
    )
    env.unwrapped = env
    return env, Namespace(obs_groups={"policy": ["policy"]}, actor_history_steps=4), action


@pytest.mark.parametrize("batched_scale", [False, True])
def test_capture_collapses_identical_environment_rows(capture_inputs, batched_scale):
    env, policy, action = capture_inputs
    if batched_scale:
        action._scale = torch.full((32, 12), 0.25)
    context = capture_context(env, policy)
    assert context["action"]["scale"] == [0.25] * 12
    assert context["action"]["offset"] == env.scene["robot"].data.default_joint_pos[0].tolist()
    assert context["action"]["offset"] == context["default_joint_pos"]
    assert context["terms"][0]["source"] == "payload"


@pytest.mark.parametrize("field", ["_scale", "_offset"])
def test_capture_rejects_different_transforms_across_environments(capture_inputs, field):
    env, policy, action = capture_inputs
    value = torch.full((32, 12), 0.25)
    value[1, 0] = 0.5
    setattr(action, field, value)
    with pytest.raises(ValueError, match="Per-environment action transforms"):
        capture_context(env, policy)
