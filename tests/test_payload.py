"""Physical invariants and reset behavior of the per-episode trunk payload."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gd_lab.core.payload_math import combine_point_payload

_SRC = Path(__file__).parents[1] / "src/gd_lab"


def _load_function(relative_path: str, name: str):
    """Exercise the real event/observation without importing Isaac Sim modules."""
    source = (_SRC / relative_path).read_text()
    tree = ast.parse(source)
    namespace = {"torch": torch, "combine_point_payload": combine_point_payload}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (name, "_resolved_body_ids"):
            exec("from __future__ import annotations\n" + ast.get_source_segment(source, node), namespace)
    return namespace[name]


randomize_payload_mass = _load_function("mdp/events.py", "randomize_payload_mass")
payload_observation = _load_function("mdp/observations.py", "payload_mass")


def _configured_payload() -> dict:
    tree = ast.parse((_SRC / "methods/dreamwaq/events.py").read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DreamwaqEventsCfg")
    event = next(
        n.value
        for n in cfg.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "randomize_payload" for t in n.targets)
    )
    params = next(kw.value for kw in event.keywords if kw.arg == "params")
    return {
        k.value: ast.literal_eval(v)
        for k, v in zip(params.keys, params.values, strict=True)
        if k.value.startswith("payload_")
    }


class _PhysicsView:
    def __init__(self, count: int):
        self.masses = torch.tensor([3.0, 16.758, 8.0]).repeat(count, 1)
        # Different startup mass/CoM randomization in each environment.
        self.masses[:, 1] += torch.linspace(-2.0, 2.0, count)
        self.coms = torch.zeros(count, 3, 7)
        self.coms[..., 6] = 1.0  # PhysX xyzw quaternion.
        self.coms[:, 1, :3] = torch.tensor([-0.013265, 0.005, 0.011765])
        self.coms[:, 1, 0] += torch.linspace(-0.05, 0.05, count)
        inertia = torch.tensor([[0.16, 0.0, 0.04], [0.0, 0.59, 0.0], [0.04, 0.0, 0.68]])
        self.inertias = inertia.flatten().repeat(count, 3, 1)

    def get_masses(self):
        return self.masses

    def get_coms(self):
        return self.coms

    def get_inertias(self):
        return self.inertias

    def set_masses(self, values, ids):
        self.masses[ids] = values[ids]

    def set_coms(self, values, ids):
        self.coms[ids] = values[ids]

    def set_inertias(self, values, ids):
        self.inertias[ids] = values[ids]


def _environment(count: int):
    view = _PhysicsView(count)
    data = SimpleNamespace(_body_com_pose_b=SimpleNamespace(timestamp=5.0))
    robot = SimpleNamespace(num_bodies=3, root_physx_view=view, data=data)
    env = SimpleNamespace(num_envs=count, device="cpu", scene={"robot": robot})
    return env, view, SimpleNamespace(name="robot", body_ids=[1])


def _state(view):
    return [v.clone() for v in (view.masses, view.coms, view.inertias)]


def test_combined_inertia_preserves_rotational_kinetic_energy():
    f64 = dict(dtype=torch.float64)
    base_mass = torch.tensor(16.758, **f64)
    payload = torch.tensor(6.0, **f64)
    base_com = torch.tensor([-0.013265, 0.004, 0.011765], **f64)
    position = torch.tensor([0.05, 0.0, 0.07], **f64)
    base_inertia = torch.tensor([[0.16, 0.0, 0.04], [0.0, 0.59, 0.0], [0.04, 0.0, 0.68]], **f64)
    mass, com, inertia = combine_point_payload(base_mass, base_com, base_inertia, payload, position)

    torch.testing.assert_close(mass * com, base_mass * base_com + payload * position)
    for omega in torch.eye(3, **f64).unbind() + (torch.tensor([1.0, -2.0, 3.0], **f64),):
        base_v = torch.linalg.cross(omega, base_com - com)
        payload_v = torch.linalg.cross(omega, position - com)
        separate = (
            0.5 * omega @ base_inertia @ omega
            + 0.5 * base_mass * base_v.square().sum()
            + 0.5 * payload * payload_v.square().sum()
        )
        torch.testing.assert_close(0.5 * omega @ inertia @ omega, separate)
    assert torch.linalg.eigvalsh(inertia).min() > 0

    # Changing the link origin must not change the physical inertia about CoM.
    shift = torch.tensor([0.1, -0.2, 0.3], **f64)
    _, com2, inertia2 = combine_point_payload(base_mass, base_com + shift, base_inertia, payload, position + shift)
    torch.testing.assert_close(com2, com + shift)
    torch.testing.assert_close(inertia2, inertia)


def test_configured_payload_distribution_and_mount_position():
    torch.manual_seed(19)
    env, view, asset_cfg = _environment(20000)
    mass0, com0, _ = _state(view)
    randomize_payload_mass(env, None, asset_cfg, **_configured_payload())
    loaded = env.payload_kg == 6.0
    assert loaded.float().mean().item() == pytest.approx(0.8, abs=0.015)
    pos = env.payload_pos_b[loaded]
    assert pos[:, 0].min() >= -0.05 and pos[:, 0].max() <= 0.05
    assert pos[:, 0].mean().item() == pytest.approx(0.0, abs=0.001)
    assert pos[:, 0].var().item() == pytest.approx(0.1**2 / 12, rel=0.05)
    torch.testing.assert_close(pos[:, 1], torch.zeros_like(pos[:, 1]))
    torch.testing.assert_close(pos[:, 2], torch.full_like(pos[:, 2], 0.07))
    torch.testing.assert_close(env.payload_pos_b[~loaded], torch.zeros_like(env.payload_pos_b[~loaded]))
    torch.testing.assert_close(view.masses[:, 1], mass0[:, 1] + env.payload_kg)
    torch.testing.assert_close(
        view.masses[:, 1, None] * view.coms[:, 1, :3],
        mass0[:, 1, None] * com0[:, 1, :3] + env.payload_kg[:, None] * env.payload_pos_b,
    )
    torch.testing.assert_close(payload_observation(env)[:, 0], env.payload_kg * 0.2)
    torch.testing.assert_close(view.masses[:, [0, 2]], mass0[:, [0, 2]])
    torch.testing.assert_close(view.coms[:, [0, 2]], com0[:, [0, 2]])


def test_partial_unload_restores_startup_properties_without_accumulation():
    env, view, asset_cfg = _environment(4)
    baseline = _state(view)
    kwargs = _configured_payload()
    kwargs["payload_probabilities"] = (0.0, 1.0)
    kwargs["payload_position_range"]["x"] = (0.03, 0.03)
    randomize_payload_mass(env, None, asset_cfg, **kwargs)
    loaded_state = _state(view)
    for _ in range(3):
        randomize_payload_mass(env, None, asset_cfg, **kwargs)
        for actual, expected in zip(_state(view), loaded_state, strict=True):
            torch.testing.assert_close(actual, expected)

    ids = torch.tensor([1, 3])
    kwargs["payload_probabilities"] = (1.0, 0.0)
    randomize_payload_mass(env, ids, asset_cfg, **kwargs)
    for actual, original, loaded in zip(_state(view), baseline, loaded_state, strict=True):
        torch.testing.assert_close(actual[ids], original[ids])
        torch.testing.assert_close(actual[[0, 2]], loaded[[0, 2]])
    torch.testing.assert_close(env.payload_pos_b[ids], torch.zeros(2, 3))
    torch.testing.assert_close(payload_observation(env)[ids], torch.zeros(2, 1))
    assert env.scene["robot"].data._body_com_pose_b.timestamp < 0


def test_payload_without_mount_position_retains_proportional_inertia():
    env, view, asset_cfg = _environment(2)
    mass0, com0, inertia0 = _state(view)
    randomize_payload_mass(env, None, asset_cfg, payload_masses=(6.0,))
    torch.testing.assert_close(view.masses[:, 1], mass0[:, 1] + 6.0)
    torch.testing.assert_close(view.coms, com0)
    ratios = view.masses[:, 1] / mass0[:, 1]
    torch.testing.assert_close(view.inertias[:, 1], inertia0[:, 1] * ratios[:, None])


def test_empty_reset_and_bad_probabilities_do_not_change_physics():
    env, view, asset_cfg = _environment(2)
    original = _state(view)
    randomize_payload_mass(env, torch.empty(0, dtype=torch.long), asset_cfg, **_configured_payload())
    with pytest.raises(ValueError, match="payload_probabilities"):
        randomize_payload_mass(env, None, asset_cfg, payload_masses=(0.0, 6.0), payload_probabilities=(1.0,))
    for actual, expected in zip(_state(view), original, strict=True):
        torch.testing.assert_close(actual, expected)
