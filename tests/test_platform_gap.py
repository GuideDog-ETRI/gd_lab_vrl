"""CPU geometry/reward checks; manager adapter uses no physics or rendering."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from gd_lab.mdp.platform_gap_math import boarding_crossing_candidates, boarding_drop_cost, boarding_support_height
from gd_lab.mdp.platform_gap_mesh import build_boarding_mesh
from gd_lab.mdp.terrain_curriculums import terrain_levels_vel_cmd_aware
from gd_lab.mdp.terrain_families import family_column_masks, terrain_family_gate


@pytest.mark.parametrize("difficulty", [0.0, 0.3, 1.0])
def test_platform_mesh_has_safe_spawn_and_deep_slots(difficulty):
    np.random.seed(42)
    meshes, origin = build_boarding_mesh(difficulty, (8, 8), (0.02, 0.24), (0, 0.16), 1.5, 0.65)
    assert len(meshes) == 5
    np.testing.assert_equal(origin, [4, 4, 0])
    center = meshes[1].bounds
    assert center[0, 0] < 4 - 1.0 and center[1, 0] > 4 + 1.0
    assert center[1, 2] == 0
    for landing, pit in ((meshes[0], meshes[3]), (meshes[2], meshes[4])):
        assert pit.bounds[1, 2] <= min(0, landing.bounds[1, 2]) - 0.65 + 1e-8
        assert pit.extents[0] == pytest.approx(0.02 + difficulty * 0.22)
    np.random.seed(42)
    repeat, _ = build_boarding_mesh(difficulty, (8, 8), (0.02, 0.24), (0, 0.16), 1.5, 0.65)
    for first, second in zip(meshes, repeat, strict=True):
        np.testing.assert_equal(first.vertices, second.vertices)


def test_platform_support_and_drop_are_pit_aware_and_finite():
    rays = torch.tensor([[0.0, -0.16, -0.65, float("nan")], [float("inf")] * 4])
    support = boarding_support_height(rays, torch.tensor([0.0, 1.0]))
    assert torch.allclose(support, torch.tensor([-0.08, 1.0]))
    feet = torch.tensor([[-0.16] * 4, [0.3] * 4])
    assert torch.equal(boarding_drop_cost(feet, support), torch.tensor([0.0, 1.0]))


def test_platform_crossing_requires_all_feet_and_no_side_bypass():
    feet = torch.zeros(3, 4, 3)
    feet[0, :, 0] = 1.9
    feet[1, :3, 0] = 1.9
    feet[2, :, 0] = -1.9
    feet[2, 0, 1] = 4.0
    candidate = boarding_crossing_candidates(feet, 1.68, 4.0)
    assert torch.equal(candidate, torch.tensor([[False, True], [False, False], [False, False]]))


class _Scene(dict):
    def __getattr__(self, key):
        return self[key]


class _TermBase:
    def __init__(self, cfg, env):
        self._env = env


def _load_adapter():
    # Execute actual term definitions with only Isaac's manager base/entity
    # configuration replaced. This checks tensor/state logic, NOT Kit/PhysX APIs.
    path = Path(__file__).parents[1] / "src/gd_lab/mdp/platform_gap_terms.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(
        torch=torch, ManagerTermBase=_TermBase, SceneEntityCfg=lambda name: NS(name=name),
        boarding_crossing_candidates=boarding_crossing_candidates, boarding_drop_cost=boarding_drop_cost,
        boarding_support_height=boarding_support_height, terrain_family_gate=terrain_family_gate,
        terrain_levels_vel_cmd_aware=terrain_levels_vel_cmd_aware,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def _environment():
    sub = {f"original_{i}": NS(proportion=1.0) for i in range(10)}
    sub["platform_gap"] = NS(proportion=5.0, gap_center_offset=1.5, gap_width_range=(0.02, 0.24))
    gen = NS(num_cols=15, size=(8, 8), sub_terrains=sub)
    updates = []
    terrain = NS(
        cfg=NS(terrain_generator=gen), terrain_types=torch.tensor([10, 11, 0]), terrain_levels=torch.zeros(3),
        update_env_origins=lambda ids, up, down: updates.append((ids.clone(), up.clone(), down.clone())),
    )
    robot = NS(data=NS(body_pos_w=torch.zeros(3, 4, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]] * 3)))
    contacts = torch.zeros(3, 4, 3)
    contacts[..., 2] = 20
    scene = _Scene(
        robot=robot, terrain=terrain, env_origins=torch.zeros(3, 3),
        height_scanner=NS(data=NS(ray_hits_w=torch.zeros(3, 187, 3))),
        contact_forces=NS(data=NS(net_forces_w=contacts)),
    )
    env = NS(num_envs=3, device="cpu", step_dt=0.02, scene=scene,
             termination_manager=NS(terminated=torch.zeros(3, dtype=torch.bool)))
    return env, updates


def test_platform_exact_family_allocation():
    env, _ = _environment()
    masks = family_column_masks(env)
    assert len(masks) == 11
    assert masks["platform_gap"].tolist() == [10, 11, 12, 13, 14]
    assert all(len(masks[f"original_{i}"]) == 1 for i in range(10))


def test_platform_bonus_once_partial_reset_and_curriculum():
    module = _load_adapter()
    env, updates = _environment()
    term = module["PlatformGapCrossing"](None, env)
    asset, sensor = NS(name="robot", body_ids=slice(None)), NS(name="contact_forces", body_ids=slice(None))
    env.scene.robot.data.body_pos_w[..., 0] = 1.9
    events = torch.zeros(3)
    for _ in range(20):
        events += term(env, asset, sensor) * env.step_dt * 0.5
    assert torch.allclose(events, torch.tensor([0.5, 0.5, 0.0]))
    env.scene.robot.data.body_pos_w[0, :, 0] = 0
    term(env, asset, sensor)
    assert term.achieved[0].any()  # success survives a return to the spawn pad
    term.reset(torch.tensor([1]))
    assert term.achieved[0].any() and not term.achieved[1].any()
    env.reward_manager = NS(get_term_cfg=lambda name: NS(func=term))
    command = NS(cmd_moving_time=torch.full((3,), 5.), cmd_dist_integral=torch.full((3,), 3.))
    env.command_manager = NS(get_term=lambda name: command)
    env.termination_manager.terminated[1] = True
    module["platform_gap_levels"](env, torch.tensor([0, 1]))
    assert updates[0][1].tolist() == [True, False]
    assert updates[0][2].tolist() == [False, True]
    term.reset()
    assert not term.paid.any()


@pytest.mark.parametrize("invalid", ["pit", "side", "fallen", "terminated", "unsupported"])
def test_platform_no_success_for_unsafe_crossing(invalid):
    module = _load_adapter()
    env, _ = _environment()
    term = module["PlatformGapCrossing"](None, env)
    env.scene.robot.data.body_pos_w[..., 0] = 1.9
    if invalid == "pit":
        env.scene.robot.data.body_pos_w[..., 2] = -0.65
    elif invalid == "side":
        env.scene.robot.data.body_pos_w[..., 1] = 4.0
    elif invalid == "fallen":
        env.scene.robot.data.projected_gravity_b[:, 2] = 0
    elif invalid == "terminated":
        env.termination_manager.terminated[:] = True
    else:
        env.scene.contact_forces.data.net_forces_w[:] = 0
    for _ in range(10):
        result = term(env, NS(name="robot", body_ids=slice(None)), NS(name="contact_forces", body_ids=slice(None)))
        assert not result.any()


def test_platform_gap_depth_ghost_changes_only_depth_and_only_gap_family():
    import torch

    from gd_lab.mdp.platform_gap_noise import PlatformGapDepthGhost, PlatformGapDepthGhostCfg

    n, cameras, h, w = 2, 4, 9, 12
    frames = torch.zeros(n, cameras, 2, h, w)
    frames[:, :, 0] = 0.2
    frames[:, :, 1] = 0.37
    depth = torch.full((n, cameras, h, w), 1.0)
    positions = torch.zeros(n, cameras, 3)
    positions[..., 0] = 1.5
    positions[..., 2] = -1.0
    rotations = torch.zeros(n, cameras, 4)
    rotations[..., 0] = 1.0
    intrinsics = torch.zeros(n, cameras, 3, 3)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 20.0
    intrinsics[..., 0, 2] = w / 2
    intrinsics[..., 1, 2] = h / 2
    intrinsics[..., 2, 2] = 1.0
    snapshot = (frames.clone(), depth, positions, rotations, intrinsics)
    cfg = PlatformGapDepthGhostCfg(probability=1.0, lifetime_steps=(3, 3), patch_fraction=(1.0, 1.0))
    noisy = PlatformGapDepthGhost(cfg)(frames, snapshot, torch.zeros(n, 3), torch.tensor([True, False]))
    assert torch.equal(noisy[1], frames[1])
    assert torch.equal(noisy[0, :, 1], frames[0, :, 1])
    assert not torch.equal(noisy[0, :, 0], frames[0, :, 0])
