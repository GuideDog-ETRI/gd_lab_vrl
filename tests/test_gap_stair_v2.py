"""v2 gap/stair teacher terms: pure math, strict clean, guard and config contracts (CPU only)."""

import ast
import math
from pathlib import Path

import pytest
import torch

from gd_lab.mdp.gap_stair_v2_math import (
    chebyshev_frame,
    downhill_force,
    front_lift_cost,
    gap_edge_margin_cost,
    nosed_edges,
    pyramid_step_edges,
    slot_low_or_contact_cost,
    stair_nose_margin_cost,
    stair_push_modes,
)
from gd_lab.mdp.platform_gap_attempts import GapAttemptTracker
from gd_lab.teachers.bivt import gap_guard

ROOT = Path(__file__).resolve().parents[1]
SLOTS = torch.tensor([[[-1.10, -1.00], [1.00, 1.20]]])  # one env, two slots


def test_gap_margin_is_zero_far_away_full_inside_and_linear_near_edges():
    x = torch.tensor([[0.0, 0.98, 1.05, 1.22]])  # deck, 2 cm before the near edge, inside, 2 cm after the far edge
    cost = gap_edge_margin_cost(x, SLOTS, margin=0.04)
    assert cost[0, 0] == 0
    assert torch.isclose(cost[0, 1], torch.tensor(0.5))
    assert cost[0, 2] == 1
    assert torch.isclose(cost[0, 3], torch.tensor(0.5))


def test_stair_margin_only_on_the_drop_off_side():
    edges = torch.tensor([[2.0, 1.75, 1.5, float("nan")]])
    cheb = torch.tensor([[1.99, 2.01, 1.60, 1.74]])
    pyramid = stair_nose_margin_cost(cheb, edges, torch.tensor([False]), margin=0.04)
    # pyramid (centre high): the tread just inside an edge is dangerous, the riser foot outside is not
    assert pyramid[0, 0] > 0.7 and pyramid[0, 1] == 0 and pyramid[0, 2] == 0 and pyramid[0, 3] > 0.7
    inverted = stair_nose_margin_cost(cheb, edges, torch.tensor([True]), margin=0.04)
    assert inverted[0, 0] == 0 and inverted[0, 1] > 0.7


def test_pyramid_edges_match_isaaclab_ring_geometry():
    edges = pyramid_step_edges(size=8.0, border_width=1.0, platform_width=3.0, step_width=0.3)
    # IsaacLab: num_steps = (8 - 2 - 3) // 0.6 + 1 = 6 (the "+1" eats into the platform) -> 7 edges, 0.3 m apart
    assert edges == pytest.approx([3.0, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2])
    num_steps = int((8.0 - 2 * 1.0 - 3.0) // (2 * 0.3) + 1)  # verbatim mesh_terrains.pyramid_stairs_terrain formula
    assert len(edges) == num_steps + 1


def test_slot_probe_uses_the_higher_deck_and_contact():
    x = torch.tensor([[1.10, 1.10, 1.10, 0.50]])
    z = torch.tensor([[0.06, 0.20, 0.20, 0.03]])  # low over slot, high over slot, high but touching, on deck
    contact = torch.tensor([[False, False, True, True]])
    upper = torch.tensor([[0.0, 0.10]])  # the right slot sits next to a 10 cm higher deck
    cost = slot_low_or_contact_cost(x, z, contact, SLOTS, upper)
    assert torch.isclose(cost, torch.tensor([0.5]))  # feet 0 and 2 of 4


def test_handle_pull_while_ascending_lifts_the_nose():
    uphill = torch.tensor([[1.0, 0.0]])  # robot faces +x, uphill
    handle = torch.tensor([-0.33, 0.0, 0.12])
    for degrees in (20.0, 30.0, 45.0):
        force = downhill_force(uphill, torch.tensor([100.0]), torch.tensor([math.radians(degrees)]))[0]
        assert force[0] < 0 and force[2] < 0  # back and down
        torque_y = handle[2] * force[0] - handle[0] * force[2]
        assert torque_y < 0  # rotation about -y: nose up, front feet unload
    up_pull = torch.tensor([-math.cos(math.radians(30)), 0.0, math.sin(math.radians(30))]) * 100
    assert handle[2] * up_pull[0] - handle[0] * up_pull[2] > 0  # the rejected draft (back AND up) pitched nose-down


def test_push_modes_require_forward_walking_along_the_slope():
    uphill = torch.tensor([[1.0, 0.0]] * 4)
    heading = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    velocity = torch.tensor([[0.4, 0.0], [-0.4, 0.0], [0.4, 0.0], [-0.4, 0.0]])
    command = torch.tensor([0.5, 0.5, 0.5, -0.5])
    ascending, descending = stair_push_modes(uphill, heading, velocity, command)
    assert ascending.tolist() == [True, False, False, False]  # sideways and backward walking excluded
    assert descending.tolist() == [False, True, False, False]


def test_chebyshev_frame_outward_direction():
    cheb, outward = chebyshev_frame(torch.tensor([[2.0, 0.5], [-0.3, -1.5]]))
    assert cheb.tolist() == [2.0, 1.5]
    assert outward.tolist() == [[1.0, 0.0], [0.0, -1.0]]


def test_front_lift_cost():
    contact = torch.tensor([[True, False], [False, False], [True, True]])
    cost = front_lift_cost(contact, torch.tensor([0.0, 0.0, 1.5]))
    assert cost.tolist() == [0.0, 1.0, 0.5]


def test_strict_clean_rejects_a_touch_over_the_slot():
    def run(strict):
        tracker = GapAttemptTracker(1, "cpu", strict_contact=strict)
        slots = torch.tensor([[[-1.2, -1.0], [1.0, 1.1]]])
        deck = torch.tensor([[-0.5, -0.5]])
        feet_z = torch.full((1, 4), 0.03)
        args = dict(foot_z=feet_z, slots=slots, lower_deck_z=deck, forward=torch.tensor([True]),
                    family=torch.tensor([True]))
        no_event, no_contact = torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)
        tracker.update(torch.tensor([[0.9, 0.8, 0.5, 0.4]]), crossing_event=no_event, contact=no_contact, **args)
        touch = torch.tensor([[True, False, False, False]])  # front foot touches while over the slot, not deep
        tracker.update(torch.tensor([[1.05, 0.8, 0.5, 0.4]]), crossing_event=no_event, contact=touch, **args)
        result = tracker.update(torch.tensor([[1.5, 1.4, 1.3, 1.2]]), crossing_event=torch.tensor([[False, True]]),
                                contact=no_contact, **args)
        return bool(result.clean_event)
    assert run(strict=False) is True
    assert run(strict=True) is False


def _v2_cli(tmp_path, **override):
    checkpoint = tmp_path / "31625_top1.pt"
    checkpoint.write_bytes(b"x")
    base = dict(resume_checkpoint=str(checkpoint), resume_sha256="ab" * 32, force_ppo_lr=1e-4, blind_init=None,
                distributed=False, rollout_only_steps=None, target_iterations=36700, baseline_gate=None,
                baseline_gate_waiver="user approved v2 verification on 2026-10-06", hash_fn=lambda path: "ab" * 32)
    base.update(override)
    return base


def test_v2_cli_accepts_an_explicitly_pinned_clean_checkpoint(tmp_path):
    task = gap_guard.V2_TASK_IDS[0]
    assert task == "Gd-VrlGapFinetuneCleanV2Raycast-Rbq10-Dreamwaq-v0"
    gap_guard.validate_cli(task, **_v2_cli(tmp_path))
    gap_guard.validate_cli(task, **_v2_cli(tmp_path, rollout_only_steps=50, target_iterations=None,
                                            baseline_gate_waiver=None))


@pytest.mark.parametrize("override", [
    dict(resume_sha256=None), dict(resume_sha256="AB" * 32), dict(hash_fn=lambda path: "cd" * 32),
    dict(baseline_gate_waiver=None), dict(baseline_gate_waiver="short"), dict(baseline_gate="gate.json"),
    dict(force_ppo_lr=None), dict(blind_init="blind.pt"), dict(target_iterations=None),
    dict(resume_checkpoint="/nonexistent.pt"),
])
def test_v2_cli_fails_closed(tmp_path, override):
    with pytest.raises(ValueError):
        gap_guard.validate_cli(gap_guard.V2_TASK_IDS[0], **_v2_cli(tmp_path, **override))


def test_legacy_arms_refuse_resume_sha256(tmp_path):
    with pytest.raises(ValueError):
        gap_guard.validate_cli(gap_guard.GAP_TASK_IDS[2], resume_checkpoint=None, force_ppo_lr=1e-4, blind_init=None,
                               distributed=False, rollout_only_steps=None, resume_sha256="ab" * 32)


def _ckpt(task="Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0", camera="vendor_new", iteration=31625):
    infos = {"gd_lab": {"cenet_optimizer_state_dict": {}, "observation_context": {"version": gap_guard.OBSERVATION_VERSION},
                        "gap_finetune": {"task": task, "camera_profile": camera, "checkpoint": {"iter": 17206}}}}
    return {"model_state_dict": {}, "optimizer_state_dict": {}, "iter": iteration, "infos": infos}


def test_v2_checkpoint_inspection(tmp_path):
    path = tmp_path / "ok.pt"
    torch.save(_ckpt(), path)
    sha = gap_guard.sha256_file(path)
    info = gap_guard.inspect_checkpoint_v2(path, sha, target_iterations=36626)
    assert info["iter"] == 31625 and gap_guard.verify_restored_iteration_v2(31626, info) == 31626
    with pytest.raises(RuntimeError):
        gap_guard.verify_restored_iteration_v2(31625, info)
    with pytest.raises(ValueError):
        gap_guard.inspect_checkpoint_v2(path, sha, target_iterations=31626)  # nothing left to train
    with pytest.raises(ValueError):
        gap_guard.inspect_checkpoint_v2(path, "00" * 32)
    for index, bad in enumerate((_ckpt(task="Gd-VrlGapFinetuneIntrusionRaycast-Rbq10-Dreamwaq-v0"),
                                 _ckpt(camera="vendor_legacy"))):
        bad_path = tmp_path / f"bad{index}.pt"
        torch.save(bad, bad_path)
        with pytest.raises(ValueError):
            gap_guard.inspect_checkpoint_v2(bad_path, gap_guard.sha256_file(bad_path))


def test_v2_config_adds_terms_in_order_and_no_observations():
    tree = ast.parse((ROOT / "src/gd_lab/teachers/bivt/tasks.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GapFinetuneCleanV2RaycastEnvCfg")
    assert [b.id for b in cls.bases] == ["GapFinetuneCleanRaycastEnvCfg"]
    assert "add_v2_terms(self)" in ast.unparse(cls) and "observations" not in ast.unparse(cls)
    module = ast.parse((ROOT / "src/gd_lab/mdp/gap_stair_v2.py").read_text())
    builder = ast.unparse(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "add_v2_terms"))
    assert "observations" not in builder  # a Clean checkpoint must resume unchanged
    assert builder.index("rewards.stair_handle_disturbance") < builder.index("rewards.stair_push_fall")
    assert builder.index("rewards.stair_handle_disturbance") < builder.index("rewards.stair_push_slip")
    assert "strict_contact" in builder and "V2_GAP_WIDTH_RANGE" in builder


def test_gast_teacher_uses_the_identical_v2_objective():
    """BIVT-Ray CleanV2 and the GAST CleanV2 teacher must be trained on the same terms (byte-identical copies)."""
    for name in ("gap_stair_v2.py", "gap_stair_v2_math.py", "gap_stair_v21.py", "platform_gap_attempts.py",
                 "platform_gap_finetune.py"):
        assert (ROOT / "src/gd_lab/mdp" / name).read_bytes() == (ROOT / "gast/src/gd_lab/mdp" / name).read_bytes(), name
    gast_tasks = (ROOT / "gast/src/gd_lab/gast/tasks.py").read_text()
    tree = ast.parse(gast_tasks)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GastGapCleanV2TeacherCfg")
    assert [b.id for b in cls.bases] == ["GastGapCleanTeacherCfg"] and "add_v2_terms(self)" in ast.unparse(cls)
    assert "task='GastGapCleanV2'" in gast_tasks
    scratch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GastScratchV2TeacherCfg")
    source = ast.unparse(scratch)
    assert [b.id for b in scratch.bases] == ["GastTeacherCfg"] and "add_v2_terms(self)" in source
    assert "range_multiplier" not in source  # from scratch keeps the command curriculum
    assert source.index("platform_gap_monitor") < source.index("add_v2_terms(self)")
    assert "task='GastScratchV2'" in gast_tasks


def test_nosing_lips_move_the_drop_off():
    edges = [3.0, 2.7, 2.4]
    assert nosed_edges(edges, 0.0, False) == edges
    assert nosed_edges(edges, 0.03, False) == pytest.approx([3.03, 2.73, 2.43])  # outward lips, every edge
    assert nosed_edges(edges, 0.03, True) == pytest.approx([3.0, 2.67, 2.37])  # inward lips, not the outermost


def test_main_training_adds_short_hard_yanks_within_the_critic_clip():
    module = ast.parse((ROOT / "src/gd_lab/mdp/gap_stair_v2.py").read_text())
    cfg = next(n for n in module.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "V2_DISTURBANCE")
    disturbance = ast.literal_eval(cfg.value)
    assert disturbance["ascend_jerk_prob"] == 0.3
    assert disturbance["ascend_jerk_force"] == (300.0, 400.0) and disturbance["ascend_jerk_duration"] == (0.1, 0.3)
    # critic stamp = F * 0.2 s / mass inside the +-2 observation clip: true with the usual 6 kg payload (46.7 kg);
    # only the extreme corner (400 N on the lightest robot, 38.7 kg) saturates at the clip (2.07), still "maximal".
    assert 400.0 * 0.2 / 46.684 < 2.0
    assert 400.0 * 0.2 / (40.684 - 2.0) < 2.1


def test_v21_speed_mix_centres_on_0p8_to_1p0_and_never_exceeds_the_ceiling():
    from gd_lab.mdp.gap_stair_v2_math import V21_SPEED_MIX, sample_v21_speed, speed_ceiling

    assert abs(sum(p for p, _, _ in V21_SPEED_MIX) - 1.0) < 1e-9
    g = torch.Generator().manual_seed(0)
    v = sample_v21_speed(20000, 1.2, generator=g)
    usual = ((v >= 0.8) & (v <= 1.0)).float().mean()
    assert 0.55 < usual < 0.65 and v.max() <= 1.2 and v.min() >= 0.2
    assert sample_v21_speed(1000, 0.6, generator=g).max() <= 0.6
    assert speed_ceiling(0, 300_000) == 0.6 and speed_ceiling(300_000, 300_000) == 1.2 and speed_ceiling(5, 0) == 1.2


def test_v21_task_and_gast_teachers_use_the_shared_builder():
    tasks = (ROOT / "src/gd_lab/teachers/bivt/tasks.py").read_text()
    assert "class GapFinetuneCleanV21RaycastEnvCfg(GapFinetuneCleanRaycastEnvCfg)" in tasks and "add_v21_terms(self)" in tasks
    assert "CleanV21" in gap_guard.V2_ARMS and "Gd-VrlGapFinetuneCleanV21Raycast-Rbq10-Dreamwaq-v0" in gap_guard.V2_SOURCE_TASKS
    gast = (ROOT / "gast/src/gd_lab/gast/tasks.py").read_text()
    assert "task='GastGapCleanV21'" in gast and "task='GastScratchV21'" in gast
