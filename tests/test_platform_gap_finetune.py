"""CPU checks for the gap fine-tuning arms (spec: claude_handoff/CLAUDE_FINAL_SPEC_V4.md).

LIMITS: these run tensor/state logic and source-level contracts only. Adapters are executed with
Isaac's ManagerTermBase/SceneEntityCfg replaced by stubs, exactly like tests/test_platform_gap.py.
They do NOT exercise Kit/PhysX, the real RewardManager/CurriculumManager lifecycle, the terrain
importer, or training. Those are the server smoke (rollout-only) and baseline-evaluation gates.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from gd_lab.mdp.platform_gap_attempts import (
    CLEAN,
    EPISODE_END,
    MAX_RECORDS,
    NONCLEAN,
    RETREAT,
    REVERSE,
    EpisodeArchive,
    GapAttemptTracker,
)
from gd_lab.mdp.platform_gap_math import gap_intrusion_cost
from gd_lab.mdp.platform_gap_mesh import build_boarding_mesh
from gd_lab.teachers.bivt import gap_guard

ROOT = Path(__file__).parents[1]
SLOTS = torch.tensor([[[-1.55, -1.45], [1.45, 1.55]]])  # width 0.10, centres +-1.5, tile-centre frame
DECK = torch.zeros(1, 2)


def feet_x(lead, forward=True):
    offsets = torch.tensor([0.0, 0.2, 0.4, 0.6])
    return (lead - offsets if forward else lead + offsets)[None]


def step(tr, lead, *, z=0.03, forward=True, family=True, event=(False, False), contact=False, foot_z=None):
    n = tr.n
    x = feet_x(lead, forward).repeat(n, 1)
    fz = torch.full((n, 4), z) if foot_z is None else foot_z
    return tr.update(
        x, fz, SLOTS.repeat(n, 1, 1), DECK.repeat(n, 1), torch.full((n,), forward), torch.full((n,), family),
        torch.tensor([event]).repeat(n, 1), torch.full((n, 4), contact),
    )


def approach(tr, forward=True):
    sign = 1 if forward else -1
    for lead in (0.5, 1.0):  # 0.45 m from the near edge: no attempt yet
        assert not step(tr, sign * lead, forward=forward).started.any()
    assert step(tr, sign * 1.2, forward=forward).started.all()


# ------------------------------------------------------------------ intrusion cost
def test_intrusion_cost_outside_inside_and_threshold():
    z = torch.tensor([[[0.03, 0.03], [0.03, 0.03], [0.03, 0.03], [0.03, 0.03]]])
    deck, inside = torch.zeros_like(z), torch.zeros_like(z, dtype=torch.bool)
    assert gap_intrusion_cost(z, inside, deck).item() == 0.0
    inside[0, 0, 1] = True  # on deck level: bottom of the sphere at 0
    assert gap_intrusion_cost(z, inside, deck).item() == 0.0
    z[0, 0, 1] = 0.005  # 2.5 cm below the deck: inside the 3 cm allowance
    assert gap_intrusion_cost(z, inside, deck).item() == 0.0
    z[0, 0, 1] = -0.05  # 8 cm below: (0.08 - 0.03) / 0.10 = 0.5 for one of four feet
    assert gap_intrusion_cost(z, inside, deck).item() == pytest.approx(0.125)
    z[0, 0, 0] = -0.05
    inside[0, 0, 0] = True  # same foot in both slots counts once (max), not twice
    assert gap_intrusion_cost(z, inside, deck).item() == pytest.approx(0.125)
    z[0, 1, 1] = -2.0
    inside[0, 1, 1] = True  # saturates at 1 per foot
    assert gap_intrusion_cost(z, inside, deck).item() == pytest.approx(0.375)
    with pytest.raises(ValueError):
        gap_intrusion_cost(z, inside[..., :1], deck)


# ------------------------------------------------------------------ attempt state machine
def test_clean_crossing_pays_once_and_closes():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    for lead in (1.5, 1.7, 1.9, 2.1):
        result = step(tr, lead)
        assert result.cost.item() == 0.0 and not result.clean_event.any()
    result = step(tr, 2.3, event=(False, True))
    assert result.clean_event.all() and result.closed.all() and not tr.active.any()
    assert tr.outcomes[0, 0].item() == 1 and tr.outcomes[0, CLEAN].item() == 1
    assert tr.paid.tolist() == [[False, True]]
    assert not step(tr, 2.4, event=(False, True)).clean_event.any()  # stale event, no attempt: nothing
    assert tr.rec_depth[0, 0].item() == 0.0 and tr.rec_dwell[0, 0].item() > 0
    assert tr.rec_width[0, 0].item() == pytest.approx(0.10)


def test_deep_dip_costs_and_makes_the_crossing_nonclean():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    foot_z = torch.full((1, 4), 0.03)
    foot_z[0, 0] = -0.05  # lead foot at x=1.5 sits in the slot 8 cm below the deck
    result = step(tr, 1.5, foot_z=foot_z)
    assert result.cost.item() == pytest.approx(0.125)
    step(tr, 1.9)
    result = step(tr, 2.3, event=(False, True))
    assert result.closed.all() and not result.clean_event.any()  # nonclean crossing: no bonus
    assert tr.outcomes[0, NONCLEAN].item() == 1 and tr.outcomes[0, CLEAN].item() == 0
    assert tr.rec_depth[0, 0].item() == pytest.approx(0.08, abs=1e-6)
    assert not tr.active.any()  # the attempt is closed even though it was not clean


@pytest.mark.parametrize("z, clean", [(0.005, True), (-0.001, False)])
def test_three_centimetre_depth_boundary(z, clean):
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    foot_z = torch.full((1, 4), 0.03)
    foot_z[0, 0] = z  # centre height; bottom = z - 0.03 against a deck at 0
    step(tr, 1.5, foot_z=foot_z)
    result = step(tr, 2.3, event=(False, True))
    assert bool(result.clean_event.item()) is clean
    assert tr.outcomes[0, CLEAN if clean else NONCLEAN].item() == 1


def test_retreat_closes_and_restart_needs_a_new_step():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    result = step(tr, 0.9)  # 0.55 m behind the near edge (> 0.40)
    assert result.closed.all() and not result.started.any() and not tr.active.any()
    assert tr.outcomes[0, RETREAT].item() == 1
    assert step(tr, 1.2).started.all()  # a fresh approach opens a new attempt on the next step


def test_heading_reversal_fails_the_held_attempt():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    result = step(tr, 1.2, forward=False)
    assert result.closed.all() and not result.started.any()
    assert tr.outcomes[0, REVERSE].item() == 1


def test_episode_end_fails_open_attempts_and_collect_reports_them():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    tr.close_episode([0])
    assert tr.outcomes[0, EPISODE_END].item() == 1 and not tr.active.any()
    stats, records = tr.collect([0])
    assert stats["attempts"] == 1 and stats["fail_episode_end_rate"] == 1.0 and stats["clean_rate"] == 0.0
    assert records[0]["outcome"] == "episode_end" and records[0]["width"] == pytest.approx(0.10)
    assert all(isinstance(v, float) for v in stats.values())  # scalars only: safe for CurriculumManager.reset


def test_attempts_after_the_bonus_are_counted_but_not_paid_again():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr)
    assert step(tr, 2.3, event=(False, True)).clean_event.all()
    assert step(tr, 1.3).started.all()  # diagnostic retry on the same side
    result = step(tr, 2.3, event=(False, True))
    assert result.closed.all() and not result.clean_event.any()
    assert tr.outcomes[0, 0].item() == 2 and tr.outcomes[0, CLEAN].item() == 2


def test_each_direction_has_its_own_bonus():
    tr = GapAttemptTracker(1, "cpu")
    approach(tr, forward=False)
    result = step(tr, -2.3, forward=False, event=(True, False))
    assert result.clean_event.all() and tr.paid.tolist() == [[True, False]]


def test_non_gap_env_and_idle_stale_events_do_nothing():
    tr = GapAttemptTracker(1, "cpu")
    for lead in (1.0, 1.2, 1.5):
        result = step(tr, lead, family=False, event=(False, True))
        assert not (result.started | result.closed | result.clean_event).any() and result.cost.item() == 0.0
    assert tr.outcomes.sum() == 0 and tr.family_steps.sum() == 0


def test_partial_reset_only_clears_the_requested_envs():
    tr = GapAttemptTracker(2, "cpu")
    assert step(tr, 1.2).started.all()
    tr.reset(torch.tensor([0]))
    assert tr.active.tolist() == [False, True] and tr.outcomes[:, 0].tolist() == [0, 1]
    assert tr.family_steps.tolist() == [0, 1]


def test_record_buffer_overflow_is_counted_not_lost():
    tr = GapAttemptTracker(1, "cpu")
    for _ in range(MAX_RECORDS + 3):
        assert step(tr, 1.2).started.all()
        assert step(tr, 0.9).closed.all()  # retreat
    stats, records = tr.collect([0])
    assert stats["attempts"] == MAX_RECORDS + 3 and len(records) == MAX_RECORDS
    assert stats["record_overflow"] == 3 and stats["fail_retreat_rate"] == 1.0


def test_slot_dwell_and_contact_fractions_are_independent_of_attempts():
    tr = GapAttemptTracker(1, "cpu")
    step(tr, 1.5, contact=True)  # foot 0 in the slot, all feet in contact
    step(tr, 3.5)  # nothing in a slot
    tr.close_episode([0])
    stats, _ = tr.collect([0])
    assert stats["slot_dwell_frac"] == 0.5 and stats["slot_contact_frac"] == 0.5


def test_slice_ids_select_only_the_requested_envs():
    tr = GapAttemptTracker(3, "cpu")
    assert step(tr, 1.2).started.all()
    tr.reset(slice(0, 1))
    assert tr.active.tolist() == [False, True, True]
    tr.close_episode(slice(1, 2))
    assert tr.active.tolist() == [False, False, True] and tr.outcomes[:, EPISODE_END].tolist() == [0, 1, 0]


def test_snapshots_are_additive_and_collect_is_their_composition():
    tr = GapAttemptTracker(2, "cpu")
    step(tr, 1.2)  # both envs open an attempt
    foot_z = torch.full((2, 4), 0.03)
    foot_z[1, 0] = -0.05
    step(tr, 1.5, foot_z=foot_z)
    tr.close_episode(None)
    first, second = tr.snapshot([0]), tr.snapshot([1])
    merged = tr.merge(tr.merge(None, first[0]), second[0])
    whole = tr.snapshot(None)
    assert merged == whole[0] and sorted(map(str, first[1] + second[1])) == sorted(map(str, whole[1]))
    stats, records = tr.collect(None)
    assert stats == tr.summarize(merged, first[1] + second[1]) and len(records) == 2
    assert stats["attempts"] == 2 and stats["deep_intrusion_frac"] == 0.25  # 1 deep env-step of 4


class _MonitorStub:
    def __init__(self, tracker):
        self.tracker = tracker

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)


def test_episode_archive_keeps_only_the_first_episode_of_each_env():
    tr = GapAttemptTracker(2, "cpu")
    monitor = _MonitorStub(tr)
    archive = EpisodeArchive(tr)
    archive.install(monitor)
    assert step(tr, 1.2).started.all()
    monitor.reset([0])  # env 0 finishes its first episode: the open attempt is archived as episode_end
    assert tr.outcomes[0].sum() == 0  # ...and the tracker really is wiped afterwards
    assert step(tr, 1.2).started.tolist() == [True, False]
    monitor.reset([0])  # second episode of env 0: ignored
    stats, records = archive.finish()
    assert archive.truncated == 1 and stats["gap_episodes"] == 2.0  # env 1 never reset: closed as truncated
    assert stats["attempts"] == 2.0 and len(records) == 2  # one attempt per env, none from env 0's second episode
    assert {record["env"] for record in records} == {0, 1} and "reset" not in vars(monitor)


# ------------------------------------------------------------------ tile metadata frame (real mesh)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("via_cache", [False, True])
def test_metadata_is_in_the_env_origin_frame(seed, difficulty, via_cache, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from gd_lab.mdp.platform_gap_metadata import boarding_tile_metadata

    size, offset = (8.0, 8.0), 1.5
    np.random.seed(seed)
    meshes, origin = build_boarding_mesh(difficulty, size, (0.02, 0.24), (0.0, 0.16), offset, 0.65)
    mesh = trimesh.util.concatenate(meshes)
    centre = np.eye(4)  # IsaacLab _get_terrain_mesh (terrain_generator.py:377-382)
    centre[0:2, -1] = -size[0] * 0.5, -size[1] * 0.5
    mesh.apply_transform(centre)
    origin = origin + centre[0:3, -1]
    if via_cache:  # use_cache=True path: OBJ round trip after centring
        mesh.export(tmp_path / "mesh.obj")
        mesh = trimesh.load_mesh(tmp_path / "mesh.obj", process=False)
    meta = boarding_tile_metadata(mesh, offset, None)
    width = 0.02 + difficulty * 0.22
    expected = [(-offset - width / 2, -offset + width / 2), (offset - width / 2, offset + width / 2)]
    np.testing.assert_allclose(meta["slots"], expected, atol=1e-5)
    np.testing.assert_allclose(origin, [0, 0, 0], atol=1e-9)
    np.random.seed(seed)
    landing = np.random.choice([-1.0, 1.0], size=2) * (0.16 * difficulty)
    np.testing.assert_allclose(meta["lower_decks"], [min(landing[0], 0.0), min(0.0, landing[1])], atol=1e-5)
    # after placement the env origin is the tile centre again, so the slots stay centre-relative
    place = np.eye(4)
    place[0:2, -1] = (5 + 0.5) * size[0], (12 + 0.5) * size[1]
    mesh.apply_transform(place)
    assert (origin + place[0:3, -1])[0] == pytest.approx(5.5 * size[0])


# ------------------------------------------------------------------ guards
def _alg(schedule="fixed", floor=3e-5, groups=2):
    class Alg:
        def __init__(self):
            self.schedule, self._min_learning_rate, self._lr = schedule, floor, 5.1e-4
            self.optimizer = NS(param_groups=[{"lr": 5.1e-4} for _ in range(groups)])

        @property
        def learning_rate(self):  # DreamwaqPPO: setter clamps to the floor
            return self._lr

        @learning_rate.setter
        def learning_rate(self, value):
            self._lr = max(self._min_learning_rate, float(value))

    return Alg()


def test_force_ppo_lr_applies_after_load_and_is_verified():
    alg = _alg()
    report = gap_guard.apply_force_ppo_lr(alg, 1e-4)
    assert report["groups"] == [1e-4, 1e-4] and alg.learning_rate == 1e-4 and report["min_learning_rate"] == 3e-5


@pytest.mark.parametrize("alg, lr", [
    (_alg(schedule="adaptive"), 1e-4), (_alg(floor=2e-4), 1e-4), (_alg(), 0.0), (_alg(), float("nan")),
    (_alg(), -1e-4),
])
def test_force_ppo_lr_rejects_unsafe_settings(alg, lr):
    with pytest.raises(ValueError):
        gap_guard.apply_force_ppo_lr(alg, lr)


def test_cenet_lr_is_reported_never_changed():
    policy = NS(cenet=NS(optimizer=NS(param_groups=[{"lr": 1e-3}])))
    assert gap_guard.cenet_lr_report(policy)["as_expected"]
    policy.cenet.optimizer.param_groups[0]["lr"] = 5e-4
    report = gap_guard.cenet_lr_report(policy)
    assert not report["as_expected"] and policy.cenet.optimizer.param_groups[0]["lr"] == 5e-4


def _gate(**override):
    record = {
        "schema": gap_guard.GATE_SCHEMA, "teacher_sha256": gap_guard.PINNED_CHECKPOINT_SHA256,
        "teacher": {"cells": 72, "attempts": 4000},
        "student": {"evaluated": True, "cells": 72, "checkpoint_sha256": "ab" * 32},
        "paired_conditions_identical": True, "flat_stairs_regression": {"evaluated": True, "passed": True},
        "verdict": gap_guard.GATE_UNLOCKING_VERDICT,
    }
    record.update(override)
    return record


def _gate_file(tmp_path, record=None, name="gate.json"):
    path = tmp_path / name
    path.write_text(json.dumps(_gate() if record is None else record))
    return str(path)


def _cli(tmp_path, **override):
    checkpoint = tmp_path / gap_guard.PINNED_CHECKPOINT_NAME
    checkpoint.write_bytes(b"x")
    base = dict(
        resume_checkpoint=str(checkpoint), force_ppo_lr=1e-4, blind_init=None, distributed=False,
        rollout_only_steps=None, target_iterations=21207, hash_fn=lambda path: gap_guard.PINNED_CHECKPOINT_SHA256,
        baseline_gate=_gate_file(tmp_path, name="default_gate.json"),  # never the file a test is probing
    )
    base.update(override)
    return base


def test_cli_accepts_the_safe_configuration(tmp_path):
    for task in gap_guard.GAP_TASK_IDS:
        gap_guard.validate_cli(task, **_cli(tmp_path))
    # the rollout-only smoke may run before the gate exists
    gap_guard.validate_cli(
        gap_guard.GAP_TASK_IDS[0], **_cli(tmp_path, rollout_only_steps=300, target_iterations=None, baseline_gate=None)
    )


@pytest.mark.parametrize("override", [
    dict(force_ppo_lr=None), dict(force_ppo_lr=0.0), dict(force_ppo_lr=float("inf")), dict(resume_checkpoint=None),
    dict(distributed=True), dict(blind_init="blind.pt"), dict(target_iterations=None),
    dict(hash_fn=lambda path: "0" * 64), dict(rollout_only_steps=0, target_iterations=None),
    dict(target_iterations=17207), dict(target_iterations=100),  # resumes at 17207: nothing left to train
    dict(baseline_gate=None), dict(baseline_gate="/nonexistent/gate.json"),
])
def test_cli_fails_closed(tmp_path, override):
    with pytest.raises(ValueError):
        gap_guard.validate_cli(gap_guard.GAP_TASK_IDS[2], **_cli(tmp_path, **override))


def test_cli_rejects_a_wrong_file_name_and_gap_flags_elsewhere(tmp_path):
    wrong = tmp_path / "model_17206.pt"
    wrong.write_bytes(b"x")
    with pytest.raises(ValueError):
        gap_guard.validate_cli(gap_guard.GAP_TASK_IDS[0], **_cli(tmp_path, resume_checkpoint=str(wrong)))
    other = gap_guard.RAYCAST_TASK_IDS[0]
    for flags in (dict(force_ppo_lr=1e-4), dict(rollout_only_steps=10)):
        kwargs = dict(resume_checkpoint=None, force_ppo_lr=None, blind_init=None, distributed=False, rollout_only_steps=None)
        kwargs.update(flags)
        with pytest.raises(ValueError):
            gap_guard.validate_cli(other, **kwargs)
    gap_guard.validate_cli(other, resume_checkpoint=None, force_ppo_lr=None, blind_init=None, distributed=True,
                           rollout_only_steps=None)  # the original task keeps its behaviour


def _checkpoint(**override):
    infos = {"gd_lab": {"cenet_optimizer_state_dict": {}, "learning_rate": 5e-4,
                        "observation_context": {"version": gap_guard.OBSERVATION_VERSION}}}
    full = {"model_state_dict": {}, "optimizer_state_dict": {}, "iter": 17206, "infos": infos}
    full.update(override)
    return full


def test_checkpoint_must_carry_a_complete_resume_state(tmp_path):
    torch.save(_checkpoint(), tmp_path / "ok.pt")
    assert gap_guard.inspect_checkpoint(tmp_path / "ok.pt")["iter"] == 17206
    broken = [{key: value for key, value in _checkpoint().items() if key != missing} for missing in
              ("model_state_dict", "optimizer_state_dict", "iter", "infos")]
    no_cenet = _checkpoint()
    del no_cenet["infos"]["gd_lab"]["cenet_optimizer_state_dict"]  # load() would silently skip it
    wrong_version = _checkpoint()
    wrong_version["infos"]["gd_lab"]["observation_context"] = {"version": "other"}
    broken += [no_cenet, wrong_version, _checkpoint(iter=17207)]
    for index, checkpoint in enumerate(broken):
        torch.save(checkpoint, tmp_path / f"broken{index}.pt")
        with pytest.raises(ValueError):
            gap_guard.inspect_checkpoint(tmp_path / f"broken{index}.pt")


def test_resume_counter_follows_the_dreamwaq_runner_convention():
    runner = (ROOT / "src/gd_lab/rl/runner.py").read_text()
    assert "self.current_learning_iteration += 1" in runner.split("def load")[1].split("def export_inference_state")[0]
    checkpoint = {"iter": 17206}
    assert gap_guard.expected_resumed_iteration(17206) == 17207
    assert gap_guard.verify_restored_iteration(17207, checkpoint) == 17207  # a correct resume must pass
    for wrong in (0, 17206, 17208):
        with pytest.raises(RuntimeError):
            gap_guard.verify_restored_iteration(wrong, checkpoint)
    for unusable in (None, {"iter": 100}):
        with pytest.raises(RuntimeError):
            gap_guard.verify_restored_iteration(17207, unusable)


def test_pinned_hash_matches_the_preserved_package():
    sums = (ROOT / "checkpoints/teachers/bivt/ray_top1_17206_20261003/SHA256SUMS.txt").read_text()
    assert gap_guard.PINNED_CHECKPOINT_SHA256 in sums


# ------------------------------------------------------------------ registration and source contracts
def test_registered_task_ids_match_the_guard():
    from gd_lab.teachers.bivt import GAP_FINETUNE_TASKS

    assert tuple(GAP_FINETUNE_TASKS) == gap_guard.GAP_TASK_IDS
    assert gap_guard.GAP_TASK_IDS == tuple(
        f"Gd-VrlGapFinetune{arm}Raycast-Rbq10-Dreamwaq-v0" for arm in ("Baseline", "Intrusion", "Clean")
    )
    assert set(gap_guard.GAP_TASK_IDS) < set(gap_guard.RAYCAST_TASK_IDS) < set(gap_guard.CAMERA_FREE_TASK_IDS)
    assert "Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0" in gap_guard.RAYCAST_TASK_IDS


def _classes():
    tree = ast.parse((ROOT / "src/gd_lab/teachers/bivt/tasks.py").read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _defaults(cls):
    return {
        node.target.id: ast.literal_eval(node.value) for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, (ast.Constant, ast.UnaryOp))
    }


def test_arms_differ_only_in_the_two_weights():
    classes = _classes()
    arms = {
        "GapFinetuneBaselineRaycastEnvCfg": (0.0, 0.0),
        "GapFinetuneIntrusionRaycastEnvCfg": (-3.0, 0.0),
        "GapFinetuneCleanRaycastEnvCfg": (-3.0, 1.5),
    }
    for name, (intrusion, clean) in arms.items():
        assert _defaults(classes[name]) == {"intrusion_weight": intrusion, "clean_weight": clean}
        assert [type(base).__name__ + ":" + base.id for base in classes[name].bases] == ["Name:GapFinetuneRaycastEnvCfg"]
        assert not [n for n in classes[name].body if isinstance(n, ast.FunctionDef)]  # no per-arm logic
    base = classes["GapFinetuneRaycastEnvCfg"]
    post_init = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    assigned = [
        ast.unparse(stmt.targets[0]) for stmt in post_init.body if isinstance(stmt, ast.Assign)
    ]
    reward_order = [t for t in assigned if t.startswith("self.rewards.")]
    assert reward_order == ["self.rewards.platform_gap_monitor", "self.rewards.platform_gap_intrusion",
                            "self.rewards.platform_gap_clean"]
    assert "self.curriculum.platform_gap_diagnostics" in assigned
    assert "self.scene.terrain.terrain_generator.class_type" in assigned
    text = ast.unparse(post_init)
    assert "range_multiplier" in text and "weight=1.0" in text and "weight=self.intrusion_weight" in text


def test_original_raycast_task_is_not_touched():
    cls = _classes()["BlindStartRaycastEnvCfg"]
    body = ast.unparse(cls)
    assert "RaycastVisibleTerrainDropout" in body and "gap" not in body.lower()


def test_trainer_validates_before_the_simulator_and_uses_the_shared_task_sets():
    source = (ROOT / "scripts/train_bivt.py").read_text()
    assert source.index("validate_cli(") < source.index("AppLauncher(args_cli)")
    assert "CAMERA_FREE_TASK_IDS" in source and "CAMERA_FREE_TASKS" not in source.replace("CAMERA_FREE_TASK_IDS", "")
    assert "args_cli.task in RAYCAST_TASK_IDS" in source and "== 'Gd-VrlBlindStartRaycast" not in source
    call = source.index("gap_manifest = _finalize_gap_run(env")  # the call site, not the definition
    assert source.index("runner.load(resume_path)") < call < source.index("runner.learn(")
    assert call < source.index("_run_rollout_only(env, runner, args_cli.rollout_only_steps)")  # LR is pinned before even the smoke rollout
    assert source.index("inspect_checkpoint(resume_path)") < source.index("runner.load(resume_path)")


def test_launcher_pins_the_safe_settings():
    text = (ROOT / "scripts/run_bivt_gap_finetune.sh").read_text()
    assert "--force_ppo_lr 1e-4" in text and "agent.algorithm.schedule=fixed" in text
    assert "--distributed" not in text.replace("Never distributed", "").replace("--distributed (", "")
    assert "GAP_BASELINE_GATE:?" in text and "--rollout_only_steps" in text
    assert "17206 + 1 + " in text  # load() resumes at iter + 1, so +N updates need target iter + 1 + N
    for setting in ("num_steps_per_env=100", "num_learning_epochs=5", "num_mini_batches=4", "min_learning_rate=0.00003"):
        assert setting in text


def test_runner_saves_the_gap_manifest():
    source = (ROOT / "src/gd_lab/rl/runner.py").read_text()
    assert "extra['gap_finetune'] = self.gap_finetune_manifest" in source


# ------------------------------------------------------------------ adapters (Isaac manager classes stubbed)
class _TermBase:
    def __init__(self, cfg, env):
        self._env = env


def _load_wrapper():
    path = ROOT / "src/gd_lab/mdp/platform_gap_finetune.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
    namespace = dict(
        torch=torch, json=json, Path=Path, ManagerTermBase=_TermBase, SceneEntityCfg=lambda name: NS(name=name),
        CONTACT_FORCE=10.0, GapAttemptTracker=GapAttemptTracker,
        boarding_family_mask=lambda env: env.family, family_column_masks=lambda env: env.family_cols,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class _Scene(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def _wrapper_env(order=None, log_dir=None, family=(True, True), crossing_weight=0.5, robot_names=None, sensor_names=None):
    meta = [[None] * 15 for _ in range(2)]
    for row in range(2):
        for col in range(10, 15):
            meta[row][col] = {"slots": [[-1.55, -1.45], [1.45, 1.55]], "lower_decks": [0.0, 0.0]}
    n = len(family)
    crossing = NS(last_event=torch.zeros(n, 2, dtype=torch.bool))
    names = list(order or ("platform_gap_crossing", "platform_gap_monitor", "platform_gap_intrusion", "platform_gap_clean"))
    holder = {}
    env = NS(
        num_envs=n, device="cpu", step_dt=0.01, common_step_counter=7, cfg=NS(log_dir=log_dir),
        family=torch.tensor(family), family_cols=NS(get=lambda key: None),
        scene=_Scene(
            terrain=NS(cfg=NS(terrain_generator=NS(gap_tile_metadata=meta)), terrain_levels=torch.zeros(n).long(),
                       terrain_types=torch.full((n,), 12)),
            env_origins=torch.zeros(n, 3),
        ),
        reward_manager=NS(active_terms=names, get_term_cfg=lambda name: (
            NS(func=crossing, weight=crossing_weight) if name == "platform_gap_crossing"
            else NS(func=holder["monitor"], weight=1.0))),
    )
    env.family_cols = {"platform_gap": torch.tensor([10, 11, 12, 13, 14])}
    robot = NS(data=NS(body_pos_w=torch.zeros(n, 4, 3), heading_w=torch.zeros(n)))
    if robot_names is not None:
        robot.body_names = robot_names
    force = torch.zeros(n, 4, 3)
    sensor = NS(data=NS(net_forces_w=force))
    if sensor_names is not None:
        sensor.body_names = sensor_names
    env.scene.robot, env.scene.contact_forces = robot, sensor
    return env, crossing, holder


def _monitor(env, holder, module):
    monitor = module["GapMonitor"](None, env)
    holder["monitor"] = monitor
    return monitor


def _call(monitor, env):
    return monitor(env, NS(name="robot", body_ids=slice(None)), NS(name="contact_forces", body_ids=slice(None)))


def test_monitor_returns_exact_zeros_even_for_nan_inputs_and_exposes_results():
    module = _load_wrapper()
    env, crossing, holder = _wrapper_env()
    monitor = _monitor(env, holder, module)
    env.scene.robot.data.body_pos_w[..., 0] = torch.tensor([1.5, 1.3, 1.1, 0.9])
    env.scene.robot.data.body_pos_w[..., 2] = 0.03
    env.scene.robot.data.body_pos_w[0, 0, 2] = -0.05
    value = _call(monitor, env)
    assert torch.equal(value, torch.zeros(2)) and monitor.cost[0].item() == pytest.approx(0.125) and monitor.cost[1] == 0
    env.scene.robot.data.body_pos_w[0, 0, 2] = float("nan")
    assert torch.equal(_call(monitor, env), torch.zeros(2))  # NaN may reach the diagnostics, never the reward


def test_penalty_and_bonus_read_the_monitor_and_scale_by_dt():
    module = _load_wrapper()
    env, crossing, holder = _wrapper_env()
    monitor = _monitor(env, holder, module)
    monitor.cost = torch.tensor([0.25, 0.0])
    monitor.clean_event = torch.tensor([True, False])
    assert torch.equal(module["gap_intrusion_penalty"](env), monitor.cost)
    # RewardManager multiplies every term by weight * dt; the one-shot event must integrate to exactly weight
    assert module["gap_clean_bonus"](env).tolist() == pytest.approx([100.0, 0.0]) and 1.5 * 100.0 * env.step_dt == 1.5


def test_monitor_refuses_a_wrong_term_order():
    module = _load_wrapper()
    for order in (
        ("platform_gap_monitor", "platform_gap_crossing", "platform_gap_intrusion", "platform_gap_clean"),
        ("platform_gap_crossing", "platform_gap_intrusion", "platform_gap_monitor", "platform_gap_clean"),
        ("platform_gap_monitor", "platform_gap_clean"),
    ):
        env, _, holder = _wrapper_env(order=order)
        monitor = _monitor(env, holder, module)
        with pytest.raises(RuntimeError):
            _call(monitor, env)
    env, _, holder = _wrapper_env(order=("x", "platform_gap_crossing", "platform_gap_monitor", "platform_gap_clean"))
    _call(_monitor(env, holder, module), env)  # an absent (zero-weight is still registered) intrusion term is fine


def test_monitor_pays_the_clean_bonus_end_to_end_with_stub_env():
    module = _load_wrapper()
    env, crossing, holder = _wrapper_env(family=(True, False))
    monitor = _monitor(env, holder, module)
    x, z = env.scene.robot.data.body_pos_w[..., 0], env.scene.robot.data.body_pos_w[..., 2]
    z[:] = 0.03
    for lead in (1.2, 1.5, 1.9, 2.3):
        x[:] = torch.tensor([lead, lead - 0.2, lead - 0.4, lead - 0.6])
        _call(monitor, env)
    crossing.last_event[:, 1] = True
    _call(monitor, env)
    assert monitor.clean_event.tolist() == [True, False]  # the non-gap env never pays
    assert module["gap_clean_bonus"](env)[0].item() == pytest.approx(100.0)
    _call(monitor, env)
    assert not monitor.clean_event.any()


def test_diagnostics_term_returns_scalars_and_writes_raw_records(tmp_path):
    module = _load_wrapper()
    env, crossing, holder = _wrapper_env(log_dir=str(tmp_path / "run"))
    monitor = _monitor(env, holder, module)
    x, z = env.scene.robot.data.body_pos_w[..., 0], env.scene.robot.data.body_pos_w[..., 2]
    z[:] = 0.03
    x[:] = torch.tensor([1.2, 1.0, 0.8, 0.6])
    _call(monitor, env)  # both envs open an attempt
    stats = module["platform_gap_diagnostics"](env, slice(None))
    assert stats["attempts"] == 2.0 and stats["fail_episode_end_rate"] == 1.0
    assert all(isinstance(value, float) for value in stats.values())
    lines = (tmp_path / "run" / "gap_attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["step"] == 7 and json.loads(lines[0])["outcome"] == "episode_end"
    monitor.reset(slice(None))  # reward_manager.reset follows the curriculum term
    assert monitor.tracker.outcomes.sum() == 0
    assert module["platform_gap_diagnostics"](env, torch.tensor([0, 1])) == {}  # nothing left to report


def test_geometry_requires_metadata_for_every_gap_tile():
    module = _load_wrapper()
    env, _, _ = _wrapper_env()
    env.scene.terrain.cfg.terrain_generator.gap_tile_metadata[1][13] = None
    with pytest.raises(ValueError):
        module["GapTileGeometry"](env)
    del env.scene.terrain.cfg.terrain_generator.gap_tile_metadata
    with pytest.raises(ValueError):
        module["GapTileGeometry"](env)


def test_evaluator_is_read_only_and_never_trains():
    source = (ROOT / "scripts/eval_gap_cases.py").read_text()
    tree = ast.parse(source)  # syntax only: the script has never been run (needs Isaac)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not {c for c in calls if c.endswith((".learn", "alg.update", "optimizer.step", ".backward"))}
    assert "UNVERIFIED" in source.split('"""')[1]
    assert "class_type" not in source and "intrusion_weight" not in source  # it must not alter the arm's rewards


def test_monitor_requires_a_live_crossing_term_and_matching_foot_order():
    module = _load_wrapper()
    env, _, holder = _wrapper_env(crossing_weight=0.0)  # a zero-weight crossing term is never executed
    with pytest.raises(RuntimeError):
        _call(_monitor(env, holder, module), env)
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    shuffled = ["FR_foot", "FL_foot", "RL_foot", "RR_foot"]
    env, _, holder = _wrapper_env(robot_names=["trunk", *feet], sensor_names=["trunk", *shuffled])
    env.scene.robot.body_names, env.scene.contact_forces.body_names = feet, shuffled
    with pytest.raises(RuntimeError):
        _call(_monitor(env, holder, module), env)
    env, _, holder = _wrapper_env(robot_names=feet, sensor_names=feet)
    _call(_monitor(env, holder, module), env)  # same feet, same order: fine


@pytest.mark.parametrize("bad", [
    dict(schema="other"), dict(teacher_sha256="0" * 64), dict(verdict="teacher_clean_student_problem"),
    dict(verdict="evaluator_error"), dict(verdict=None), dict(teacher={"cells": 0, "attempts": 5}),
    dict(teacher={"cells": 72}), dict(teacher={"cells": True, "attempts": 5}),
    dict(student={"evaluated": False, "cells": 72, "checkpoint_sha256": "ab" * 32}),
    dict(student={"evaluated": True, "cells": 72, "checkpoint_sha256": "short"}), dict(student={}),
    dict(paired_conditions_identical=False), dict(flat_stairs_regression={"evaluated": True, "passed": False}),
    dict(flat_stairs_regression={"evaluated": False, "passed": True}), dict(flat_stairs_regression=None),
])
def test_training_gate_rejects_every_incomplete_record(tmp_path, bad):
    path = _gate_file(tmp_path, _gate(**bad))
    with pytest.raises(ValueError):
        gap_guard.validate_gate_record(path)
    with pytest.raises(ValueError):  # and the CLI refuses to train with it
        gap_guard.validate_cli(gap_guard.GAP_TASK_IDS[1], **_cli(tmp_path, baseline_gate=path))


def test_training_gate_rejects_missing_empty_and_garbage_files(tmp_path):
    for name, text in (("empty.json", ""), ("garbage.json", "not json"), ("list.json", "[]"), ("null.json", "null")):
        (tmp_path / name).write_text(text)
        with pytest.raises(ValueError):
            gap_guard.validate_gate_record(tmp_path / name)
    with pytest.raises(ValueError):
        gap_guard.validate_gate_record(tmp_path / "absent.json")
    assert gap_guard.validate_gate_record(_gate_file(tmp_path))["verdict"] == "teacher_problem_reproduced"


def test_the_evaluator_cannot_create_or_satisfy_the_gate():
    source = (ROOT / "scripts/eval_gap_cases.py").read_text()
    assert "gap_baseline_gate" not in source.replace("schema gap_baseline_gate_v1", "")
    assert "INCOMPLETE" in source.split('"' * 3)[1] and "validate_gate_record" not in source
    assert "GATE_SCHEMA" not in source and "verdict" not in source.lower().replace("the gate-c verdict", "")
    trainer = (ROOT / "scripts/train_bivt.py").read_text()
    assert "validate_gate_record(args_cli.baseline_gate)" in trainer  # in-process defense, not only the launcher
    assert trainer.index("validate_gate_record(args_cli.baseline_gate)") < trainer.index("runner.learn(")


def test_launcher_uses_local_container_defaults_and_hashes_the_whole_implementation():
    text = (ROOT / "scripts/run_bivt_gap_finetune.sh").read_text()
    assert "${GD_LAB_SIF:-/home/user/workspace/gd_lab_isaaclab.sif}" in text
    assert "${GD_LAB_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}" in text
    assert ":-/data/users" not in text  # the other server's paths are only documented as overrides
    assert "git ls-files --others --exclude-standard" in text and "full_patch | sha256sum" in text
    assert '--baseline_gate "$GAP_BASELINE_GATE"' in text


def test_archive_and_curriculum_close_do_not_double_count_a_keyword_reset():
    module = _load_wrapper()
    env, _, holder = _wrapper_env()
    monitor = _monitor(env, holder, module)
    archive = EpisodeArchive(monitor.tracker)
    archive.install(monitor)
    x, z = env.scene.robot.data.body_pos_w[..., 0], env.scene.robot.data.body_pos_w[..., 2]
    z[:] = 0.03
    x[:] = torch.tensor([1.2, 1.0, 0.8, 0.6])
    _call(monitor, env)  # both envs open an attempt
    ids = torch.tensor([0, 1])
    module["platform_gap_diagnostics"](env, ids)  # CurriculumManager.compute runs first and closes the attempts
    monitor.reset(env_ids=ids)  # then RewardManager.reset calls term.reset(env_ids=...) with a keyword
    stats, records = archive.finish()
    assert stats["attempts"] == 2.0 and len(records) == 2 and archive.truncated == 0
    assert monitor.tracker.outcomes.sum() == 0 and "reset" not in vars(monitor)


def test_foot_order_check_says_so_when_it_cannot_run(capsys):
    module = _load_wrapper()
    env, _, holder = _wrapper_env()  # stub robot/sensor without body_names
    _call(_monitor(env, holder, module), env)
    assert "foot-order check skipped" in capsys.readouterr().out
