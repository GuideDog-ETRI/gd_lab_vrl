"""Fail-closed guards for the gap fine-tuning tasks (no simulator imports; CPU-testable).

The three arms must resume exactly the frozen 17206 Top-1 teacher, with a fixed PPO
learning rate that is applied *after* ``runner.load`` (which restores the saved LR).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from gd_lab.core.registry import make_task_id

GAP_ARMS = ("Baseline", "Intrusion", "Clean")
GAP_TASK_IDS = tuple(
    make_task_id(task=f"VrlGapFinetune{arm}Raycast", robot="Rbq10", method="Dreamwaq") for arm in GAP_ARMS
)
RAYCAST_TASK_IDS = (make_task_id(task="VrlBlindStartRaycast", robot="Rbq10", method="Dreamwaq"), *GAP_TASK_IDS)
CAMERA_FREE_TASK_IDS = (make_task_id(task="VrlBlindStart", robot="Rbq10", method="Dreamwaq"), *RAYCAST_TASK_IDS)

PINNED_CHECKPOINT_NAME = "17206_top1.pt"
PINNED_CHECKPOINT_SHA256 = "a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d"
PINNED_CHECKPOINT_ITER = 17206  # the file's own ``iter`` (audited read-only); DreamwaqRunner.load resumes at iter + 1
OBSERVATION_VERSION = "bivt_ray_occlusion_v2"
REQUIRED_CHECKPOINT_KEYS = ("model_state_dict", "optimizer_state_dict", "iter")
EXPECTED_CENET_LR = 1.0e-3

# A training run needs a positive, structured gate-C record written AFTER a real paired evaluation
# (teacher 17206 and the GAST student on identical conditions, plus the flat/stairs regression set).
# Nothing in this repository writes such a record: scripts/eval_gap_cases.py is an incomplete,
# unexecuted teacher-only draft and must never create or satisfy it.
GATE_SCHEMA = "gap_baseline_gate_v1"
GATE_UNLOCKING_VERDICT = "teacher_problem_reproduced"
# Instead of a gate record, the user may waive gate C for a run; the reason is recorded in the manifest.
MIN_WAIVER_REASON = 20


def arm_of(task: str) -> str | None:
    return GAP_ARMS[GAP_TASK_IDS.index(task)].lower() if task in GAP_TASK_IDS else None


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_gate_record(path) -> dict:
    """Fail closed unless ``path`` is a structured positive gate-C verdict for the pinned teacher."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"baseline gate {path} is missing, empty or not JSON: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema") != GATE_SCHEMA:
        raise ValueError(f"baseline gate must be a JSON object with schema {GATE_SCHEMA!r}")
    if record.get("teacher_sha256") != PINNED_CHECKPOINT_SHA256:
        raise ValueError("baseline gate was not produced for the pinned 17206 teacher")
    if record.get("verdict") != GATE_UNLOCKING_VERDICT:
        raise ValueError(
            f"baseline gate verdict {record.get('verdict')!r} does not unlock teacher fine-tuning "
            f"(only {GATE_UNLOCKING_VERDICT!r} does; a clean teacher or an evaluator error must not train)"
        )
    positive_int = lambda value: isinstance(value, int) and not isinstance(value, bool) and value > 0  # noqa: E731
    teacher, student = record.get("teacher") or {}, record.get("student") or {}
    if not (positive_int(teacher.get("cells")) and positive_int(teacher.get("attempts"))):
        raise ValueError("baseline gate: the teacher evaluation has no cells/attempts")
    student_sha = student.get("checkpoint_sha256")
    if not (student.get("evaluated") is True and positive_int(student.get("cells"))
            and isinstance(student_sha, str) and len(student_sha) == 64):
        raise ValueError("baseline gate: the paired GAST student was not evaluated")
    if record.get("paired_conditions_identical") is not True:
        raise ValueError("baseline gate: teacher and student were not evaluated on identical conditions")
    regression = record.get("flat_stairs_regression") or {}
    if regression.get("evaluated") is not True or regression.get("passed") is not True:
        raise ValueError("baseline gate: the flat/stairs regression set is missing or failed")
    return record


def validate_cli(
    task, *, resume_checkpoint, force_ppo_lr, blind_init, distributed, rollout_only_steps, target_iterations=None,
    baseline_gate=None, baseline_gate_waiver=None, hash_fn=sha256_file,
):
    """Raise ValueError for any unsafe combination. Cheap: no torch, no simulator."""
    if task not in GAP_TASK_IDS:
        if force_ppo_lr is not None or rollout_only_steps is not None:
            raise ValueError("--force_ppo_lr/--rollout_only_steps are only valid for the gap fine-tuning tasks")
        return
    if distributed:
        # PPO/CENet gradients, optimizer steps, normalizers and the AdaBoot draw are rank-synchronized by
        # gd_lab.rl.distributed_sync (tests/test_rl_distributed_sync.py); refuse if that layer is absent.
        from gd_lab.rl import distributed_sync  # noqa: F401
    if blind_init is not None:
        raise ValueError("gap fine-tuning must resume the 17206 teacher, not --blind_init")
    if force_ppo_lr is None or not math.isfinite(force_ppo_lr) or force_ppo_lr <= 0:
        raise ValueError("gap fine-tuning requires --force_ppo_lr <finite positive value>")
    if rollout_only_steps is not None and rollout_only_steps <= 0:
        raise ValueError("--rollout_only_steps must be positive")
    if rollout_only_steps is None and target_iterations is None:
        raise ValueError("gap fine-tuning requires --target_iterations (total completed updates)")
    if target_iterations is not None and target_iterations <= PINNED_CHECKPOINT_ITER + 1:
        raise ValueError(
            f"--target_iterations {target_iterations} leaves no update after resuming at {PINNED_CHECKPOINT_ITER + 1}"
        )
    if rollout_only_steps is None:  # training: a gate record or an explicit waiver; the smoke may run before it
        if baseline_gate and baseline_gate_waiver:
            raise ValueError("give either --baseline_gate or --baseline_gate_waiver, not both")
        if baseline_gate_waiver is not None:
            if len(str(baseline_gate_waiver).strip()) < MIN_WAIVER_REASON:
                raise ValueError(f"--baseline_gate_waiver needs a reason of at least {MIN_WAIVER_REASON} characters")
        elif not baseline_gate:
            raise ValueError("gap fine-tuning requires --baseline_gate <passed gate-C record> "
                             "or --baseline_gate_waiver '<who decided and why>'")
        else:
            validate_gate_record(baseline_gate)
    if not resume_checkpoint:
        raise ValueError(f"gap fine-tuning requires --resume_checkpoint .../{PINNED_CHECKPOINT_NAME}")
    path = Path(resume_checkpoint)
    if path.name != PINNED_CHECKPOINT_NAME or not path.is_file():
        raise ValueError(f"resume checkpoint must be an existing {PINNED_CHECKPOINT_NAME}: {path}")
    digest = hash_fn(path)
    if digest != PINNED_CHECKPOINT_SHA256:
        raise ValueError(f"resume checkpoint SHA256 mismatch: {digest} != {PINNED_CHECKPOINT_SHA256}")


def inspect_checkpoint(path) -> dict:
    """After the app has launched: the file must carry everything a faithful resume needs."""
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in loaded]
    if missing:
        raise ValueError(f"checkpoint lacks required state {missing}; refusing to fall back to another policy")
    extra = (loaded.get("infos") or {}).get("gd_lab") or {}
    if "cenet_optimizer_state_dict" not in extra:
        raise ValueError("checkpoint lacks the CENet optimizer state (infos.gd_lab); runner.load would skip it silently")
    version = (extra.get("observation_context") or {}).get("version")
    if version != OBSERVATION_VERSION:
        raise ValueError(f"checkpoint observation contract {version!r} != {OBSERVATION_VERSION!r}")
    if int(loaded["iter"]) != PINNED_CHECKPOINT_ITER:
        raise ValueError(f"checkpoint iter {loaded['iter']} != pinned {PINNED_CHECKPOINT_ITER}")
    return {"iter": int(loaded["iter"]), "sha256": sha256_file(path)}


def expected_resumed_iteration(checkpoint_iter: int) -> int:
    """``DreamwaqRunner.load`` ends with ``current_learning_iteration += 1`` (the file stores the finished one)."""
    return int(checkpoint_iter) + 1


def verify_restored_iteration(runner_iteration: int, checkpoint: dict | None) -> int:
    if checkpoint is None or checkpoint["iter"] != PINNED_CHECKPOINT_ITER:
        raise RuntimeError("gap fine-tuning: the pinned 17206 checkpoint was not inspected")
    expected = expected_resumed_iteration(checkpoint["iter"])
    if int(runner_iteration) != expected:
        raise RuntimeError(f"resume counter {runner_iteration} != {expected}: the 17206 state was not restored")
    return expected


def apply_force_ppo_lr(alg, requested: float) -> dict:
    """Set a fixed PPO LR after every restore step and prove it took effect."""
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError(f"invalid PPO learning rate {requested}")
    if getattr(alg, "schedule", None) != "fixed":
        raise ValueError(f"--force_ppo_lr needs agent.algorithm.schedule=fixed, got {getattr(alg, 'schedule', None)!r}")
    floor = float(getattr(alg, "_min_learning_rate", 0.0))
    if floor > requested:
        raise ValueError(f"min_learning_rate {floor} is above the requested {requested}; lower it explicitly")
    alg.learning_rate = requested
    if alg.learning_rate != requested:
        raise ValueError(f"learning_rate setter altered the value: {alg.learning_rate} != {requested}")
    for group in alg.optimizer.param_groups:
        group["lr"] = alg.learning_rate
    groups = [float(group["lr"]) for group in alg.optimizer.param_groups]
    if any(value != requested for value in groups):
        raise ValueError(f"optimizer groups disagree with the requested LR: {groups}")
    return {"requested": requested, "getter": float(alg.learning_rate), "groups": groups,
            "min_learning_rate": floor, "schedule": alg.schedule}


def cenet_lr_report(policy, expected: float = EXPECTED_CENET_LR) -> dict:
    """CENet has its own optimizer (restored by runner.load); report, never change."""
    groups = [float(group["lr"]) for group in policy.cenet.optimizer.param_groups]
    return {"groups": groups, "expected": expected, "as_expected": all(value == expected for value in groups)}
