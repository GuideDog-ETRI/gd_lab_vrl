"""CPU checks for the RVLD/GAVD distillation-loop fixes (no Isaac; script checked at source level)."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from gd_lab.students.gavd.distillation import AttentionDistillation

ROOT = Path(__file__).parents[1]


class _Teacher:
    """act_inference -> +1 everywhere; act_with_terrain_latent -> the latent's first value (0 when blind)."""

    def act_inference(self, obs):
        return torch.ones(obs.shape[0], 2)

    def act_with_terrain_latent(self, obs, latent):
        return latent[:, :1].expand(-1, 2) - 5.0  # distinguishable from the teacher (+1)


def _distillation(stale_student_rollout):
    d = AttentionDistillation(_Teacher(), NS(), 2, "cpu", warmup=0, ramp=1, stale_student_rollout=stale_student_rollout)
    d.ready[:] = True
    d.latent[:] = 1.0
    d.stamp[:] = torch.tensor([0.0, -10.0])  # env 0 fresh, env 1 stale (age 10 s)
    return d


def test_legacy_stale_env_falls_back_to_the_privileged_teacher():
    actions = _distillation(False).actions(torch.zeros(2, 3), iteration=10, current_time_seconds=0.1)
    assert actions[0].tolist() == [-4.0, -4.0]  # fresh: student with its latent
    assert actions[1].tolist() == [1.0, 1.0]  # stale: teacher takes over (legacy behaviour kept by default)


def test_opt_in_stale_env_walks_the_student_blind_route():
    d = _distillation(True)
    actions = d.actions(torch.zeros(2, 3), iteration=10, current_time_seconds=0.1)
    assert actions[0].tolist() == [-4.0, -4.0]
    assert actions[1].tolist() == [-5.0, -5.0]  # zero latent -> blind route, never the teacher
    assert d.metrics["stale_student_rollout_fraction"] == 0.5


def test_default_constructor_keeps_the_legacy_behaviour():
    d = AttentionDistillation(_Teacher(), NS(), 1, "cpu")
    assert d.stale_student_rollout is False


def test_distill_loop_drops_non_finite_rows_before_the_update_and_wires_the_flag():
    source = (ROOT / "scripts/distill_student.py").read_text()
    ast.parse(source)
    bad, drop = source.index("bad_rows = valid & ~finite"), source.index("valid &= finite")
    assert bad < drop < source.index("rows = valid.nonzero(as_tuple=False).flatten()")
    assert "stale_student_rollout=args_cli.stale_student_rollout" in source
    assert source.count("if not args_cli.stale_student_rollout:") == 1  # the RVLD branch
    assert '"--stale_student_rollout",\n    action="store_true",\n    default=False' in source
