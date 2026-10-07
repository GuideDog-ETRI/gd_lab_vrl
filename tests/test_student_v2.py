"""v2 student distillation common pieces (RVLD/GAVD in scripts/distill_student.py; GAST in gast/). CPU only."""

import ast
from pathlib import Path

import torch
from torch.nn import functional as F

from gd_lab.students.gap_focus import near_gap_from_scan, row_mse, row_weights, weighted_mean
from gd_lab.students.gavd.model import spatial_loss

ROOT = Path(__file__).resolve().parents[1]


def test_near_gap_needs_a_deep_hit_on_a_gap_tile():
    ray_z = torch.tensor([[0.0, 0.0, -1.5], [0.0, -0.1, 0.0], [0.0, float("nan"), -1.5]])
    origin = torch.zeros(3)
    family = torch.tensor([True, True, False])
    assert near_gap_from_scan(ray_z, origin, family).tolist() == [True, False, False]


def test_unit_weights_reproduce_the_plain_losses():
    torch.manual_seed(0)
    a, b = torch.rand(6, 12), torch.rand(6, 12)
    assert torch.allclose(weighted_mean(row_mse(a, b), torch.ones(6)), F.mse_loss(a, b))
    assert row_weights(torch.tensor([True, False]), 4.0).tolist() == [4.0, 1.0]
    pred = torch.randn(3, 187, 5)
    terrain = torch.cat((torch.rand(3, 187) * 5, (torch.rand(3, 187) > .3).float()), -1)
    plain = spatial_loss(pred, terrain, return_components=True)
    ones = spatial_loss(pred, terrain, return_components=True, row_weight=torch.ones(3))
    for x, y in zip(plain, ones):
        assert torch.allclose(x, y)


def test_top5_copy_matches_the_gast_board():
    root = (ROOT / "src/gd_lab/students/top5.py").read_text().splitlines()[1:]
    gast = (ROOT / "src/gd_lab/gast/student_top5.py").read_text().splitlines()[1:]
    assert root == gast


def test_distill_student_wires_near_gap_and_both_boards():
    source = (ROOT / "scripts/distill_student.py").read_text()
    assert "near_gap_envs(env))" in source  # packet slot [9]
    assert "packet.payload[:10]" in source and "packet.payload[10]" in source and "packet.payload[9]" not in source
    assert 'StudentTop5(os.path.join(log_dir, "top5")' in source and '"top5_gap"' in source
    assert "add_student_v2_env(env_cfg)" in source and 'GD_LAB_V2_FORCE_RAMP_STEPS"] = "0"' in source
    gast = (ROOT / "scripts/gast/train_student_live.py").read_text()
    assert "add_student_v2_env(env_cfg)" in gast and "set_gap_terrain_columns" in gast


def test_student_env_adds_only_the_disturbance():
    module = ast.parse((ROOT / "src/gd_lab/mdp/gap_stair_v2.py").read_text())
    fn = ast.unparse(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "add_student_v2_env"))
    assert "rewards.stair_handle_disturbance" in fn and "V2_DISTURBANCE" in fn and "V2_GAP_WIDTH_RANGE" in fn
    assert "observations" not in fn and "gap_foothold_margin" not in fn
