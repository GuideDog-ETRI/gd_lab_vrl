"""Seen-cell supervision: off = original behavior; on = recently seen cells get clean labels."""
import math
import subprocess

import torch

from gd_lab.gast import geometry
from gd_lab.gast.geometry import SeenCells, reconstruction_loss, targets


def batch(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(b, 187, generator=g) * 0.5
    valid = torch.ones(b, 187)
    gap = torch.zeros(b, 11, 17); gap[:, :, 10:12] = 1
    known = torch.ones(b, 187)
    clean = torch.cat((h, valid, gap.flatten(1), known), 1)
    visible = torch.zeros(b, 11, 17); visible[:, :, :8] = 1             # gap columns 10-11 not visible
    teacher = torch.cat((torch.where(visible.flatten(1) > 0, h, 0.), visible.flatten(1)), 1)
    return clean, teacher


def original_targets_source():
    """targets() at the commit before this change, to prove the default path is unchanged."""
    src = subprocess.run(["git", "show", "8b09c46:src/gd_lab/gast/geometry.py"], capture_output=True, text=True,
                         check=True).stdout
    scope = {}
    exec(compile(src, "geometry_8b09c46", "exec"), scope)
    return scope["targets"], scope["reconstruction_loss"]


def test_default_path_identical_to_previous_commit():
    old_targets, old_loss = original_targets_source()
    clean, teacher = batch()
    pred = torch.randn(4, 187, 6)
    for args in ((clean, teacher), (clean, None)):
        y0, m0 = old_targets(*args); y1, m1 = targets(*args)
        assert torch.equal(y0, y1) and torch.equal(m0, m1)
    assert torch.equal(old_loss(pred, clean, teacher), reconstruction_loss(pred, clean, teacher))


def test_seen_cells_add_gap_supervision_with_clean_labels():
    clean, teacher = batch()
    _, mask0 = targets(clean, teacher)
    gap_cells = clean[:, 374:561].bool()
    assert mask0[..., 2][gap_cells].sum() == 0                          # gap never supervised without memory
    seen = torch.zeros(4, 187, dtype=torch.bool); seen.view(4, 11, 17)[:, :, 10:12] = True
    y1, mask1 = targets(clean, teacher, seen)
    assert mask1[..., 2][gap_cells].all()                               # now supervised
    assert torch.allclose(y1[..., 0], torch.nan_to_num(clean[:, :187] / 5))  # clean height labels
    assert torch.equal(y1[..., 1], teacher[:, 187:])                    # visibility target unchanged


def test_seen_cells_follow_body_motion_and_expire():
    s = SeenCells(1, "cpu", window_s=1.0)
    vis = torch.zeros(1, 11, 17, dtype=torch.bool); vis[0, 5, 16] = True    # a cell 0.8 m ahead
    m = s.update(torch.tensor([0]), vis.flatten(1), torch.tensor([[0., 0., 0.]]), torch.tensor([0.]))
    assert m.view(11, 17)[5, 16]
    none = torch.zeros(1, 187, dtype=torch.bool)
    m = s.update(torch.tensor([0]), none, torch.tensor([[0.3, 0., 0.]]), torch.tensor([0.5]))   # moved 0.3 m
    grid = m.view(11, 17)
    assert grid[5, 13] and not grid[5, 16]                                  # now 0.5 m ahead
    m = s.update(torch.tensor([0]), none, torch.tensor([[0.3, 0., 0.]]), torch.tensor([2.0]))   # 1.5 s later
    assert not m.any()                                                      # expired
    s.reset(torch.tensor([True]))
    assert s.value.sum() == 0
