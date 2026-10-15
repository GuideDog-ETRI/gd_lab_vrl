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


def cell(row, col):
    m = torch.zeros(1, 11, 17, dtype=torch.bool); m[0, row, col] = True
    return m.flatten(1)


def pose(x=0., y=0., yaw=0.):
    return torch.tensor([[x, y, yaw]])


def step(s, visible, p, t, received=True):
    return s.update(torch.tensor([0]), visible, p, torch.tensor([t]), torch.tensor([received])).view(11, 17)


def test_single_cell_survives_sub_cell_steps_and_moves_with_the_body():
    s = SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=0.08)
    none = torch.zeros(1, 187, dtype=torch.bool)
    assert step(s, cell(5, 16), pose(), 0.0)[5, 16]
    for i in range(1, 7):                                   # six 0.05 m steps = 0.3 m
        g = step(s, none, pose(0.05 * i), 0.1 * i)
        assert 1 <= g.sum() <= 2, i                         # never lost; an exact half-cell tie may mark 2
    assert g[5, 13]                                         # 0.8 - 0.3 = 0.5 m ahead
    assert step(s, none, pose(0.05 * 6), 0.65).any()        # still inside the window when stationary


def test_rotation_out_of_range_expiry_and_reset():
    s = SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=0.08)
    none = torch.zeros(1, 187, dtype=torch.bool)
    step(s, cell(5, 13), pose(), 0.0)                       # 0.5 m ahead
    g = step(s, none, pose(yaw=torch.pi / 2), 0.1)          # turn left 90 deg: now on the right
    assert g[0, 8] and g.sum() == 1
    assert not step(s, none, pose(2.0), 0.2).any()          # moved 2 m: outside the grid
    s.reset(torch.tensor([True]))
    step(s, cell(5, 8), pose(), 10.0)
    assert step(s, none, pose(), 11.9).any() and not step(s, none, pose(), 12.1).any()   # 2 s window


def test_frames_the_student_did_not_get_are_not_recorded():
    s = SeenCells(2, "cpu", window_s=2.0, min_capture_period_s=0.08)
    vis = torch.zeros(2, 187, dtype=torch.bool); vis[:, 100] = True
    out = s.update(torch.tensor([0, 1]), vis, torch.zeros(2, 3), torch.tensor([0., 0.]), torch.tensor([True, False]))
    assert out[0, 100] and not out[1].any()
    out = s.update(torch.tensor([1]), torch.zeros(1, 187, dtype=torch.bool), torch.zeros(1, 3), torch.tensor([.1]))
    assert not out.any()                                    # rows update independently


def test_nearest_mode_keeps_the_default_warp_unchanged():
    m = torch.rand(3, 187, 32)
    a, b = torch.rand(3, 3), torch.rand(3, 3)
    assert torch.equal(geometry.warp_memory(m, a, b), geometry.warp_memory(m, a, b, mode="bilinear"))


def test_small_move_and_turn_keeps_a_single_cell_codex_case():
    """Codex 2nd review: backward nearest lookup lost this centre cell (1 -> 0) although it stays in the grid."""
    s = SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=0.08)
    none = torch.zeros(1, 187, dtype=torch.bool)
    step(s, cell(5, 8), pose(), 0.0)
    g = step(s, none, pose(-0.0725515, 0.0204573, 0.4284343), 0.1)
    assert g.sum() >= 1 and g[5, 9]                        # new body coordinates ~(0.0575, -0.0487) m


def test_random_small_moves_never_lose_a_cell_inside_the_grid():
    gen = torch.Generator().manual_seed(1)
    n = 4000
    s = SeenCells(n, "cpu", window_s=2.0, min_capture_period_s=0.08)
    rows = torch.arange(n)
    rc = torch.stack((torch.randint(1, 10, (n,), generator=gen), torch.randint(1, 16, (n,), generator=gen)), 1)
    vis = torch.zeros(n, 11, 17, dtype=torch.bool); vis[rows, rc[:, 0], rc[:, 1]] = True
    s.update(rows, vis.flatten(1), torch.zeros(n, 3), torch.zeros(n))
    p = torch.cat((torch.rand(n, 2, generator=gen) * .16 - .08, torch.rand(n, 1, generator=gen) * 1.2 - .6), 1)
    out = s.update(rows, torch.zeros(n, 187, dtype=torch.bool), p, torch.full((n,), .1), torch.zeros(n, dtype=torch.bool))
    # the cell centre in the new body frame
    qx, qy = (rc[:, 1] * .1 - .8).float(), (rc[:, 0] * .1 - .5).float()
    c, si = p[:, 2].cos(), p[:, 2].sin()
    nx, ny = c * (qx - p[:, 0]) + si * (qy - p[:, 1]), -si * (qx - p[:, 0]) + c * (qy - p[:, 1])
    inside = (nx.abs() <= .85) & (ny.abs() <= .55)
    assert inside.sum() > n * .9
    assert out[inside].any(1).all()


def test_rotated_area_has_no_holes():
    s = SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=0.08)
    step(s, torch.ones(1, 187, dtype=torch.bool), pose(), 0.0)
    g = step(s, torch.zeros(1, 187, dtype=torch.bool), pose(yaw=0.3), 0.1)
    assert g[3:8, 5:12].all()                               # interior of the rotated full grid stays covered


def keep_first_snapshot(period, write_every):
    s = SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=period)
    none = torch.zeros(1, 187, dtype=torch.bool)
    step(s, cell(5, 8), pose(), 0.0)
    t, ok = 0.0, True
    for i in range(1, 400):
        t = round(i * write_every, 6)
        if t >= 2.0:
            break
        ok &= bool(step(s, none, pose(), t)[5, 8])
    expired = not step(s, none, pose(), 2.0)[5, 8]
    return ok, expired


def test_ring_capacity_follows_the_capture_interval():
    # default transport 70-100 ms at 20 ms policy steps -> 80 ms minimum spacing
    assert keep_first_snapshot(0.08, 0.08) == (True, True)
    assert keep_first_snapshot(0.08, 0.1) == (True, True)
    # Codex case: 10 ms captures; a 70 ms-sized ring overwrote the first snapshot at 0.31 s
    assert keep_first_snapshot(0.01, 0.01) == (True, True)
    s = SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=0.01)
    assert s.stamps.shape[1] >= 201
    # a ring sized for a longer period than the real one grows instead of overwriting a live snapshot
    assert keep_first_snapshot(0.1, 0.01) == (True, True)


def test_splat_inverts_the_backward_lookup_on_whole_cell_moves():
    m = torch.zeros(2, 187, dtype=torch.bool); m[:, 5 * 17 + 8] = True; m[:, 2 * 17 + 3] = True
    a = torch.tensor([[1., 2., .7], [0., 0., 0.]])
    b = a.clone(); b[0, :2] += torch.tensor([math.cos(.7), math.sin(.7)]) * .2; b[1, 1] += .1   # 2 cells ahead, 1 left
    back = geometry.warp_memory(m[..., None].float(), a, b, mode="nearest")[..., 0] > .5
    assert torch.equal(geometry.splat_cells(m, a, b), back)


def test_seen_requires_the_capture_period():
    import pytest
    with pytest.raises(ValueError):
        SeenCells(1, "cpu", window_s=2.0, min_capture_period_s=0)
