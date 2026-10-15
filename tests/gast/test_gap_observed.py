"""Gap-observed (free-space) supervision: label-side gap channel only; default path unchanged."""
import math

import torch

from gd_lab.gast.geometry import gap_lift_heights, reconstruction_loss, targets
from gd_lab.teachers.bivt.raycast_visibility import raycast_freespace_points


def test_lift_uses_the_lower_adjacent_lip_and_skips_rows_without_deck():
    z = torch.zeros(1, 11, 17)
    gap = torch.zeros(1, 11, 17, dtype=torch.bool)
    z[0, 5, :8] = 0.0; z[0, 5, 8:10] = -0.65; gap[0, 5, 8:10] = True; z[0, 5, 10:] = 0.16
    z[0, 6, :] = -0.65; gap[0, 6, :] = True                                 # no deck in the row
    z[0, 7, :12] = 0.1; z[0, 7, 12:] = -0.65; gap[0, 7, 12:] = True         # deck only before
    lift = gap_lift_heights(z.flatten(1), gap.flatten(1), (~gap).flatten(1)).view(11, 17)
    assert torch.allclose(lift[5, 8:10], torch.zeros(2))                    # min(0.0, 0.16)
    assert torch.isnan(lift[6]).all()
    assert torch.allclose(lift[7, 12:], torch.full((5,), 0.1))
    assert torch.isnan(lift[5, :8]).all()                                    # non-gap cells: NaN


def test_lift_right_side_is_the_nearest_deck_after_the_gap_codex_case():
    z = torch.zeros(1, 11, 17); gap = torch.zeros(1, 11, 17, dtype=torch.bool); deck = torch.zeros_like(gap)
    z[0, 5, 3] = 0.0; deck[0, 5, 3] = True                                   # left lip col 3
    z[0, 5, 4:10] = -0.65; gap[0, 5, 4:10] = True
    z[0, 5, 10] = 0.16; deck[0, 5, 10] = True                                # right lip col 10
    z[0, 5, 11:] = 0.30; deck[0, 5, 11:] = True                              # farther, higher deck
    z[0, 4, 2:6] = -0.65; gap[0, 4, 2:6] = True; z[0, 4, 6] = -0.16; deck[0, 4, 6] = True   # right side only
    lift = gap_lift_heights(z.flatten(1), gap.flatten(1), deck.flatten(1)).view(11, 17)
    assert torch.allclose(lift[5, 4:10], torch.zeros(6))                     # min(0, 0.16), never the pit floor
    assert torch.allclose(lift[4, 2:6], torch.full((4,), -0.16))
    for width in range(1, 9):                                                # random widths / lips
        for zl, zr in ((0., .16), (.16, 0.), (-.16, .16), (0., 0.)):
            z = torch.full((1, 11, 17), -0.65); gap = torch.zeros(1, 11, 17, dtype=torch.bool); deck = torch.zeros_like(gap)
            z[0, 0, :4] = zl; deck[0, 0, :4] = True; gap[0, 0, 4:4 + width] = True
            z[0, 0, 4 + width:] = zr; deck[0, 0, 4 + width:] = True
            got = gap_lift_heights(z.flatten(1), gap.flatten(1), deck.flatten(1)).view(11, 17)[0, 4:4 + width]
            assert torch.allclose(got, torch.full((width,), min(zl, zr))), (width, zl, zr)


def batch():
    h = torch.randn(2, 187) * .5
    gap = torch.zeros(2, 11, 17); gap[:, :, 10:12] = 1
    clean = torch.cat((h, torch.ones(2, 187), gap.flatten(1), torch.ones(2, 187)), 1)
    vis = torch.zeros(2, 11, 17); vis[:, :, :8] = 1
    teacher = torch.cat((torch.where(vis.flatten(1) > 0, h, 0.), vis.flatten(1)), 1)
    return clean, teacher


def test_gap_observed_widens_only_the_gap_channel_on_real_gap_cells():
    clean, teacher = batch()
    y0, m0 = targets(clean, teacher)
    obs = torch.zeros(2, 11, 17, dtype=torch.bool); obs[:, :, 9:13] = True  # includes non-gap cols 9 and 12
    y1, m1 = targets(clean, teacher, gap_observed=obs.flatten(1))
    assert torch.equal(y0, y1)
    for ch in (0, 1, 3, 4, 5):
        assert torch.equal(m0[..., ch], m1[..., ch])
    added = (m1[..., 2] & ~m0[..., 2]).view(2, 11, 17)
    assert added[:, :, 10:12].all() and not added[:, :, 9].any() and not added[:, :, 12].any()
    pred = torch.randn(2, 187, 6)
    assert torch.equal(reconstruction_loss(pred, clean, teacher), reconstruction_loss(pred, clean, teacher, gap_observed=None))
    assert not torch.equal(reconstruction_loss(pred, clean, teacher), reconstruction_loss(pred, clean, teacher, gap_observed=obs.flatten(1)))


def look_down():
    # camera 1 m above the origin, optical axis -z (OpenCV-style camera frame mapped by the quaternion)
    pos = torch.tensor([[0., 0., 1.]])
    quat = torch.tensor([[0., 1., 0., 0.]])                                  # 180 deg about x: camera z -> world -z
    k = torch.tensor([[[40., 0., 40.], [0., 40., 22.5], [0., 0., 1.]]])
    return pos, quat, k


def test_freespace_passes_when_rays_go_beyond_the_point_and_fails_when_blocked():
    pos, quat, k = look_down()
    pts = torch.tensor([[[0.05, 0.05, 0.0]]])                                 # 1 m in front of the camera
    beyond = lambda o, d, m: torch.full((o.shape[0],), 1.65)                 # rays land 0.65 m deeper
    blocked = lambda o, d, m: torch.full((o.shape[0],), 0.5)                 # something 0.5 m in front
    nothing = lambda o, d, m: torch.full((o.shape[0],), math.inf)            # no return
    args = (pts, pos, quat, k, (45, 80), (0.15, 5.0))
    assert raycast_freespace_points(*args, beyond).all()
    assert not raycast_freespace_points(*args, blocked).any()
    assert raycast_freespace_points(*args, nothing).all()
    nan = lambda o, d, m: torch.full((o.shape[0],), math.nan)                # no evidence: fail closed
    assert not raycast_freespace_points(*args, nan).any()
    far = torch.tensor([[[0.05, 0.05, -6.0]]])                               # beyond the depth clip
    assert not raycast_freespace_points(far, pos, quat, k, (45, 80), (0.15, 5.0), beyond).any()
