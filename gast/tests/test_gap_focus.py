"""Near-gap sample weighting for the GAST gap-focused student fine-tune (CPU only)."""
import torch
from torch.nn import functional as F

from gd_lab.gast.geometry import near_gap_rows, reconstruction_loss, weighted_mean


def _clean(batch, gap_rows=()):
    clean = torch.zeros(batch, 4, 11, 17)
    clean[:, 0] = torch.rand(batch, 11, 17)  # height
    clean[:, 1] = 1  # scan valid
    clean[:, 3] = 1  # gap known
    for row in gap_rows:
        clean[row, 2, 5, 12] = 1  # one gap cell ahead of the body
    return clean.flatten(1)


def test_near_gap_rows_reads_only_known_gap_cells():
    clean = _clean(3, gap_rows=(1,)).reshape(3, 4, 11, 17)
    clean[2, 2, 0, 0] = 1
    clean[2, 3, 0, 0] = 0  # gap label on an unknown cell does not count
    assert near_gap_rows(clean.flatten(1)).tolist() == [False, True, False]


def test_unit_weights_reproduce_the_original_losses():
    torch.manual_seed(0)
    values = torch.rand(7)
    assert torch.allclose(weighted_mean(values, torch.ones(7)), values.mean())
    a, b = torch.rand(5, 12), torch.rand(5, 12)
    assert torch.allclose(weighted_mean((a - b).pow(2).mean(-1), torch.ones(5)), F.mse_loss(a, b))
    pred, clean = torch.randn(4, 187, 6), _clean(4, gap_rows=(0, 2))
    terrain = torch.cat((torch.rand(4, 187) * 5, (torch.rand(4, 187) > .3).float()), -1)
    base = reconstruction_loss(pred, clean, terrain, return_components=True)
    ones = reconstruction_loss(pred, clean, terrain, return_components=True, row_weight=torch.ones(4))
    for x, y in zip(base, ones):
        assert torch.allclose(x, y)


def test_gap_weight_moves_the_loss_toward_near_gap_rows():
    torch.manual_seed(1)
    pred, clean = torch.randn(2, 187, 6), _clean(2, gap_rows=(0,))
    terrain = torch.cat((torch.rand(2, 187) * 5, torch.ones(2, 187)), -1)
    first = reconstruction_loss(pred[:1], clean[:1], terrain[:1])
    weight = 1 + 4 * near_gap_rows(clean).float()
    weighted = reconstruction_loss(pred, clean, terrain, row_weight=weight)
    plain = reconstruction_loss(pred, clean, terrain)
    assert abs(weighted - first) < abs(plain - first)


def test_gap_top5_skips_windows_with_too_few_near_gap_rows(tmp_path):
    from gd_lab.gast.student_top5 import StudentTop5
    board = StudentTop5(tmp_path, start_iteration=0, keep=2, smoothing_windows=1, min_visible_fraction=None,
                        min_visible_sample_fraction=None, min_hazard_supervised_fraction=0,
                        score_description="near_gap_action_mse", min_score_rows=64)
    metrics = {"latent_mse": .1, "hazard_mse": .1, "extra_loss": .1, "visible_fraction": 1,
               "hazard_supervised_fraction": 1, "updates": 1}
    saved = []
    save = lambda path, record: (saved.append(record), path.write_bytes(b"x"))  # noqa: E731
    assert board.consider(1, .01, {**metrics, "score_rows": 10}, save)["reason"] == "too_few_score_rows"
    assert board.consider(2, .01, {**metrics, "score_rows": 100}, save)["saved"]
    assert board.criteria["score"] == "near_gap_action_mse" and saved[0]["iteration"] == 2
