import pytest
import torch

from gd_lab.gast.student_utils import finite_batch_rows


def test_finite_batch_rows_accepts_rank_one_and_rank_two():
    assert finite_batch_rows(torch.tensor([1.0, 2.0])).tolist() == [True, True]
    assert finite_batch_rows(torch.tensor([[1.0, 2.0], [3.0, float("nan")]])).tolist() == [True, False]


def test_finite_batch_rows_rejects_scalar():
    with pytest.raises(ValueError, match="batch dimension"):
        finite_batch_rows(torch.tensor(1.0))
