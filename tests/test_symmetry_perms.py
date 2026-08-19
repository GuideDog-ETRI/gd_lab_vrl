"""Mirror permutations must be involutions (applying them twice is identity)."""

from __future__ import annotations

import torch

from gd_lab.core.types import ObsSpec, ObsTermSpec
from gd_lab.mdp.symmetry import (
    _build_joint_perm_sign,
    _build_scan_perm,
    build_group_perm_sign,
    resolve_mirror_tag,
)

JOINT_NAMES = [
    "FL_HIP", "FR_HIP", "HL_HIP", "HR_HIP",
    "FL_THIGH", "FR_THIGH", "HL_THIGH", "HR_THIGH",
    "FL_KNEE", "FR_KNEE", "HL_KNEE", "HR_KNEE",
]


def _apply(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return x[..., perm] * sign


def test_joint_perm_is_involution():
    perm, sign = _build_joint_perm_sign(JOINT_NAMES)
    perm_t = torch.tensor(perm)
    sign_t = torch.tensor(sign)
    x = torch.randn(4, 12)
    assert torch.allclose(_apply(_apply(x, perm_t, sign_t), perm_t, sign_t), x)


def test_scan_perm_is_involution():
    perm = torch.tensor(_build_scan_perm(17, 11))
    x = torch.randn(3, 17 * 11)
    assert torch.allclose(x[..., perm][..., perm], x)


def test_every_tag_is_involution():
    joint_ps = _build_joint_perm_sign(JOINT_NAMES)
    scan_perm = _build_scan_perm(17, 11)
    dims = {
        "ang_vel": 3, "lin_vec": 3, "commands": 3, "joint": 12, "scan": 187,
        "feet4": 4, "feet16": 16, "feet_vec3": 12, "gain24": 24, "identity": 7,
    }
    for tag, dim in dims.items():
        perm, sign = resolve_mirror_tag(tag, dim, joint_ps, scan_perm)
        perm_t = torch.tensor(perm if perm is not None else list(range(dim)))
        sign_t = torch.tensor(sign if sign is not None else [1.0] * dim)
        x = torch.randn(5, dim)
        assert torch.allclose(_apply(_apply(x, perm_t, sign_t), perm_t, sign_t), x), tag


def test_group_perm_covers_history_layout():
    spec = ObsSpec(
        terms=(ObsTermSpec("base_ang_vel", 3, "ang_vel"), ObsTermSpec("joint_pos", 12, "joint")),
        history=5,
    )
    joint_ps = _build_joint_perm_sign(JOINT_NAMES)
    perm, sign = build_group_perm_sign(spec, joint_ps, _build_scan_perm(17, 11), "cpu")
    assert perm.shape[0] == spec.total == 75
    x = torch.randn(2, 75)
    assert torch.allclose(_apply(_apply(x, perm, sign), perm, sign), x)
