from __future__ import annotations

import pytest

from gd_lab.core.types import ObsSpec, ObsTermSpec
from gd_lab.methods.dreamwaq.spec import (
    DREAMWAQ_SPEC,
    ONE_STEP_OBS_DIM,
    POLICY_OBS_DIM,
    VELOCITY_TARGET_SLICE,
)


def test_policy_spec_dims():
    assert ONE_STEP_OBS_DIM == 45
    assert POLICY_OBS_DIM == 225
    assert DREAMWAQ_SPEC.policy.history == 5


def test_velocity_target_slice():
    assert VELOCITY_TARGET_SLICE == (45, 48)


def test_critic_spec_resolution():
    with pytest.raises(ValueError):
        _ = DREAMWAQ_SPEC.critic.total
    critic = DREAMWAQ_SPEC.critic.resolve(height_scan=187)
    assert critic.total == 45 + 3 + 16 + 4 + 12 + 2 + 1 + 24 + 187 == 294
    assert critic.slice("base_lin_vel") == slice(45, 48)


def test_latest_slices_term_major_layout():
    spec = ObsSpec(
        terms=(ObsTermSpec("a", 2, "identity"), ObsTermSpec("b", 3, "identity")),
        history=4,
    )
    # Layout: [a(t-3..t) 2x4 | b(t-3..t) 3x4]; newest step of each term is the last block.
    assert spec.latest_slices() == [slice(6, 8), slice(17, 20)]
    assert spec.latest_indices() == [6, 7, 17, 18, 19]
    assert spec.total == 20


def test_unknown_mirror_tag_rejected():
    with pytest.raises(ValueError):
        ObsTermSpec("x", 3, "not_a_tag")


def test_duplicate_term_names_rejected():
    with pytest.raises(ValueError):
        ObsSpec(terms=(ObsTermSpec("a", 2, "identity"), ObsTermSpec("a", 3, "identity")))
