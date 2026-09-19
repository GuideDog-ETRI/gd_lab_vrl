"""Arm selection must fail before launching the simulator on invalid input."""

import pytest

from gd_lab.core.experiments import training_arm_overrides


def test_no_arm_leaves_the_original_recipe_selected():
    assert training_arm_overrides(None) == []


@pytest.mark.parametrize("arm", ["", "0", "5", "01", "arm_1", "../arm_1", "1,2"])
def test_invalid_training_arm_is_rejected(arm):
    with pytest.raises(ValueError, match="TRAIN_ARM must be"):
        training_arm_overrides(arm)
