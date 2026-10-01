"""Action-smoothness penalty terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from gd_lab.managers.action_history import get_prev_prev_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_acceleration_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action jerk via the second-order action difference (L2 squared).

    ``action - 2*prev_action + prev_prev_action``, zeroed for the first two
    steps after a reset (before both history slots are populated).
    """
    prev_prev_action = get_prev_prev_action(env)
    diff = torch.square(
        env.action_manager.action - 2 * env.action_manager.prev_action + prev_prev_action
    )
    diff = diff * (env.action_manager.prev_action != 0)
    diff = diff * (prev_prev_action != 0)
    return torch.sum(diff, dim=1)
