"""Patches an env's action manager at runtime to track the previous-previous
action, for action-acceleration reward terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_prev_prev_action_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Create the prev-prev-action buffer once and expose a public alias."""
    action_manager = env.action_manager
    if not hasattr(action_manager, "_prev_prev_action"):
        action_manager._prev_prev_action = torch.zeros_like(action_manager.action)
    action_manager.prev_prev_action = action_manager._prev_prev_action
    return action_manager._prev_prev_action


def _reset_prev_prev_action_rows(prev_prev_action: torch.Tensor, env_ids) -> None:
    """Reset all or a subset of prev-prev-action rows."""
    if env_ids is None:
        prev_prev_action.zero_()
        return

    if isinstance(env_ids, slice):
        prev_prev_action[env_ids] = 0
        return

    if isinstance(env_ids, Sequence) and len(env_ids) == 0:
        return

    prev_prev_action[env_ids] = 0


def ensure_prev_prev_action_tracking(env: ManagerBasedRLEnv) -> None:
    """Ensure that prev_prev_action tracking is set up.

    Patches the action manager's ``process_action`` (to also update
    ``prev_prev_action``) and ``reset`` (to also reset it). Idempotent: safe
    to call every time a caller needs the buffer.

    Args:
        env: The environment instance.
    """
    action_manager = env.action_manager

    if hasattr(action_manager, "_prev_prev_action_patched"):
        return

    prev_prev_action = _ensure_prev_prev_action_buffer(env)

    original_process_action = action_manager.process_action

    def patched_process_action(action: torch.Tensor):
        # Update prev_prev_action before updating prev_action.
        prev_prev_action[:] = action_manager.prev_action
        original_process_action(action)

    action_manager.process_action = patched_process_action

    if hasattr(action_manager, "reset"):
        original_reset = action_manager.reset

        def patched_reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)

            env_ids = kwargs.get("env_ids")
            if env_ids is None and len(args) > 0:
                env_ids = args[0]

            _reset_prev_prev_action_rows(prev_prev_action, env_ids)
            action_manager.prev_prev_action = prev_prev_action
            return result

        action_manager.reset = patched_reset

    action_manager.prev_prev_action = prev_prev_action
    action_manager._prev_prev_action_patched = True


def get_prev_prev_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the tracked previous-previous action buffer for ``env``.

    Installs the tracking patch on first use (a no-op if already installed),
    so callers do not need to call :func:`ensure_prev_prev_action_tracking`
    themselves.
    """
    ensure_prev_prev_action_tracking(env)
    return env.action_manager.prev_prev_action
