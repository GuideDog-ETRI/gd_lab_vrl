"""Joint-position action with hard-limit saturation.

Implements the position-target saturation of Bjelonic et al. 2025 (arXiv
2509.06342, Eq. 9): a position target reaching past a hard joint limit is
pulled back toward that limit in proportion to how far the current joint
position has entered a soft band. Applying the same saturation at train time
keeps train and deploy consistent.
"""

from __future__ import annotations

import math

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

__all__ = [
    "JointPositionActionWithLimit",
    "JointPositionActionWithLimitCfg",
]


class JointPositionActionWithLimit(JointPositionAction):
    r"""``JointPositionAction`` + hard-limit saturation (Eq. 9).

    Before sending the position target to the actuator, a target past a hard
    joint limit is blended back to that limit as the current joint position
    enters the soft band (``hard`` shrunk inward by ``soft_margin``). Only the
    actuator target is modified — the raw action seen by the policy
    (``last_action``) is untouched.
    """

    cfg: JointPositionActionWithLimitCfg

    def __init__(self, cfg: JointPositionActionWithLimitCfg, env) -> None:
        super().__init__(cfg, env)
        self._soft_margin = float(cfg.soft_margin_deg) * math.pi / 180.0

    def _position_target(self) -> torch.Tensor:
        """Position target before limit saturation."""
        return self.processed_actions

    def apply_actions(self):
        target = self._position_target()
        q = self._asset.data.joint_pos[:, self._joint_ids]
        lim = self._asset.data.joint_pos_limits[:, self._joint_ids, :]
        hard_min = lim[..., 0]
        hard_max = lim[..., 1]
        soft_min = hard_min + self._soft_margin
        soft_max = hard_max - self._soft_margin

        up = (q > soft_max) & (target > hard_max)
        a_up = ((q - soft_max) / (hard_max - soft_max).clamp_min(1e-6)).clamp(0.0, 1.0)
        target = torch.where(up, target - a_up * (target - hard_max), target)

        lo = (q < soft_min) & (target < hard_min)
        a_lo = ((q - soft_min) / (hard_min - soft_min).clamp_max(-1e-6)).clamp(0.0, 1.0)
        target = torch.where(lo, target - a_lo * (target - hard_min), target)

        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class JointPositionActionWithLimitCfg(JointPositionActionCfg):

    class_type: type = JointPositionActionWithLimit

    soft_margin_deg: float = 5.0
    """Soft-band width inside each hard limit, degrees (matches the deployment
    controller's joint-limit soft margin). Linear PD-torque rolloff to zero at
    the hard limit starts ``soft_margin_deg`` before it."""
