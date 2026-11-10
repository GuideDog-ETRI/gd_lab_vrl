"""DC motor + Coulomb joint friction (S2b): MuJoCo (frictionloss 0.5 N m) and the real drives have it, Isaac does not.

S2 v1 policies held the handstand in Isaac but overshot past vertical and fell in MuJoCo; training against a random
per-env friction in [0, friction_max] N m closes that part of the gap.
"""
from __future__ import annotations

import torch
from isaaclab.utils import configclass

from gd_lab.robots.actuators import DelayedDCMotor, DelayedDCMotorCfg


class FrictionDCMotor(DelayedDCMotor):
    cfg: "FrictionDCMotorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._friction = torch.rand(self._num_envs, self.num_joints, device=self._device) * float(cfg.friction_max)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action = super().compute(control_action, joint_pos, joint_vel)
        friction = self._friction * torch.tanh(joint_vel / 0.05)
        self.applied_effort = self.applied_effort - friction
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class FrictionDCMotorCfg(DelayedDCMotorCfg):
    class_type: type = FrictionDCMotor
    friction_max: float = 1.0
