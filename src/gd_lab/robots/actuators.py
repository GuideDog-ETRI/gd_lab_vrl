"""Custom actuator models combining IsaacLab building blocks."""

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.actuators.actuator_pd import DCMotor, DelayedPDActuator
from isaaclab.utils import configclass


class DelayedDCMotor(DelayedPDActuator, DCMotor):
    """DC motor actuator with delayed command application.

    MRO chains DelayedPDActuator.compute (delays the setpoints) into
    DCMotor.compute (torque-speed curve clipping), both on top of IdealPDActuator.
    """

    cfg: DelayedDCMotorCfg


@configclass
class DelayedDCMotorCfg(DelayedPDActuatorCfg, DCMotorCfg):

    class_type: type = DelayedDCMotor
