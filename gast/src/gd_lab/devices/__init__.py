"""Gamepad input devices.

Importing this package pulls in :mod:`gd_lab.devices.xbox`, which requires the
`inputs` pip package (a training-machine-only extra). Callers that must import
without `inputs` installed should not import this package at module load time
— import it lazily, e.g. inside a command term's ``__init__``.
"""

from __future__ import annotations

from .se2_controller import Se2Controller, Se2ControllerCfg
from .xbox import XboxController

__all__ = ["Se2Controller", "Se2ControllerCfg", "XboxController"]
