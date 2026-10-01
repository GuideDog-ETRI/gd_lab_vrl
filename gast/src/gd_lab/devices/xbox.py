"""Xbox gamepad reader backed by the `inputs` package (a training-machine-only extra).

Importing this module requires `inputs` to be installed, so callers that must
import without it must not import this module eagerly.
"""

from __future__ import annotations

import math
import threading
import time

import inputs
from inputs import get_gamepad


class XboxController:
    def __init__(self):
        self.MAX_TRIG_VAL = math.pow(2, 8)
        self.MAX_JOY_VAL = math.pow(2, 15)
        self.deadzone = 1200
        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0
        self.connected = False
        self._error_logged = False

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            dead_val = 0
        else:
            if value > 0:
                dead_val = value - self.deadzone
            else:
                dead_val = value + self.deadzone
        return dead_val

    def zero_axes(self):
        """Reset every analog axis to neutral (used when the device goes away)."""
        self.LeftJoystickX = 0
        self.LeftJoystickY = 0
        self.RightJoystickX = 0
        self.RightJoystickY = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0

    def _monitor_controller(self):
        while True:
            try:
                events = get_gamepad()
            except Exception as exc:
                # Unplugged or read error: drop to neutral and keep retrying, so a
                # stale stick value never latches into the velocity command forever.
                if not self._error_logged:
                    print(f"[XboxController] gamepad unavailable ({exc}); command held at zero.")
                    self._error_logged = True
                self.connected = False
                self.zero_axes()
                try:
                    inputs.devices = inputs.DeviceManager()
                except Exception:
                    pass
                time.sleep(0.5)
                continue

            if not self.connected:
                print("[XboxController] gamepad connected.")
                self.connected = True
                self._error_logged = False

            for event in events:
                if event.code == "ABS_Y":
                    self.LeftJoystickY = -self.apply_deadzone(event.state) / self.MAX_JOY_VAL
                elif event.code == "ABS_X":
                    self.LeftJoystickX = self.apply_deadzone(event.state) / self.MAX_JOY_VAL
                elif event.code == "ABS_RY":
                    self.RightJoystickY = -self.apply_deadzone(event.state) / self.MAX_JOY_VAL
                elif event.code == "ABS_RX":
                    self.RightJoystickX = -self.apply_deadzone(event.state) / self.MAX_JOY_VAL
                elif event.code == "ABS_Z":
                    self.LeftTrigger = event.state / self.MAX_TRIG_VAL
                elif event.code == "ABS_RZ":
                    self.RightTrigger = event.state / self.MAX_TRIG_VAL
                elif event.code == "BTN_TL":
                    self.LeftBumper = event.state
                elif event.code == "BTN_TR":
                    self.RightBumper = event.state
                elif event.code == "BTN_SOUTH":
                    self.A = event.state
                elif event.code == "BTN_WEST":
                    self.Y = event.state
                elif event.code == "BTN_NORTH":
                    self.X = event.state
                elif event.code == "BTN_EAST":
                    self.B = event.state
                elif event.code == "BTN_THUMBL":
                    self.LeftThumb = event.state
                elif event.code == "BTN_THUMBR":
                    self.RightThumb = event.state
                elif event.code == "BTN_SELECT":
                    self.Back = event.state
                elif event.code == "BTN_START":
                    self.Start = event.state
                elif event.code == "ABS_HAT0X":
                    self.LeftDPad = int(event.state == -1)
                    self.RightDPad = int(event.state == 1)
                elif event.code == "ABS_HAT0Y":
                    self.UpDPad = int(event.state == -1)
                    self.DownDPad = int(event.state == 1)
