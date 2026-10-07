"""Seeded camera capture/transport simulation, independent of Isaac and torch."""

from dataclasses import asdict, dataclass
import heapq
import math
import random
from typing import Any


@dataclass(frozen=True)
class CameraTransportConfig:
    interval_ms: tuple[float, float] = (70.0, 100.0)
    delay_ms: tuple[float, float] = (0.0, 50.0)
    drop_probability: float = 0.05

    def __post_init__(self):
        for bounds, positive in ((self.interval_ms, True), (self.delay_ms, False)):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds):
                raise ValueError("Camera timing ranges must contain two finite numbers")
            if bounds[0] < 0 or bounds[1] < bounds[0] or (positive and bounds[0] == 0):
                raise ValueError("Camera timing ranges must be ordered and nonnegative; interval must be positive")
        if not 0 <= self.drop_probability < 1:
            raise ValueError("Camera drop probability must be in [0, 1)")

    def manifest(self, policy_dt):
        return {**asdict(self), "policy_dt": policy_dt,
                "sampling": "uniform_integer_policy_steps", "scope": "shared_batch_four_camera_set",
                "target_time": "capture", "out_of_order": "discard_older_than_last_delivered"}


@dataclass(frozen=True)
class CameraPacket:
    capture_step: int
    delivery_step: int
    payload: Any


def delivery_mask(capture_step, captured_episodes, current_episodes, last_delivered):
    """Exclude pre-reset and reordered packets, independently for each robot."""
    return (captured_episodes == current_episodes) & (capture_step > last_delivered)


class CameraTransport:
    def __init__(self, config: CameraTransportConfig, policy_dt: float, seed: int):
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("policy_dt must be finite and positive")
        self.config = config
        self.rng = random.Random(seed)
        self.interval = self._steps(config.interval_ms, policy_dt)
        self.delay = self._steps(config.delay_ms, policy_dt)
        if self.interval[0] < 1:
            raise ValueError("Camera interval must be at least one policy step")
        self.pending = []
        self.captured = self.dropped = self.delivered = 0

    @staticmethod
    def _steps(bounds, dt):
        lo = math.ceil(bounds[0] / (dt * 1000) - 1e-9)
        hi = math.floor(bounds[1] / (dt * 1000) + 1e-9)
        if lo > hi:
            raise ValueError("Camera timing range contains no integer policy step")
        return lo, hi

    def next_capture(self, step):
        return step + self.rng.randint(*self.interval)

    def capture(self, step, payload):
        self.captured += 1
        if self.rng.random() < self.config.drop_probability:
            self.dropped += 1
            return
        delivery = step + self.rng.randint(*self.delay)
        packet = CameraPacket(step, delivery, payload)
        heapq.heappush(self.pending, (delivery, self.captured, packet))

    def receive(self, step):
        packets = []
        while self.pending and self.pending[0][0] <= step:
            packets.append(heapq.heappop(self.pending)[2])
            self.delivered += 1
        return packets
