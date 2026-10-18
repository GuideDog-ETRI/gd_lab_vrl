"""Sensor-derived landing events enter through LandingObservation, never GT phase labels."""
from dataclasses import dataclass
from collections import deque
from .commands import Release

@dataclass(frozen=True)
class LandingObservation:
    hint_matches: bool
    landing_detected: bool
    brake_point: bool
    entered: bool
    four_contacts: bool
    speed_mps: float
    yaw_rate_radps: float
    slipping: bool
    descent_visible: bool
    descending: bool = False
    exited_and_next_segment: bool = False

class LandingFSM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = 'APPROACH'
        self.dwell_timer = self.announce_timer = 0.
        self.event_seq = 0
        self.auto_seq = 0
        self.context = None
        self.queue = deque()
        self.consumed = set()
        self.acks = []

    def reset(self, context=None):
        self.phase = 'APPROACH'
        self.dwell_timer = self.announce_timer = 0.
        self.context = context
        self.queue.clear()
        # Consumed identities include epoch: never resurrect a duplicate within an epoch.
        self.event_seq += 1

    def _phase(self, phase):
        if phase != self.phase:
            self.phase = phase; self.event_seq += 1

    def enqueue(self, release):
        self.queue.append(release)

    def tick(self, *, now_ms, dt, command, observation, hold=False, sensor_timeout=False, odom_epoch=0):
        if dt < 0: raise ValueError('negative dt')
        h = command.route_hint
        context = (command.route_epoch, command.requested_mode, command.landing_id, odom_epoch)
        if context != self.context or sensor_timeout:
            self.reset(context)
        o = observation
        stable = o.four_contacts and not o.slipping and abs(o.speed_mps) < self.cfg['stable_v_mps'] and abs(o.yaw_rate_radps) < self.cfg['stable_w_radps']
        valid_hint = h is not None and command.landing_id is not None and o.hint_matches
        reason = ''
        if self.phase == 'APPROACH' and o.landing_detected:
            if valid_hint: self._phase('PRE_BRAKE')
            else: reason = 'HINT_MISSING_OR_MISMATCH'
        if self.phase == 'PRE_BRAKE' and o.brake_point and o.entered: self._phase('LANDING_ENTERED')
        if self.phase == 'LANDING_ENTERED' and stable: self._phase('DWELL')
        if self.phase != 'APPROACH' and not valid_hint:
            reason = 'HINT_MISSING_OR_MISMATCH'
        if self.phase in ('LANDING_ENTERED', 'DWELL') and not o.descent_visible:
            reason = 'NO_DESCENT_VIEW'
        if sensor_timeout: reason = 'SENSOR_TIMEOUT'
        unsafe = hold or bool(reason)
        if unsafe:
            self.dwell_timer = self.announce_timer = 0.
        elif not stable:
            self.dwell_timer = 0.
        if self.phase == 'DWELL' and not unsafe and stable and command.requested_go:
            self.announce_timer += dt
            if stable: self.dwell_timer += dt
        threshold = self.cfg['announce_s'] + self.cfg['stable_after_announce_s']
        ready = (self.phase == 'DWELL' and self.dwell_timer >= threshold
                 and self.announce_timer >= self.cfg['announce_s'] and stable and command.requested_go
                 and valid_hint and o.descent_visible and not unsafe)
        if command.landing_release is not None: self.enqueue(command.landing_release)
        if ready and not any((command.route_epoch, r.landing_id, r.source, r.seq) not in self.consumed for r in self.queue):
            self.auto_seq += 1
            self.enqueue(Release(command.landing_id, self.auto_seq, 'AUTO', now_ms))
        self.acks = []
        while self.queue:
            release = self.queue.popleft()
            key = (command.route_epoch, release.landing_id, release.source, release.seq)
            rejection = ''
            if key in self.consumed: rejection = 'DUPLICATE'
            elif release.source != 'AUTO': rejection = 'SOURCE_NOT_ENABLED'
            elif release.landing_id != command.landing_id: rejection = 'LANDING_ID'
            elif not 0 <= now_ms - release.timestamp_ms <= self.cfg['release_timeout_ms']: rejection = 'STALE_RELEASE'
            elif not ready: rejection = reason or 'NOT_READY'
            elif now_ms - command.timestamp_ms > self.cfg['release_timeout_ms']: rejection = 'STALE_COMMAND'
            self.consumed.add(key)
            accepted = not rejection
            self.acks.append({'seq': release.seq, 'source': release.source, 'accepted': accepted, 'reason': rejection})
            if accepted: self._phase('RELEASED'); ready = False
        if self.phase == 'RELEASED' and o.descending: self._phase('DESCEND')
        if self.phase == 'DESCEND' and o.exited_and_next_segment:
            self.reset(context)
        blocked = self.phase in ('LANDING_ENTERED', 'DWELL') or unsafe or bool(reason)
        return {'phase': self.phase, 'effective_go': command.requested_go and not blocked,
                'hold_reason': reason or ('HOLD' if hold else ''), 'acks': self.acks,
                'event_seq': self.event_seq, 'dwell_progress': min(1., self.dwell_timer / threshold) if threshold else 0.,
                'stall_masked': blocked or self.phase == 'PRE_BRAKE' or not command.requested_go}
