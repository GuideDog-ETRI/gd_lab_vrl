"""NAV1/NAV1-ACK local IPC: versioned canonical JSON payload + binary length/CRC.
No transport side effects. Caller supplies CLOCK_MONOTONIC ms. Bad packets raise;
the receiver retains the error and the adapter emits hold (never silent defaults).
"""
from dataclasses import dataclass, asdict
import json
import math
import struct
import zlib
from . import schema

HEADER = struct.Struct('<4sHII')
VERSION = 1

@dataclass(frozen=True)
class Conditions:
    y_off: float
    margin: float
    turn_cap: float

@dataclass(frozen=True)
class RouteHint:
    segment_id: int
    segment_type: str
    route_axis_yaw: float
    dist_to_next_m: float
    next_segment_type: str

@dataclass(frozen=True)
class Release:
    landing_id: int
    seq: int
    source: str
    timestamp_ms: float

@dataclass(frozen=True)
class NavCommand:
    seq: int
    timestamp_ms: float
    requested_mode: str
    requested_go: bool
    v_cap: float
    turn_rate: float
    c: Conditions
    manoeuvre_id: int
    route_epoch: int
    heartbeat: int
    route_hint: RouteHint | None = None
    landing_id: int | None = None
    landing_release: Release | None = None

@dataclass(frozen=True)
class NavAck:
    request_seq: int
    timestamp_ms: float
    active_mode: str
    handshake: str
    effective_go: bool
    phase: str
    dwell_progress: float
    hold_reason: str
    requested_c: Conditions
    applied_c: Conditions
    manoeuvre_id: int
    route_epoch: int
    release_seq: int | None = None
    release_accepted: bool | None = None
    release_reason: str = ''
    phase_event_seq: int = 0

def _finite(value):
    if isinstance(value, dict): return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(_finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)

def _id(value):
    return type(value) is int and value >= 0

def validate_command(cmd, cfg, now_ms):
    numeric = (cmd.timestamp_ms, cmd.v_cap, cmd.turn_rate, cmd.c.y_off, cmd.c.margin, cmd.c.turn_cap)
    if any(type(v) not in (int, float) for v in numeric): raise ValueError('NUMERIC_TYPE')
    if cmd.route_hint and any(type(v) not in (int, float) for v in
                              (cmd.route_hint.route_axis_yaw, cmd.route_hint.dist_to_next_m)):
        raise ValueError('HINT_NUMERIC_TYPE')
    if cmd.landing_release and type(cmd.landing_release.timestamp_ms) not in (int, float):
        raise ValueError('RELEASE_TIME_TYPE')
    if not _finite(asdict(cmd)): raise ValueError('NONFINITE')
    if cmd.requested_mode not in schema.MODES or type(cmd.requested_go) is not bool:
        raise ValueError('MODE_OR_GO')
    if not all(_id(v) for v in (cmd.seq, cmd.manoeuvre_id, cmd.route_epoch, cmd.heartbeat)):
        raise ValueError('ID')
    age = now_ms - cmd.timestamp_ms
    if age > cfg['timeout_ms'] or age < -cfg['future_tolerance_ms']:
        raise ValueError('TIMEOUT_OR_FUTURE')
    for key in schema.CONDITION_KEYS:
        lo, hi = cfg[key + '_range']
        if not lo <= getattr(cmd.c, key) <= hi: raise ValueError('RANGE_' + key)
    if not cfg['v_cap_range'][0] <= cmd.v_cap <= cfg['v_cap_range'][1]: raise ValueError('RANGE_v_cap')
    if abs(cmd.turn_rate) > cfg['turn_cap_range'][1]: raise ValueError('RANGE_turn_rate')
    if cmd.landing_id is not None and not _id(cmd.landing_id): raise ValueError('LANDING_ID')
    h = cmd.route_hint
    if h and (not _id(h.segment_id) or h.segment_type not in schema.SEGMENT_TYPES or
              h.next_segment_type not in schema.SEGMENT_TYPES or h.dist_to_next_m < 0):
        raise ValueError('ROUTE_HINT')
    r = cmd.landing_release
    if r and (not _id(r.seq) or not _id(r.landing_id) or r.source not in schema.RELEASE_SOURCES):
        raise ValueError('RELEASE')
    return cmd

def encode(message):
    if not isinstance(message, (NavCommand, NavAck)) or not _finite(asdict(message)):
        raise ValueError('message type/nonfinite')
    payload = json.dumps(asdict(message), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    magic = b'NAV1' if isinstance(message, NavCommand) else b'NAK1'
    return HEADER.pack(magic, VERSION, len(payload), zlib.crc32(payload)) + payload

def decode(data, *, cfg, now_ms):
    if len(data) < HEADER.size: raise ValueError('TRUNCATED')
    magic, version, length, crc = HEADER.unpack_from(data)
    payload = data[HEADER.size:]
    if magic not in (b'NAV1', b'NAK1') or version != VERSION or length != len(payload) or zlib.crc32(payload) != crc:
        raise ValueError('HEADER_OR_CRC')
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError('DUPLICATE_KEY')
                result[key] = value
            return result
        d = json.loads(payload, object_pairs_hook=unique)
        if not _finite(d): raise ValueError('NONFINITE')
        if magic == b'NAV1':
            d['c'] = Conditions(**d['c'])
            for key, cls in [('route_hint', RouteHint), ('landing_release', Release)]:
                if d.get(key) is not None: d[key] = cls(**d[key])
            return validate_command(NavCommand(**d), cfg, now_ms)
        for key in ('requested_c', 'applied_c'): d[key] = Conditions(**d[key])
        ack = NavAck(**d)
        if ack.active_mode not in schema.MODES or ack.phase not in schema.PHASES or not 0 <= ack.dwell_progress <= 1:
            raise ValueError('ACK_ENUM_RANGE')
        if (type(ack.effective_go) is not bool or
            not all(_id(v) for v in (ack.request_seq, ack.manoeuvre_id, ack.route_epoch, ack.phase_event_seq)) or
            ack.handshake not in ('STOPPING', 'STABLE', 'ACTIVE') or
            ack.release_seq is not None and not _id(ack.release_seq) or
            ack.release_accepted is not None and type(ack.release_accepted) is not bool):
            raise ValueError('ACK_TYPE')
        for conditions in (ack.requested_c, ack.applied_c):
            for key in schema.CONDITION_KEYS:
                lo, hi = cfg[key + '_range']
                if not lo <= getattr(conditions, key) <= hi: raise ValueError('ACK_CONDITION_RANGE')
        if now_ms - ack.timestamp_ms > cfg['timeout_ms'] or now_ms < ack.timestamp_ms - cfg['future_tolerance_ms']:
            raise ValueError('ACK_STALE')
        return ack
    except (TypeError, KeyError, json.JSONDecodeError) as e:
        raise ValueError('SCHEMA') from e

class CommandAdapter:
    def __init__(self, cfg, *, initial_yaw):
        self.cfg = cfg
        self.active_mode = 'LEGACY'
        self.pending_mode = None
        self.handshake = 'ACTIVE'
        self.current = None
        self.error = ''
        self.last_key = None
        self.axis_yaw = initial_yaw  # frozen if no hint; never follows the robot each tick.
        self.c = Conditions(**cfg['defaults'])

    def accept(self, cmd, *, now_ms, stationary):
        try:
            validate_command(cmd, self.cfg, now_ms)
            key = (cmd.route_epoch, cmd.seq)
            if self.last_key is not None and key < self.last_key: raise ValueError('REORDER')
            if self.last_key == key:
                if cmd != self.current: raise ValueError('SEQ_REUSE')
                return False
            if cmd.c.margin < self.c.margin and not stationary: raise ValueError('MARGIN_DECREASE_MOVING')
        except (ValueError, TypeError) as e:
            self.error = str(e); return False
        self.current, self.c, self.last_key, self.error = cmd, cmd.c, key, ''
        if cmd.route_hint: self.axis_yaw = cmd.route_hint.route_axis_yaw
        if cmd.requested_mode != self.active_mode:
            if self.pending_mode != cmd.requested_mode:
                self.pending_mode, self.handshake = cmd.requested_mode, 'STOPPING'
        elif self.pending_mode is not None:
            self.pending_mode, self.handshake = None, 'ACTIVE'
        return True

    def tick(self, *, now_ms, stable, hold=False, fsm_go=True, pre_brake_v=None, legacy=(0., 0., 0.)):
        c = self.current
        reason = self.error
        if c:
            try: validate_command(c, self.cfg, now_ms)
            except (ValueError, TypeError) as e: reason = str(e)
        else: reason = reason or 'NO_COMMAND'
        if reason: return {'velocity': (0., 0., 0.), 'effective_go': False, 'hold_reason': reason, 'mode': self.active_mode, 'v_cap': c.v_cap if c else 0.}
        if self.pending_mode and not hold:
            if self.handshake == 'STOPPING' and stable: self.handshake = 'STABLE'
            elif self.handshake == 'STABLE':
                if stable: self.active_mode, self.pending_mode, self.handshake = self.pending_mode, None, 'ACTIVE'
                else: self.handshake = 'STOPPING'
        go = c.requested_go and fsm_go and not hold and self.pending_mode is None
        velocity = (0., 0., 0.)
        if go:
            if self.active_mode == 'LEGACY': velocity = tuple(legacy)
            elif self.active_mode == 'NAV': velocity = (c.v_cap, 0., 0.)
            else:
                verified = self.cfg['omega_max_verified']
                if verified is None: return {'velocity': velocity, 'effective_go': False, 'hold_reason': 'TURN_UNVERIFIED', 'mode': self.active_mode, 'v_cap': c.v_cap if c else 0.}
                velocity = (0., 0., max(-min(c.c.turn_cap, verified), min(c.turn_rate, 0.)))
        if pre_brake_v is not None and c.requested_go and not hold and self.pending_mode is None and self.active_mode == 'NAV':
            if not math.isfinite(pre_brake_v): raise ValueError('invalid braking profile')
            velocity = (max(0., min(c.v_cap, pre_brake_v)), 0., 0.)
        return {'velocity': velocity, 'effective_go': go, 'hold_reason': 'HOLD' if hold else '', 'mode': self.active_mode, 'v_cap': c.v_cap if c else 0.}

def policy_features(command, applied_c, phase, dwell_progress):
    """Fixed 16-wide actor input; use effective command, never requested_go."""
    return [float(command['mode'] == m) for m in schema.MODES] + [float(command['effective_go']),
           command['v_cap'], command['velocity'][2]] + [float(phase == p) for p in schema.PHASES] + [dwell_progress,
           applied_c.y_off, applied_c.margin, applied_c.turn_cap]
