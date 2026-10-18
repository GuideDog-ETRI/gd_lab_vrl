"""OBS1 v1 local-IPC wire format (C2.5), fixed little-endian header + CRC32.

Header: <4s HH I QQQ 4d = magic, version, header bytes, total bytes,
CLOCK_MONOTONIC sample ms, seq, reset epoch, sample x/y/z/yaw. Metadata:
<4H17f = wide nx/ny, near nx/ny, wide res/x0/y0, near res/x0/y0,
clearance max, age step, six band bounds, support z0/step, TTL.
Payload (x-fastest/band-major): wide 2-bit states; wide 4-bit ages;
wide uint8 support; near two uint8 clearance planes; near 1-bit unknown;
near 4-bit age. Each array separately zero-padded to a byte boundary.
CRC32 of all preceding bytes is a final little-endian uint32. Strict length,
metadata, codes and zero padding are checked before constructing observations.
"""
import math
import struct
import zlib
from dataclasses import dataclass
import torch
from . import schema as S
from .map_config import load_config
from .map_types import MapFrame, NearFrame

MAGIC = b'OBS1'
VERSION = 1
BASE = struct.Struct('<4sHHIQQQ4d')
META = struct.Struct('<4H17f')
CRC = struct.Struct('<I')
HEADER_BYTES = BASE.size+META.size
WIDE_CELLS = S.WIDE_NX*S.WIDE_NY
BAND_CELLS = WIDE_CELLS*len(S.BANDS)
NEAR_CELLS = S.NEAR_NX*S.NEAR_NY
PAYLOAD_BYTES = ((BAND_CELLS*2+7)//8 + (BAND_CELLS*4+7)//8 + WIDE_CELLS
                 + NEAR_CELLS*2 + (NEAR_CELLS+7)//8 + (NEAR_CELLS*4+7)//8)
PACKET_BYTES = HEADER_BYTES+PAYLOAD_BYTES+CRC.size


def age_codes(age_s):
    """Conservative ceil(10*age); inf/>=1.5 -> 15, no floor/round TTL extension."""
    if bool(torch.isnan(age_s).any() | (age_s < 0).any()):
        raise ValueError('invalid evidence age')
    return torch.ceil(age_s/S.AGE_STEP_S).clamp(0, S.AGE_MAX_CODE).to(torch.uint8)


def age_upper_s(code):
    if bool((code > S.AGE_MAX_CODE).any()):
        raise ValueError('invalid age code')
    value = code.float()*S.AGE_STEP_S
    return torch.where(code == S.AGE_MAX_CODE, torch.inf, value)


def pack_bits(values, bits):
    """Flat LSB-first fields, zero high padding; CPU serializer, not a GPU kernel."""
    if bits not in (1, 2, 4):
        raise ValueError('packing bits')
    if bool(((values < 0) | (values >= (1 << bits))).any()):
        raise ValueError('out-of-range packed value')
    values = values.detach().cpu().flatten().tolist()
    out = bytearray((len(values)*bits+7)//8)
    for i, code in enumerate(values):
        out[i*bits//8] |= int(code) << (i*bits % 8)
    return bytes(out)


def unpack_bits(data, count, bits):
    if bits not in (1, 2, 4) or len(data) != (count*bits+7)//8:
        raise ValueError('packed array length')
    padding = count*bits % 8
    if padding and data[-1] >> padding:
        raise ValueError('nonzero packed padding')
    return torch.tensor([(data[i*bits//8] >> (i*bits % 8)) & ((1 << bits)-1) for i in range(count)], dtype=torch.uint8)


def _metadata(cfg):
    bands = [v for _, low, high in S.BANDS for v in (low, high)]
    return META.pack(S.WIDE_NX, S.WIDE_NY, S.NEAR_NX, S.NEAR_NY,
                     S.WIDE_RES, S.WIDE_X0, S.WIDE_Y0, S.NEAR_RES, S.NEAR_X0, S.NEAR_Y0,
                     S.NEAR_CLEAR_MAX, S.AGE_STEP_S, *bands, S.SUPPORT_Z0, S.SUPPORT_STEP, cfg.ttl_s)


@dataclass(frozen=True)
class DecodedOBS1:
    wide: MapFrame                 # batch size 1, CPU tensors; raw age upper bounds
    near: NearFrame
    evaluated_at_ms: int | None = None

    def valid_at(self, now_ms, *, config=None):
        """Same decoded packet, with transport/history elapsed time added ONCE."""
        cfg = config or load_config()
        reference_ms = int(self.wide.sample_time_ms[0]) if self.evaluated_at_ms is None else self.evaluated_at_ms
        elapsed = (now_ms-reference_ms)/1000
        if elapsed < 0:
            raise ValueError('future snapshot timestamp')
        age = self.wide.age_s+elapsed
        state = torch.where(age <= cfg.ttl_s, self.wide.state, S.UNKNOWN)
        # A per-cell age must also protect clearance values derived from OTHER
        # cells. Expire the complete distance snapshot when any contributing
        # occupied evidence might have expired. Current raw-only builders give
        # all occupied sources age zero; for arbitrary decoded inputs this is a
        # deliberately conservative minimum finite lifetime across near cells.
        finite = self.near.age_s[torch.isfinite(self.near.age_s)]
        global_stale = bool((finite+elapsed > cfg.ttl_s).any()) if finite.numel() else False
        near_age = self.near.age_s+elapsed
        unknown = self.near.unknown | (near_age > cfg.ttl_s) | global_stale
        return DecodedOBS1(MapFrame(state, age, self.wide.support, self.wide.sample_pose,
                                   self.wide.sample_time_ms, self.wide.seq, self.wide.epoch),
                           NearFrame(self.near.clearance, unknown, near_age), now_ms)


def encode_obs1(wide, near, *, env_index=0, config=None):
    """Encode one env's MapFrame/NearFrame to a complete, immutable bytes object."""
    cfg = config or load_config()
    n = wide.state.shape[0]
    if wide.state.shape != (n, 3, S.WIDE_NY, S.WIDE_NX) or wide.age_s.shape != wide.state.shape or wide.support.shape != (n, S.WIDE_NY, S.WIDE_NX):
        raise ValueError('wide shapes')
    if near.clearance.shape != (n, 2, S.NEAR_NY, S.NEAR_NX) or near.unknown.shape != (n, S.NEAR_NY, S.NEAR_NX) or near.age_s.shape != near.unknown.shape:
        raise ValueError('near shapes')
    if wide.state.dtype != torch.uint8 or wide.support.dtype != torch.uint8 or near.clearance.dtype != torch.uint8 or near.unknown.dtype != torch.bool:
        raise ValueError('wire array dtypes')
    if wide.sample_pose.shape != (n, 4) or not bool(torch.isfinite(wide.sample_pose).all()):
        raise ValueError('pose')
    for field in (wide.sample_time_ms, wide.seq, wide.epoch):
        if field.shape != (n,) or field.dtype != torch.int64 or bool((field < 0).any()):
            raise ValueError('timestamp/seq/epoch')
    if not 0 <= env_index < n or bool((wide.state > S.OCCUPIED).any()):
        raise ValueError('reserved state/env index')
    i = env_index
    max_support_code = round((cfg.max_z_m-S.SUPPORT_Z0)/S.SUPPORT_STEP)
    bad_support = (wide.support[i] > max_support_code) & (wide.support[i] != S.SUPPORT_UNKNOWN)
    if bool(bad_support.any()):
        raise ValueError('support code outside configured range')
    wa = age_codes(wide.age_s[i]); na = age_codes(near.age_s[i])
    state = torch.where(age_upper_s(wa) <= cfg.ttl_s, wide.state[i], S.UNKNOWN)
    state = torch.where(wide.support[i][None] != S.SUPPORT_UNKNOWN, state, S.UNKNOWN)
    unknown = near.unknown[i] | (age_upper_s(na) > cfg.ttl_s)
    payload = (pack_bits(state, 2)+pack_bits(wa, 4)+bytes(wide.support[i].flatten().cpu().tolist())
               +bytes(near.clearance[i].flatten().cpu().tolist())+pack_bits(unknown.to(torch.uint8), 1)+pack_bits(na, 4))
    header = BASE.pack(MAGIC, VERSION, HEADER_BYTES, PACKET_BYTES, int(wide.sample_time_ms[i]),
                       int(wide.seq[i]), int(wide.epoch[i]), *wide.sample_pose[i].cpu().tolist())+_metadata(cfg)
    data = header+payload
    return data+CRC.pack(zlib.crc32(data))


def decode_obs1(data, *, now_ms=None, config=None):
    """Strict wire decode, optionally age at now_ms; rejects before using payload."""
    cfg = config or load_config()
    if not isinstance(data, bytes) or len(data) != PACKET_BYTES:
        raise ValueError('OBS1 exact length')
    magic, version, header, total, stamp, seq, epoch, x, y, z, yaw = BASE.unpack_from(data)
    if (magic, version, header, total) != (MAGIC, VERSION, HEADER_BYTES, PACKET_BYTES):
        raise ValueError('OBS1 header/version')
    if any(v > torch.iinfo(torch.int64).max for v in (stamp, seq, epoch)) or not all(math.isfinite(v) for v in (x, y, z, yaw)):
        raise ValueError('OBS1 pose/counters')
    if data[BASE.size:HEADER_BYTES] != _metadata(cfg):
        raise ValueError('OBS1 geometry metadata')
    if CRC.unpack_from(data, len(data)-CRC.size)[0] != zlib.crc32(data[:-CRC.size]):
        raise ValueError('OBS1 CRC32')
    cursor = HEADER_BYTES
    def read(count, bits=8):
        nonlocal cursor
        size = (count*bits+7)//8
        block = data[cursor:cursor+size]
        cursor += size
        return torch.tensor(list(block), dtype=torch.uint8) if bits == 8 else unpack_bits(block, count, bits)
    state = read(BAND_CELLS, 2).reshape(1, 3, S.WIDE_NY, S.WIDE_NX)
    if bool((state == S.RESERVED).any()):
        raise ValueError('OBS1 reserved state 3')
    wa = read(BAND_CELLS, 4).reshape_as(state)
    support = read(WIDE_CELLS).reshape(1, S.WIDE_NY, S.WIDE_NX)
    max_support_code = round((cfg.max_z_m-S.SUPPORT_Z0)/S.SUPPORT_STEP)
    if bool(((support > max_support_code) & (support != S.SUPPORT_UNKNOWN)).any()):
        raise ValueError('OBS1 support code outside configured range')
    clearance = read(NEAR_CELLS*2).reshape(1, 2, S.NEAR_NY, S.NEAR_NX)
    unknown = read(NEAR_CELLS, 1).reshape(1, S.NEAR_NY, S.NEAR_NX).bool()
    na = read(NEAR_CELLS, 4).reshape(1, S.NEAR_NY, S.NEAR_NX)
    age = age_upper_s(wa)
    state = torch.where((age <= cfg.ttl_s) & (support[:, None] != S.SUPPORT_UNKNOWN), state, S.UNKNOWN)
    near_age = age_upper_s(na)
    unknown |= near_age > cfg.ttl_s
    wide = MapFrame(state, age, support, torch.tensor([[x, y, z, yaw]], dtype=torch.float64),
                    torch.tensor([stamp]), torch.tensor([seq]), torch.tensor([epoch]))
    decoded = DecodedOBS1(wide, NearFrame(clearance, unknown, near_age))
    return decoded if now_ms is None else decoded.valid_at(now_ms, config=cfg)


class OBS1Receiver:
    """Monotonic per-stream seq/epoch acceptance; receiver owns reset handling."""
    def __init__(self, *, config=None):
        self.config = config or load_config()
        self.last = None

    def receive(self, data, *, now_ms):
        value = decode_obs1(data, now_ms=now_ms, config=self.config)
        key = (int(value.wide.epoch[0]), int(value.wide.seq[0]))
        if self.last is not None and key <= self.last:
            raise ValueError('old/duplicate OBS1 epoch or seq')
        self.last = key
        return value
