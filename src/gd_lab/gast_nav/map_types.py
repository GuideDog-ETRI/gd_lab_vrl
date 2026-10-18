"""Explicit C2 tensor records. Coordinates are metres in the sample body-yaw frame."""
from dataclasses import dataclass
import torch
from . import schema as S


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    res: float
    x0: float
    y0: float

    def xy(self, *, device, dtype):
        y, x = torch.meshgrid(torch.arange(self.ny, device=device, dtype=dtype),
                              torch.arange(self.nx, device=device, dtype=dtype), indexing='ij')
        return torch.stack((self.x0 + self.res*x, self.y0 + self.res*y), -1)


WIDE_GRID = Grid(S.WIDE_NX, S.WIDE_NY, S.WIDE_RES, S.WIDE_X0, S.WIDE_Y0)
NEAR_GRID = Grid(S.NEAR_NX, S.NEAR_NY, S.NEAR_RES, S.NEAR_X0, S.NEAR_Y0)


@dataclass(frozen=True)
class RayBatch:
    origins: torch.Tensor           # [N,R,3]
    directions: torch.Tensor        # [N,R,3], nonzero (normalized by resolve_first_hits)
    distances: torch.Tensor         # [N,R], +inf means no return within configured range
    self_hit: torch.Tensor          # [N,R] bool, applies to the supplied static hit only


@dataclass(frozen=True)
class DynamicBoxes:
    centres: torch.Tensor           # [N,K,3], sample frame
    half_extents: torch.Tensor      # [N,K,3], strictly positive
    yaw: torch.Tensor               # [N,K], rotation around +z


@dataclass(frozen=True)
class DynamicCapsules:
    starts: torch.Tensor            # [N,K,3]
    ends: torch.Tensor              # [N,K,3]
    radii: torch.Tensor             # [N,K], positive; arbitrary segment orientation


@dataclass(frozen=True)
class ResolvedRays:
    origins: torch.Tensor
    directions: torch.Tensor
    distances: torch.Tensor        # first stop, including self, bounded by sensor range
    occupied_hit: torch.Tensor     # false for self return or no return
    self_hit: torch.Tensor


@dataclass(frozen=True)
class MapFrame:
    state: torch.Tensor            # [N,3,NY,NX], uint8
    age_s: torch.Tensor            # same shape, conservative evidence age, inf if absent
    support: torch.Tensor          # [N,NY,NX], uint8
    sample_pose: torch.Tensor      # [N,4] odom x,y,z,yaw
    sample_time_ms: torch.Tensor   # [N] int64 CLOCK_MONOTONIC
    seq: torch.Tensor              # [N] int64, epoch-local
    epoch: torch.Tensor            # [N] int64
    support_world_z: torch.Tensor | None = None  # optional exact retained support; not an extra wire field


@dataclass(frozen=True)
class NearFrame:
    clearance: torch.Tensor        # [N,2,25,37], uint8, lower(B1+B2)/upper(B3)
    unknown: torch.Tensor          # [N,25,37], bool, OR over all bands
    age_s: torch.Tensor            # [N,25,37], oldest evidence used by the cell
