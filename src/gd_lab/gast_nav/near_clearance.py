"""Independent raw-ray 5 cm occupancy and unsigned cell-area clearance (C2.4)."""
import math
import torch
from .map_backend import use_triton
from . import schema as S
from .map_config import load_config
from .map_types import Grid, NearFrame
from .obstacle_map import ray_evidence
from .ray_geometry import resolve_first_hits


def extended_near_grid():
    pad = int(math.ceil(S.NEAR_CLEAR_MAX/S.NEAR_RES))
    return Grid(S.NEAR_NX+2*pad, S.NEAR_NY+2*pad, S.NEAR_RES,
                S.NEAR_X0-pad*S.NEAR_RES, S.NEAR_Y0-pad*S.NEAR_RES), pad


def clearance_reference(state, *, grid, config=None):
    """[N,3,Y,X] -> [N,2,Y,X] metres to nearest occupied CELL BOUNDARY.

    Unknown is not an obstacle. With no occupied cells distance saturates at
    NEAR_CLEAR_MAX, and unknown is signalled separately. Chunked tensor reference
    computes exact Euclidean point-to-axis-aligned-cell distance (no scipy).
    """
    cfg = config or load_config()
    if state.ndim != 4 or state.shape[1:] != (3, grid.ny, grid.nx):
        raise ValueError('fine state shape')
    if bool((state > S.OCCUPIED).any()):
        raise ValueError('reserved state')
    n = state.shape[0]
    xy = grid.xy(device=state.device, dtype=torch.float32).reshape(-1, 2)
    occupied = torch.stack(((state[:, :2] == S.OCCUPIED).any(1), state[:, 2] == S.OCCUPIED), 1).flatten(2)
    result = torch.full((n, 2, len(xy)), S.NEAR_CLEAR_MAX, device=state.device)
    # Only offsets within clipping radius matter. Grid-cell distance is exact;
    # unlike centre-distance EDT, distance at the occupied cell itself is zero.
    radius = int(math.ceil(S.NEAR_CLEAR_MAX/grid.res+.5))
    full = result.reshape(n, 2, grid.ny, grid.nx)
    occ = occupied.reshape_as(full)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            dist = math.hypot(max(abs(dx)*grid.res-grid.res/2, 0), max(abs(dy)*grid.res-grid.res/2, 0))
            if dist > S.NEAR_CLEAR_MAX:
                continue
            moved = torch.roll(occ, (dy, dx), (-2, -1))
            if dy > 0: moved[..., :dy, :] = False
            if dy < 0: moved[..., dy:, :] = False
            if dx > 0: moved[..., :, :dx] = False
            if dx < 0: moved[..., :, dx:] = False
            full = torch.minimum(full, torch.where(moved, dist, S.NEAR_CLEAR_MAX))
    return full


def clearance_from_state(state, *, grid, config=None):
    if use_triton(state):
        if state.ndim!=4 or state.shape[1:]!=(3,grid.ny,grid.nx):raise ValueError('fine state shape')
        if bool((state>S.OCCUPIED).any()):raise ValueError('reserved state')
        from .map_cuda import clearance_cuda
        return clearance_cuda(state,grid)
    return clearance_reference(state,grid=grid,config=config)


def build_near(rays, fine_support_height, *, boxes=None, capsules=None, config=None, return_evidence=False, measurement=None, certified=False, return_proof=False):
    """Raw rays + INDEPENDENT observed support on ROI+0.5m -> NearFrame.

    fine_support_height must have [N,45,57]; this function never upsamples wide
    occupancy. Legacy mode builds each fine capture from current rays only. UPDATE167
    opt-in measurement uses explicit timestamped frusta and preserves their
    oldest necessary age; no implicit stale/free cache is introduced.
    """
    cfg = config or load_config()
    if return_proof and not (certified or measurement is not None):
        raise ValueError('return_proof requires certified=True or measurement')
    grid, pad = extended_near_grid()
    resolved = resolve_first_hits(rays, boxes=boxes, capsules=capsules, config=cfg)
    state, _, evidence_age = ray_evidence(resolved, fine_support_height, grid=grid, config=cfg, measurement=measurement, certified=certified, return_age=True)
    metres = clearance_from_state(state, grid=grid, config=cfg)
    region = (slice(None), slice(pad, pad+S.NEAR_NY), slice(pad, pad+S.NEAR_NX))
    unknown = (state == S.UNKNOWN).any(1)[region]
    code = (metres[..., pad:pad+S.NEAR_NY, pad:pad+S.NEAR_NX]/S.NEAR_CLEAR_MAX*255).round().clamp(0, 255).to(torch.uint8)
    observed = (state != S.UNKNOWN).any(1)[region]
    age = torch.where(observed, 0., torch.inf)
    if certified or measurement is not None:
        # Retain the oldest necessary band age. All-band unknown stays unchanged.
        age = evidence_age.amax(1)[region]
    frame = NearFrame(code, unknown, age)
    if return_proof:
        # Supervisor low-band proof consumes state/support/age independently;
        # it must not reinterpret the policy's all-band unknown bit.
        return frame, state[..., pad:pad+S.NEAR_NY, pad:pad+S.NEAR_NX], (torch.isfinite(fine_support_height[region]) & (fine_support_height[region]>=cfg.min_z_m) & (fine_support_height[region]<=cfg.max_z_m)), evidence_age[..., pad:pad+S.NEAR_NY, pad:pad+S.NEAR_NX]
    if return_evidence:
        # Internal evidence only: preserve the all-band policy/wire unknown bit.
        cropped = state[..., pad:pad+S.NEAR_NY, pad:pad+S.NEAR_NX]
        support = torch.isfinite(fine_support_height[region])
        return frame, cropped, support
    return frame
