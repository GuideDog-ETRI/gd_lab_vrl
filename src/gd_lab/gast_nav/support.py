"""Observed support growth, including the UPDATE159 C2 clarification.

FLAT/PASSAGE/TURN_SPOT (including absent route hint) accept surfaces only within
current support +/- configured 0.10m. Higher observed surfaces retain the measured
current floor reference for obstacle classification; they are NOT new support.
STAIR_UP/STAIR_DOWN/LANDING allow measured tread growth up to 0.25m per step.
Slope is the fitted surface normal's angle to +z, NOT riser/run.
"""
import math
import torch
from . import schema as S
from .map_config import load_config
from .map_types import WIDE_GRID


def encode_support(height, valid=None, *, config=None):
    """[N,Y,X] metres -> uint8; 0..254 = -1.0..+4.08m, 255 unknown."""
    cfg = config or load_config()
    good = torch.isfinite(height) & (height >= cfg.min_z_m) & (height <= cfg.max_z_m)
    if valid is not None:
        good &= valid.bool()
    code = ((height - S.SUPPORT_Z0) / S.SUPPORT_STEP).round().clamp(0, 254).to(torch.uint8)
    return torch.where(good, code, S.SUPPORT_UNKNOWN)


def decode_support(code, *, config=None):
    """uint8 -> (height metres, known bool); never clamp an invalid byte to floor."""
    cfg = config or load_config()
    if code.dtype != torch.uint8:
        raise ValueError('support must be uint8')
    z = code.float() * S.SUPPORT_STEP + S.SUPPORT_Z0
    max_code = round((cfg.max_z_m - S.SUPPORT_Z0) / S.SUPPORT_STEP)
    known = (code != S.SUPPORT_UNKNOWN) & (code <= max_code)
    return torch.where(known, z, torch.nan), known


def contact_seed_mask(contact_xyz, contact_valid, *, grid=WIDE_GRID):
    """Observed foot contacts [N,F,3]/[N,F] -> [N,NY,NX] seeds.

    This only selects cells: grow_support still requires an observed horizontal
    hit in every selected cell. It does not fill missing terrain with foot/GT z.
    """
    if contact_xyz.ndim != 3 or contact_xyz.shape[-1] != 3 or contact_valid.shape != contact_xyz.shape[:2]:
        raise ValueError('contact tensor shapes')
    n = contact_xyz.shape[0]
    x = torch.floor((contact_xyz[..., 0] - grid.x0) / grid.res + .5).long()
    y = torch.floor((contact_xyz[..., 1] - grid.y0) / grid.res + .5).long()
    ok = contact_valid.bool() & torch.isfinite(contact_xyz).all(-1) & (x >= 0) & (x < grid.nx) & (y >= 0) & (y < grid.ny)
    out = torch.zeros((n, grid.ny*grid.nx), dtype=torch.int64, device=contact_xyz.device)
    out.scatter_add_(1, (y.clamp(0, grid.ny-1)*grid.nx+x.clamp(0, grid.nx-1)), ok.long())
    return out.reshape(n, grid.ny, grid.nx).bool()


def _grow(eligible, seeds, height=None, *, max_step=None, epsilon=0.):
    connected = seeds & eligible
    for _ in range(eligible.shape[1]*eligible.shape[2]):
        grown = connected.clone()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = torch.roll(connected, (dy, dx), (1, 2))
            if dy == -1: other[:, -1, :] = False
            if dy == 1: other[:, 0, :] = False
            if dx == -1: other[:, :, -1] = False
            if dx == 1: other[:, :, 0] = False
            edge = eligible & other
            if height is not None:
                z = torch.roll(height, (dy, dx), (1, 2))
                edge &= (height-z).abs() <= max_step+epsilon
            grown |= edge
        if torch.equal(grown, connected):
            break
        connected = grown
    return connected


def grow_support(height, observed, normals, seed_mask, *, segment_type=None,
                 current_support_height=None, surface_candidate=None, config=None):
    """[N,Y,X] hits + [N,Y,X,3] fitted normals -> (reference_z, code, is_support).

    segment_type is a schema SEGMENT_TYPES name, scalar integer code, or [N] code
    tensor (gdm-derived, NOT a GT terrain family). None means FLAT. Invalid hints
    fail explicitly. current_support_height is measured [N] base-relative floor
    z, required for flat-mode rows. surface_candidate optionally further limits
    support using measured evidence, never a mesh/material/semantic oracle.

    On flat routes, observed connected higher surfaces receive CURRENT floor
    reference in reference_z for band classification, but is_support=False.
    Missing hits/gaps remain NaN/255. For stairs, reference_z equals accepted
    observed tread heights; heights rejected by step/slope remain unknown.
    """
    cfg = config or load_config()
    if height.ndim != 3 or observed.shape != height.shape or seed_mask.shape != height.shape or normals.shape != (*height.shape, 3):
        raise ValueError('support tensor shapes')
    n = height.shape[0]
    if segment_type is None:
        segment_type = S.SEGMENT_TYPES.index('FLAT')
    if isinstance(segment_type, str):
        if segment_type not in S.SEGMENT_TYPES:
            raise ValueError('invalid segment_type')
        segment_type = S.SEGMENT_TYPES.index(segment_type)
    segment = torch.as_tensor(segment_type, device=height.device, dtype=torch.long).expand(n)
    if bool(((segment < 0) | (segment >= len(S.SEGMENT_TYPES))).any()):
        raise ValueError('invalid segment_type')
    stair = ((segment == S.SEGMENT_TYPES.index('STAIR_UP')) |
             (segment == S.SEGMENT_TYPES.index('STAIR_DOWN')) |
             (segment == S.SEGMENT_TYPES.index('LANDING')))
    if current_support_height is None:
        if bool((~stair).any()):
            raise ValueError('measured current_support_height required for flat routes')
        current_support_height = torch.zeros(n, device=height.device, dtype=height.dtype)
    reference = torch.as_tensor(current_support_height, device=height.device, dtype=height.dtype).expand(n)
    if bool((~stair & ~torch.isfinite(reference)).any()):
        raise ValueError('invalid measured current support')
    if surface_candidate is None:
        surface_candidate = torch.ones_like(observed, dtype=torch.bool)
    if surface_candidate.shape != height.shape:
        raise ValueError('surface_candidate shape')
    norm = torch.linalg.vector_norm(normals, dim=-1)
    cosz = normals[..., 2] / norm.clamp_min(torch.finfo(height.dtype).eps)
    observed_hit = observed.bool() & torch.isfinite(height)
    horizontal = (torch.isfinite(normals).all(-1) & (norm > 0)
                  & (cosz >= math.cos(math.radians(min(cfg.max_slope_deg, cfg.horizontal_max_slope_deg)))))
    eligible = (observed_hit & surface_candidate.bool() & horizontal
                & (height >= cfg.min_z_m) & (height <= cfg.max_z_m))
    flat_height = (height-reference[:, None, None]).abs() <= cfg.flat_support_tolerance_m+cfg.intersection_epsilon
    eligible &= stair[:, None, None] | flat_height
    connected = _grow(eligible, seed_mask.bool(), height,
                      max_step=cfg.max_step_m, epsilon=cfg.intersection_epsilon)
    # With the explicit FLAT route prior, a higher measured hit is an obstacle
    # above current observed support, not an elevated walkable tread. Never fill
    # a missing hit (platform gap) or a disconnected region with this prior.
    flat_observed = observed_hit & (height >= reference[:, None, None]-cfg.flat_support_tolerance_m)
    flat_reference = _grow(flat_observed, seed_mask.bool()) & ~stair[:, None, None]
    reference_z = torch.where(stair[:, None, None] | connected, height, reference[:, None, None])
    valid = connected | flat_reference
    code = encode_support(reference_z, valid, config=cfg)
    return torch.where(valid, reference_z, torch.nan), code, connected


def make_seed_mask(contact_xyz, contact_valid, terrain_xyz, terrain_valid, body_near_mask, *, grid=WIDE_GRID):
    """OR of measured foot contacts and valid body-near downward-terrain cells.

    body_near_mask is the caller's observed geometric body-neighbourhood mask,
    [N,T]; no floor labels are inferred or GT heights filled by this function.
    All returned cells still need hits/normals for grow_support to accept them.
    """
    if terrain_xyz.ndim != 3 or terrain_xyz.shape[-1] != 3 or terrain_valid.shape != terrain_xyz.shape[:2] or body_near_mask.shape != terrain_valid.shape:
        raise ValueError('terrain seed shapes')
    if terrain_xyz.shape[0] != contact_xyz.shape[0]:
        raise ValueError('seed env count')
    return (contact_seed_mask(contact_xyz, contact_valid, grid=grid)
            | contact_seed_mask(terrain_xyz, terrain_valid.bool() & body_near_mask.bool(), grid=grid))
