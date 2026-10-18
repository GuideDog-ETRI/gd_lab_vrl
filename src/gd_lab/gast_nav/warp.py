"""Area-overlap categorical yaw warp, not nearest/bilinear interpolation.

Occupied wins any positive-area overlap. FREE requires every overlapping source
cell FREE and the entire destination cell inside the source map. This cannot
expand free into unknown or erase a subcell occupied hit on rotation.
"""
import torch
from . import schema as S
from .map_types import WIDE_GRID
from .map_config import load_config


def warp_categorical(state, age_s, sample_pose, current_pose, *, grid=WIDE_GRID, config=None, support=None, support_world_z=None, return_support_height=False):
    """[N,B,Y,X] + poses[N,4] -> (uint8 state, evidence age, support or None).

    Age excludes time since sample; caller adds that once. Scalar support is
    valid only when overlapped source cells agree on world support height; its height
    is not used to reclassify historical bands. Poses use odom x,y,z,yaw (radians).
    """
    cfg = config or load_config()
    if state.ndim != 4 or state.shape[-2:] != (grid.ny, grid.nx) or age_s.shape != state.shape or state.dtype != torch.uint8:
        raise ValueError('categorical state/age shape or dtype')
    n, bands = state.shape[:2]
    if sample_pose.shape != (n, 4) or current_pose.shape != (n, 4) or not bool(torch.isfinite(sample_pose).all() & torch.isfinite(current_pose).all()):
        raise ValueError('pose shape/values')
    if bool((state > S.OCCUPIED).any()) or bool(torch.isnan(age_s).any() | (age_s < 0).any()):
        raise ValueError('invalid state or age')
    if support is not None and (support.shape != (n, grid.ny, grid.nx) or support.dtype != torch.uint8):
        raise ValueError('support shape/dtype')
    xy = grid.xy(device=state.device, dtype=sample_pose.dtype)
    angle = current_pose[:, 3]-sample_pose[:, 3]
    ca, sa = angle.cos()[:, None, None], angle.sin()[:, None, None]
    delta = current_pose[:, :2]-sample_pose[:, :2]
    co, si = sample_pose[:, 3].cos(), sample_pose[:, 3].sin()
    tx = (co*delta[:, 0]+si*delta[:, 1])[:, None, None]
    ty = (-si*delta[:, 0]+co*delta[:, 1])[:, None, None]
    # Destination cell centres expressed in the sample frame.
    x = ca*xy[..., 0]-sa*xy[..., 1]+tx
    y = sa*xy[..., 0]+ca*xy[..., 1]+ty
    ix = torch.floor((x-grid.x0)/grid.res+.5).long()
    iy = torch.floor((y-grid.y0)/grid.res+.5).long()
    half = grid.res/2
    extent = half*(ca.abs()+sa.abs())
    eps = cfg.overlap_epsilon
    inside = ((x-extent >= grid.x0-half-eps) & (x+extent <= grid.x0+(grid.nx-1)*grid.res+half+eps)
              & (y-extent >= grid.y0-half-eps) & (y+extent <= grid.y0+(grid.ny-1)*grid.res+half+eps))
    any_occ = torch.zeros_like(state, dtype=torch.bool)
    all_free = inside[:, None].expand_as(state).clone()
    max_occ_age = torch.zeros_like(age_s)
    max_age = torch.zeros_like(age_s)
    count = torch.zeros((n, grid.ny, grid.nx), dtype=torch.int64, device=state.device)
    support_code = torch.full((n, grid.ny, grid.nx), S.SUPPORT_UNKNOWN, dtype=torch.uint8, device=state.device)
    support_same = inside.clone()
    support_world = torch.zeros((n, grid.ny, grid.nx), device=state.device, dtype=sample_pose.dtype)
    if support_world_z is not None and support_world_z.shape != (n, grid.ny, grid.nx):
        raise ValueError('support_world_z shape')
    flat = state.flatten(2)
    age_flat = age_s.flatten(2)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix+dx, iy+dy
            sx, sy = grid.x0+cx*grid.res, grid.y0+cy*grid.res
            vx, vy = sx-x, sy-y
            # Separating axes: source x/y, and destination's rotated x/y.
            bound = half+extent-eps
            overlap = ((vx.abs() < bound) & (vy.abs() < bound)
                       & ((ca*vx+sa*vy).abs() < bound) & ((-sa*vx+ca*vy).abs() < bound))
            valid = (cx >= 0) & (cx < grid.nx) & (cy >= 0) & (cy < grid.ny)
            index = (cy.clamp(0, grid.ny-1)*grid.nx+cx.clamp(0, grid.nx-1)).flatten(1)
            codes = flat.gather(2, index[:, None].expand(-1, bands, -1)).reshape_as(state)
            ages = age_flat.gather(2, index[:, None].expand(-1, bands, -1)).reshape_as(age_s)
            real = overlap & valid
            occ = real[:, None] & (codes == S.OCCUPIED)
            any_occ |= occ
            all_free &= ~(overlap[:, None] & (~valid[:, None] | (codes != S.FREE)))
            max_occ_age = torch.maximum(max_occ_age, torch.where(occ, ages, 0.))
            max_age = torch.maximum(max_age, torch.where(real[:, None], ages, 0.))
            if support is not None:
                code = support.flatten(1).gather(1, index).reshape(n, grid.ny, grid.nx)
                first = real & (count == 0)
                if support_world_z is None:
                    world = code.to(sample_pose.dtype)*S.SUPPORT_STEP+S.SUPPORT_Z0+sample_pose[:, 2, None, None]
                else:
                    world = support_world_z.flatten(1).gather(1, index).reshape(n, grid.ny, grid.nx)
                support_code = torch.where(first, code, support_code)
                support_world = torch.where(first, world, support_world)
                support_same &= ~(real & ((world-support_world).abs() > eps))
                support_same &= ~(real & ((code == S.SUPPORT_UNKNOWN) | ~torch.isfinite(world)))
            count += real.long()
    all_free &= (count > 0)[:, None]
    out = torch.full_like(state, S.UNKNOWN)
    out = torch.where(all_free, S.FREE, out)
    out = torch.where(any_occ, S.OCCUPIED, out)
    age = torch.where(any_occ, max_occ_age, torch.where(all_free, max_age, torch.inf))
    support_out = None
    if support is not None:
        from .support import encode_support
        height = support_world-current_pose[:, 2, None, None]
        valid_support = support_same & (count > 0)
        if return_support_height:
            support_out = torch.where(valid_support, height, torch.nan)
        else:
            support_out = encode_support(height, valid_support, config=cfg)
    return out, age, support_out
