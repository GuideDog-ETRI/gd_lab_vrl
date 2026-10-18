"""Public policy-side conversion: decoded/packed capture history -> 27ch/3ch.

No simulator imports. Occupied/free/normalized age are interleaved by frame,
then by band as schema.py requires. Missing history is UNKNOWN with age=1.
"""
import torch
from . import schema as S
from .map_config import load_config
from .map_types import NEAR_GRID, NearFrame
from .packets import age_codes, age_upper_s, decode_obs1
from .warp import warp_categorical


def _warp_near(near, sample_pose, current_pose, *, config):
    """Conservative centre-distance lower bound and unknown-preserving warp."""
    grid = NEAR_GRID
    n = sample_pose.shape[0]
    state = torch.where(near.unknown[:, None], S.UNKNOWN, S.FREE).to(torch.uint8)
    states, ages, _ = warp_categorical(state, near.age_s[:, None], sample_pose, current_pose, grid=grid, config=config)
    unknown = states[:, 0] != S.FREE
    xy = grid.xy(device=sample_pose.device, dtype=sample_pose.dtype)
    angle = current_pose[:, 3]-sample_pose[:, 3]
    ca, sa = angle.cos()[:, None, None], angle.sin()[:, None, None]
    delta = current_pose[:, :2]-sample_pose[:, :2]
    co, si = sample_pose[:, 3].cos(), sample_pose[:, 3].sin()
    tx = (co*delta[:, 0]+si*delta[:, 1])[:, None, None]
    ty = (-si*delta[:, 0]+co*delta[:, 1])[:, None, None]
    x, y = ca*xy[..., 0]-sa*xy[..., 1]+tx, sa*xy[..., 0]+ca*xy[..., 1]+ty
    ix = torch.floor((x-grid.x0)/grid.res+.5).long()
    iy = torch.floor((y-grid.y0)/grid.res+.5).long()
    half = grid.res/2
    bound = half+half*(ca.abs()+sa.abs())-config.overlap_epsilon
    clearance = torch.full_like(near.clearance, 255)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix+dx, iy+dy
            vx, vy = grid.x0+cx*grid.res-x, grid.y0+cy*grid.res-y
            overlap = ((vx.abs() < bound) & (vy.abs() < bound)
                       & ((ca*vx+sa*vy).abs() < bound) & ((-sa*vx+ca*vy).abs() < bound)
                       & (cx >= 0) & (cx < grid.nx) & (cy >= 0) & (cy < grid.ny))
            index = (cy.clamp(0, grid.ny-1)*grid.nx+cx.clamp(0, grid.nx-1)).flatten(1)
            code = near.clearance.flatten(2).gather(2, index[:, None].expand(-1, 2, -1)).reshape_as(clearance)
            # Distance is 1-Lipschitz. Subtract centre displacement before taking
            # a minimum. Integer identity warp remains exactly unchanged.
            diff = torch.sqrt(vx.square()+vy.square())
            diff = torch.where(diff <= config.overlap_epsilon, 0., diff)
            lower = (code.float()-diff[:, None]/S.NEAR_CLEAR_MAX*255).clamp_min(0).floor().to(torch.uint8)
            clearance = torch.minimum(clearance, torch.where(overlap[:, None], lower, 255))
    return NearFrame(clearance, unknown, ages[:, 0])


def network_inputs(frames, near, current_pose, now_ms, *, config=None):
    """Newest-first MapFrame list + latest NearFrame -> (wide27, near3).

    now_ms is int or [N] CLOCK_MONOTONIC ms. Uses only supplied arrived captures.
    Quantization is applied before elapsed time, matching packet reconstruction.
    Old-epoch rows become UNKNOWN; duplicate/non-descending same-epoch seq is
    rejected. Caller must not pass DecodedOBS1.valid_at output here, as elapsed
    time is applied once inside this function; use raw decode_obs1 instead.
    """
    cfg = config or load_config()
    if not frames or len(frames) > S.WIDE_FRAMES:
        raise ValueError('one to three arrived captures required')
    latest = frames[0]
    n = latest.state.shape[0]
    if current_pose.shape != (n, 4):
        raise ValueError('current pose shape')
    now = torch.as_tensor(now_ms, device=latest.state.device, dtype=torch.int64).expand(n)
    out = []
    for i in range(S.WIDE_FRAMES):
        if i >= len(frames):
            empty = torch.zeros((n, 3, 3, S.WIDE_NY, S.WIDE_NX), device=latest.state.device)
            empty[:, :, 2] = 1
            out.append(empty.flatten(1, 2)); continue
        frame = frames[i]
        same = frame.epoch == latest.epoch
        if i and bool((same & (frame.seq >= frames[i-1].seq)).any()):
            raise ValueError('history must be newest first, unique seq')
        elapsed = (now-frame.sample_time_ms).double()/1000
        if bool((elapsed < 0).any()):
            raise ValueError('future snapshot')
        upper = age_upper_s(frame.age_s if frame.age_s.dtype == torch.uint8 else age_codes(frame.age_s))
        encoded_state = torch.where((upper <= cfg.ttl_s) & (frame.support[:, None] != S.SUPPORT_UNKNOWN), frame.state, S.UNKNOWN)
        state, age, _ = warp_categorical(encoded_state, upper, frame.sample_pose, current_pose, config=cfg)
        age = age+elapsed[:, None, None, None]
        valid = (age <= cfg.ttl_s) & same[:, None, None, None]
        state = torch.where(valid, state, S.UNKNOWN)
        normalized_age = (age/(S.AGE_STEP_S*S.AGE_MAX_CODE)).clamp(0, 1).float()
        normalized_age = torch.where(same[:, None, None, None], normalized_age, 1.)
        channels = torch.stack(((state == S.OCCUPIED).float(), (state == S.FREE).float(), normalized_age), 2)
        out.append(channels.flatten(1, 2))
    wide = torch.cat(out, 1)
    near_raw = NearFrame(near.clearance, near.unknown, age_upper_s(near.age_s if near.age_s.dtype == torch.uint8 else age_codes(near.age_s)))
    aligned = _warp_near(near_raw, latest.sample_pose, current_pose, config=cfg)
    elapsed = ((now-latest.sample_time_ms).double()/1000)[:, None, None]
    unknown = aligned.unknown | (aligned.age_s+elapsed > cfg.ttl_s)
    fine = torch.cat((aligned.clearance.float()/255., unknown[:, None].float()), 1)
    return wide, fine


def packed_network_inputs(packets, current_pose, now_ms, *, config=None):
    """Single-env newest-first OBS1 bytes list -> tensors; same dense conversion."""
    decoded = [decode_obs1(p, config=config) for p in packets]
    return network_inputs([v.wide for v in decoded], decoded[0].near, current_pose, now_ms, config=config)
