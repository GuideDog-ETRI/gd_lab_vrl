"""C2 first-hit/sub-bin evidence and batched three-capture history.

Free is certified only when *every* band sub-bin was traversed in this capture.
No partial free evidence is promoted by combining unrelated poses or captures.
Occupied carry is conservative up to TTL; history retains each sample support.
"""
from dataclasses import replace
import math
import torch
from . import schema as S
from .map_config import load_config
from .map_types import WIDE_GRID, MapFrame
from .ray_geometry import resolve_first_hits, ray_box_interval
from .support import encode_support


def subbin_edges():
    """B1 has a clipped last bin (0.23..0.30); B2=5, B3=14."""
    out = []
    for _, low, high in S.BANDS:
        count = int(math.ceil((high-low)/S.SUBBIN-1e-9))
        out.append([(low+i*S.SUBBIN, min(high, low+(i+1)*S.SUBBIN)) for i in range(count)])
    return out


def ray_evidence(resolved, support_height, *, grid=WIDE_GRID, config=None):
    """Resolved rays + observed support [N,NY,NX] -> (state, subbin_free).

    Exact segment/AABB positive-length traversal; rays stop strictly before
    their first hit. Bands use half-open height intervals [low,high), except the
    final upper edge which is included. NaN support => all bands UNKNOWN.
    subbin_free is a tuple of [N,NY,NX,subbins] diagnostic tensors.
    """
    cfg = config or load_config()
    n = resolved.origins.shape[0]
    if support_height.shape != (n, grid.ny, grid.nx):
        raise ValueError('support/grid shape')
    if resolved.origins.device != support_height.device:
        raise ValueError('ray/support device mismatch')
    centres = grid.xy(device=support_height.device, dtype=support_height.dtype).reshape(-1, 2)
    support = support_height.flatten(1)
    known = torch.isfinite(support) & (support >= cfg.min_z_m) & (support <= cfg.max_z_m)
    result = torch.zeros((n, len(S.BANDS), grid.ny*grid.nx), dtype=torch.uint8, device=support.device)
    diagnostics = []
    end = resolved.origins + resolved.directions*resolved.distances[..., None]
    ix = torch.floor((end[..., 0]-grid.x0)/grid.res+.5).long()
    iy = torch.floor((end[..., 1]-grid.y0)/grid.res+.5).long()
    hit_in = resolved.occupied_hit & (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    index = iy.clamp(0, grid.ny-1)*grid.nx+ix.clamp(0, grid.nx-1)
    hit_support = support.gather(1, index)
    hit_z = end[..., 2]-hit_support
    for band, edges in enumerate(subbin_edges()):
        passed = torch.zeros((n, grid.ny*grid.nx, len(edges)), dtype=torch.bool, device=support.device)
        zlo = support.new_tensor([a for a, _ in edges])
        zhi = support.new_tensor([b for _, b in edges])
        for c in range(0, centres.shape[0], cfg.cell_chunk):
            xy = centres[c:c+cfg.cell_chunk]
            h = support[:, c:c+cfg.cell_chunk]
            # [N,C,K,3] boxes; absent support boxes are never accepted.
            k = len(edges)
            low = torch.cat(((xy[None, :, None, :]-grid.res/2).expand(n, -1, k, -1), (h[..., None]+zlo)[..., None]), -1)
            high = torch.cat(((xy[None, :, None, :]+grid.res/2).expand(n, -1, k, -1), (h[..., None]+zhi)[..., None]), -1)
            traversed = passed[:, c:c+xy.shape[0]]
            for r in range(0, resolved.origins.shape[1], cfg.ray_chunk):
                o = resolved.origins[:, r:r+cfg.ray_chunk, None, None, :]
                d = resolved.directions[:, r:r+cfg.ray_chunk, None, None, :]
                entry, exit = ray_box_interval(o, d, low[:, None], high[:, None], cfg.intersection_epsilon)
                stop = resolved.distances[:, r:r+cfg.ray_chunk, None, None]
                before_hit = torch.minimum(exit, stop-cfg.intersection_epsilon)
                # A ray exactly on a voxel face does not certify either open
                # volume beside it. Do not double-count horizontal sub-bin edges.
                boundary = ((d.abs() <= cfg.intersection_epsilon) &
                            ((o <= low[:, None]+cfg.intersection_epsilon) |
                             (o >= high[:, None]-cfg.intersection_epsilon))).any(-1)
                crossed = ((before_hit-entry.clamp_min(0) > cfg.intersection_epsilon)
                           & ~boundary & known[:, None, c:c+xy.shape[0], None])
                traversed |= crossed.any(1)
        result[:, band] = torch.where(passed.all(-1) & known, S.FREE, S.UNKNOWN)
        _, lo, hi = S.BANDS[band]
        in_z = (hit_z >= lo) & ((hit_z < hi) if band < len(S.BANDS)-1 else (hit_z <= hi))
        hits = hit_in & known.gather(1, index) & in_z
        occupied_count = torch.zeros_like(support, dtype=torch.int64)
        occupied_count.scatter_add_(1, index, hits.long())
        result[:, band] = torch.where(occupied_count > 0, S.OCCUPIED, result[:, band])
        diagnostics.append(passed.reshape(n, grid.ny, grid.nx, len(edges)))
    return result.reshape(n, len(S.BANDS), grid.ny, grid.nx), tuple(diagnostics)


class ObstacleMap:
    """Batched capture builder. update/reset are CPU-testable, device preserving.

    Epoch must increase on reset; seq starts at zero per env. History slots are
    newest first and immutable copies. No stale epoch/seq is accepted. One call
    is one capture for every supplied env; partial capture schedules should use
    separate instances or a caller-level per-env scheduler.
    """
    def __init__(self, num_envs, *, config=None, device='cpu'):
        self.config = config or load_config()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.epoch = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.next_seq = torch.zeros_like(self.epoch)
        self.history = []

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.epoch[ids] += 1
        self.next_seq[ids] = 0
        # Invalidate just reset rows in already pinned snapshots via new copies.
        updated = []
        for frame in self.history:
            state, age, support = frame.state.clone(), frame.age_s.clone(), frame.support.clone()
            state[ids] = S.UNKNOWN; age[ids] = torch.inf; support[ids] = S.SUPPORT_UNKNOWN
            updated.append(replace(frame, state=state, age_s=age, support=support))
        self.history = updated

    def update(self, rays, support_height, sample_pose, sample_time_ms, *, boxes=None, capsules=None):
        """Capture -> MapFrame, keeping up to three capture-local snapshots."""
        from .warp import warp_categorical
        from .packets import age_codes, age_upper_s
        if sample_pose.shape != (self.num_envs, 4) or sample_time_ms.shape != (self.num_envs,):
            raise ValueError('pose/time shapes')
        if not bool(torch.isfinite(sample_pose).all()) or sample_time_ms.dtype != torch.int64 or bool((sample_time_ms < 0).any()):
            raise ValueError('invalid pose/time')
        if sample_pose.device != self.device or support_height.device != self.device:
            raise ValueError('map device mismatch')
        resolved = resolve_first_hits(rays, boxes=boxes, capsules=capsules, config=self.config)
        state, _ = ray_evidence(resolved, support_height, config=self.config)
        current_occupied = state == S.OCCUPIED
        support = encode_support(support_height, config=self.config)
        world_support = support_height+sample_pose[:, 2, None, None]
        known_support = support != S.SUPPORT_UNKNOWN
        state = torch.where(known_support[:, None], state, S.UNKNOWN)
        age = torch.where(state != S.UNKNOWN, 0., torch.inf)
        if self.history:
            prev = self.history[0]
            elapsed = (sample_time_ms-prev.sample_time_ms).double()/1000
            same_epoch = prev.epoch == self.epoch
            if bool(((elapsed <= 0) & same_epoch).any()):
                raise ValueError('non-increasing capture timestamp')
            warped, old_age, old_height = warp_categorical(
                prev.state, prev.age_s, prev.sample_pose, sample_pose,
                config=self.config, support=prev.support,
                support_world_z=prev.support_world_z, return_support_height=True)
            old_age = old_age + elapsed[:, None, None, None]
            # The same ceil-age TTL semantics as the wire, but no refresh from
            # free rays or from changing the current support estimate.
            valid_old = age_upper_s(age_codes(old_age)) <= self.config.ttl_s
            old_occ = (warped == S.OCCUPIED) & valid_old & same_epoch[:, None, None, None]
            old_support = encode_support(old_height, config=self.config)
            reference_valid = old_support != S.SUPPORT_UNKNOWN
            carry = old_occ & reference_valid[:, None]
            keep_cell = carry.any(1)
            changed_reference = keep_cell & ((old_support != support) | ~known_support)
            # Retain all occupied bands on their ORIGINAL world support. A new
            # reference must not silently reclassify or refresh old evidence.
            # Other bands become UNKNOWN when the two references conflict.
            state = torch.where(carry, S.OCCUPIED, state)
            age = torch.where(carry & ~current_occupied, old_age, age)
            state = torch.where(changed_reference[:, None],
                                torch.where(carry, S.OCCUPIED, S.UNKNOWN).to(torch.uint8), state)
            age = torch.where(changed_reference[:, None],
                              torch.where(carry, old_age, torch.inf), age)
            support = torch.where(keep_cell, old_support, support)
            world_support = torch.where(keep_cell, old_height+sample_pose[:, 2, None, None], world_support)
            # Multiple source supports at a rotated edge cannot be represented
            # by ONE support byte. Fail closed; never manufacture FREE there.
            ambiguous = old_occ.any(1) & ~reference_valid
            state = torch.where(ambiguous[:, None] & ~current_occupied, S.UNKNOWN, state)
            age = torch.where(ambiguous[:, None] & ~current_occupied, torch.inf, age)
        frame = MapFrame(state.clone(), age.clone(), support.clone(), sample_pose.clone(), sample_time_ms.clone(), self.next_seq.clone(), self.epoch.clone(), world_support.clone())
        self.next_seq += 1
        self.history = [frame] + self.history[:S.WIDE_FRAMES-1]
        return frame

