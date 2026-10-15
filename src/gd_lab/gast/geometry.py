"""Simulator-independent terrain labels and ego-motion memory alignment."""
import math
import torch
from torch.nn import functional as F


def targets(clean, teacher_terrain=None, seen=None, gap_observed=None):
    """Packed clean height(187), valid(187), semantic gap(187), gap-known(187).

    Height uses the existing scanner_z-ground_z-0.5 convention, scaled by 5.
    Gap labels come from known gap terrain geometry, never missing depth.
    ``seen`` [B,187] bool (optional): cells a camera saw recently (``SeenCells``). They are supervised with
    the clean height and gap labels as well, so terrain that just left the field of view (e.g. a gap under
    the body) still trains the memory. The visibility channel keeps the current teacher-visible target.
    ``gap_observed`` [B,187] bool (optional): gap cells a camera looks through at deck height (rays pass the
    deck level and land deeper -- the depth pattern of a pit, even when no ray reaches the pit floor). Only the
    semantic gap channel (2) is additionally supervised there, with the clean gap label.
    """
    if gap_observed is not None:
        y, mask = targets(clean, teacher_terrain, seen)
        _, _, gap, known = clean.reshape(-1, 4, 11, 17).unbind(1)
        extra = gap_observed.reshape(-1, 11, 17).bool() & known.bool() & (gap > .5)
        mask = mask.clone()
        mask[..., 2] |= extra.flatten(1)
        return y, mask
    h, scan_valid, gap, known = clean.reshape(-1, 4, 11, 17).unbind(1)
    scan_valid = scan_valid.bool()
    if seen is not None:
        if teacher_terrain is None:
            raise ValueError("seen-cell supervision needs the teacher terrain (current visibility)")
        teacher_visible = teacher_terrain.reshape(-1, 2, 11, 17)[:, 1].bool()
        h = h / 5
        visibility_target = teacher_visible
        valid = (teacher_visible | seen.reshape(-1, 11, 17).bool()) & scan_valid & torch.isfinite(h)
        return _pack(h, visibility_target, valid, gap, known)
    if teacher_terrain is None:
        # Teacher-PPO auxiliary target: privileged scan, unchanged.
        h = h / 5
        visibility_target = scan_valid & torch.isfinite(h)
    else:
        if teacher_terrain.shape[-1] != 374:
            raise ValueError(f"teacher terrain must contain 187 heights + 187 visibility values, got {teacher_terrain.shape[-1]}")
        teacher_h, teacher_visible = teacher_terrain.reshape(-1, 2, 11, 17).unbind(1)
        h = teacher_h / 5
        # Predict teacher visibility independently from height-label validity.
        visibility_target = teacher_visible.bool()
    valid = visibility_target & scan_valid & torch.isfinite(h)
    return _pack(h, visibility_target, valid, gap, known)


def _pack(h, visibility_target, valid, gap, known):
    dx = -(h[:, :, 1:] - h[:, :, :-1])
    pair = valid[:, :, 1:] & valid[:, :, :-1]
    up, down = F.pad((dx > .04).float(), (0, 1)), F.pad((dx < -.04).float(), (0, 1))
    pair = F.pad(pair, (0, 1))
    dy = F.pad((h[:, 1:] - h[:, :-1]).abs(), (0, 0, 0, 1))
    local = F.pad(dx.abs(), (0, 1))
    support_valid = pair & F.pad(valid[:, 1:] & valid[:, :-1], (0, 0, 0, 1)) & known.bool()
    support = ((local < .04) & (dy < .04) & (gap < .5)).float()
    y = torch.stack((torch.nan_to_num(h), visibility_target.float(), gap, up, down, support), -1).flatten(1, 2)
    # Visibility is supervised across all cells; unknown heights remain masked.
    mask = torch.stack((valid, torch.ones_like(valid), known.bool() & valid, pair, pair, support_valid), -1).flatten(1, 2)
    return y, mask


def gap_lift_heights(z, gap, deck):
    """[N,187] world z of the walking deck over each gap cell: the lower of the nearest deck cell before and
    after it in the same grid row (x direction), or only the one present; NaN where the row has no deck cell.
    ``gap``/``deck`` [N,187] bool (deck = finite non-gap scan hits). Platform-gap decks differ by up to 0.16 m, so
    one tile-wide height would put the lifted point above or below one of the lips."""
    z, gap, deck = z.reshape(-1, 11, 17), gap.reshape(-1, 11, 17), deck.reshape(-1, 11, 17)
    col = torch.arange(17, device=z.device).expand_as(z)
    left = torch.where(deck, col, -1).cummax(-1).values                         # nearest deck at or before
    right = torch.where(deck, col, 17).flip(-1).cummin(-1).values.flip(-1)  # nearest deck at or after (17 = none)
    zl = torch.where(left >= 0, z.gather(-1, left.clamp(min=0)), torch.nan)
    zr = torch.where(right <= 16, z.gather(-1, right.clamp(max=16)), torch.nan)
    lift = torch.fmin(zl, zr)                                                    # fmin ignores a NaN side
    return torch.where(gap, lift, torch.nan).flatten(1)


def near_gap_rows(clean):
    """[B] bool: a known gap cell (semantic gap map, never missing depth) lies in the 11x17 grid around the body."""
    _, _, gap, known = clean.reshape(-1, 4, 11, 17).unbind(1)
    return ((gap > .5) & known.bool()).flatten(1).any(1)


def weighted_mean(values, weight):
    """Per-row values [B] averaged with row weights; equals values.mean() when every weight is 1."""
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def reconstruction_loss(pred, clean, teacher_terrain=None, return_components=False, row_weight=None, seen=None,
                        gap_observed=None):
    """``row_weight`` [B] scales whole samples (e.g. near-gap emphasis); None keeps every row at 1.
    ``seen``: optional recently-seen cells, see ``targets``."""
    y, mask = targets(clean, teacher_terrain, seen, gap_observed)
    loss = torch.cat((F.smooth_l1_loss(pred[..., :1], y[..., :1], reduction='none'),
                      F.binary_cross_entropy_with_logits(pred[..., 1:], y[..., 1:], reduction='none')), -1)
    weight = torch.ones_like(loss)
    weight[..., 2:5] = 1 + 4 * y[..., 2:5]
    if row_weight is not None:
        weight = weight * row_weight[:, None, None]
    weighted = loss * mask * weight
    total = weighted.sum() / (mask * weight).sum().clamp_min(1)
    if not return_components:
        return total
    height = weighted[..., 0].sum() / (mask * weight)[..., 0].sum().clamp_min(1)
    visibility = weighted[..., 1].sum() / (mask * weight)[..., 1].sum().clamp_min(1)
    visible_fraction = y[..., 1].mean()
    return total, height, visibility, visible_fraction


class SeenCells:
    """Per-env cells a camera delivered to the student within the last ``window_s`` seconds, in the current grid.

    Keeps the received visibility snapshots of that window with the pose they were taken at (ring buffer) and
    moves each one straight to the current pose (one transform per snapshot, so sub-cell motion never accumulates
    rounding loss). Each seen cell centre is carried to the current body frame and written to its nearest new
    cell (forward), so a cell still inside the grid is never lost before it expires; the nearest backward lookup
    (new cell centre -> old grid) is OR-ed in to fill the holes forward rounding leaves inside rotated areas.

    ``min_capture_period_s``: the shortest spacing between recorded stamps (the camera transport's minimum
    capture interval in policy steps x policy dt). It sizes the ring so a snapshot is never overwritten while it
    is inside the window; should a write still hit a live slot, the ring grows instead.
    """

    def __init__(self, num_envs, device, window_s, min_capture_period_s):
        if not (window_s > 0 and window_s == window_s and window_s < 60):
            raise ValueError("window_s must be finite and in (0, 60)")
        if not (min_capture_period_s > 0 and min_capture_period_s == min_capture_period_s):
            raise ValueError("min_capture_period_s must be finite and positive")
        self.window = float(window_s)
        k = math.ceil(window_s / min_capture_period_s - 1e-9) + 2
        self.masks = torch.zeros(num_envs, k, 187, dtype=torch.bool, device=device)
        self.poses = torch.zeros(num_envs, k, 3, device=device)
        self.stamps = torch.full((num_envs, k), float("-inf"), device=device)
        self.next = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, done):
        self.stamps[done] = float("-inf")
        self.masks[done] = False
        self.next[done] = 0

    def _grow(self):
        """Double the ring, keeping every slot index (new slots are empty and expired)."""
        n, k = self.stamps.shape
        self.masks = torch.cat((self.masks, self.masks.new_zeros(n, k, 187)), 1)
        self.poses = torch.cat((self.poses, self.poses.new_zeros(n, k, 3)), 1)
        self.stamps = torch.cat((self.stamps, torch.full_like(self.stamps, float("-inf"))), 1)
        # Continue writing into the new (empty) half; later writes are checked again before they overwrite.
        self.next = torch.full_like(self.next, k)

    @torch.no_grad()
    def update(self, rows, visible, pose_xyyaw, stamp, received=None):
        """visible [R,187] bool (cells delivered now); ``received`` [R] bool: False for frames the student did not
        get (augmentation drop or a hidden camera) -> nothing new is recorded for that row. Returns the seen mask
        [R,187] over the window (snapshots still inside it, moved to ``pose_xyyaw``)."""
        rows = torch.as_tensor(rows, device=self.masks.device)
        if received is None:
            received = torch.ones(len(rows), dtype=torch.bool, device=rows.device)
        write = rows[received]
        if len(write):
            slot = self.next[write]
            if ((stamp[received] - self.stamps[write, slot]) < self.window).any():
                self._grow()
                slot = self.next[write]
            self.masks[write, slot] = visible[received]
            self.poses[write, slot] = pose_xyyaw[received]
            self.stamps[write, slot] = stamp[received]
            self.next[write] = (slot + 1) % self.masks.shape[1]
        r, k = len(rows), self.masks.shape[1]
        age = stamp[:, None] - self.stamps[rows]                       # [R, K]
        live = (age >= 0) & (age < self.window)
        masks = self.masks[rows] & live[..., None]                      # [R, K, 187]
        old, new = self.poses[rows].reshape(r * k, 3), pose_xyyaw.repeat_interleave(k, 0)
        backward = warp_memory(masks.reshape(r * k, 187, 1).float(), old, new, mode="nearest").reshape(r, k, 187) > .5
        return (backward | splat_cells(masks.reshape(r * k, 187), old, new).reshape(r, k, 187)).any(1)


def splat_cells(cells, previous_pose, current_pose):
    """[B,187] bool cells of the grid at ``previous_pose`` -> [B,187] bool, each cell centre moved to the grid at
    ``current_pose`` and written to its nearest cell (dropped when it leaves the grid). Inverse of the
    ``warp_memory`` lookup: there a new cell p reads the old grid at R(angle) p + t; here old q lands at
    R(-angle) (q - t)."""
    y, x = torch.meshgrid(torch.linspace(-.5, .5, 11, device=cells.device),
                          torch.linspace(-.8, .8, 17, device=cells.device), indexing='ij')
    x, y = x.flatten()[None], y.flatten()[None]
    angle = (current_pose[:, 2] - previous_pose[:, 2])[:, None]
    delta = current_pose[:, :2] - previous_pose[:, :2]
    c, s = previous_pose[:, 2].cos(), previous_pose[:, 2].sin()
    tx = (c * delta[:, 0] + s * delta[:, 1])[:, None]
    ty = (-s * delta[:, 0] + c * delta[:, 1])[:, None]
    qx, qy = x - tx, y - ty
    ca, sa = angle.cos(), angle.sin()
    col = torch.round((ca * qx + sa * qy + .8) / .1)
    row = torch.round((-sa * qx + ca * qy + .5) / .1)
    keep = cells & (col >= 0) & (col <= 16) & (row >= 0) & (row <= 10)
    b = torch.arange(cells.shape[0], device=cells.device)[:, None].expand_as(cells)
    out = torch.zeros_like(cells)
    out[b[keep], (row * 17 + col)[keep].long()] = True
    return out


def warp_memory(memory, previous_pose, current_pose, mode="bilinear"):
    """B,187,C yaw-grid memory. Pose=(world x,y,yaw); new cells query old grid. ``mode`` as in grid_sample."""
    y, x = torch.meshgrid(torch.linspace(-.5, .5, 11, device=memory.device),
                          torch.linspace(-.8, .8, 17, device=memory.device), indexing='ij')
    angle = current_pose[:, 2] - previous_pose[:, 2]
    delta = current_pose[:, :2] - previous_pose[:, :2]
    c, s = previous_pose[:, 2].cos(), previous_pose[:, 2].sin()
    tx = c * delta[:, 0] + s * delta[:, 1]
    ty = -s * delta[:, 0] + c * delta[:, 1]
    ca, sa = angle.cos()[:, None, None], angle.sin()[:, None, None]
    gx = (ca*x - sa*y + tx[:, None, None]) / .8
    gy = (sa*x + ca*y + ty[:, None, None]) / .5
    grid = torch.stack((gx, gy), -1)
    old = memory.transpose(1, 2).reshape(-1, memory.shape[-1], 11, 17)
    return F.grid_sample(old, grid, mode=mode, align_corners=True).flatten(2).transpose(1, 2)
