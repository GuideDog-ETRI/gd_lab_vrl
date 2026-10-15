"""Simulator-independent terrain labels and ego-motion memory alignment."""
import torch
from torch.nn import functional as F


def targets(clean, teacher_terrain=None, seen=None):
    """Packed clean height(187), valid(187), semantic gap(187), gap-known(187).

    Height uses the existing scanner_z-ground_z-0.5 convention, scaled by 5.
    Gap labels come from known gap terrain geometry, never missing depth.
    ``seen`` [B,187] bool (optional): cells a camera saw recently (``SeenCells``). They are supervised with
    the clean height and gap labels as well, so terrain that just left the field of view (e.g. a gap under
    the body) still trains the memory. The visibility channel keeps the current teacher-visible target.
    """
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


def near_gap_rows(clean):
    """[B] bool: a known gap cell (semantic gap map, never missing depth) lies in the 11x17 grid around the body."""
    _, _, gap, known = clean.reshape(-1, 4, 11, 17).unbind(1)
    return ((gap > .5) & known.bool()).flatten(1).any(1)


def weighted_mean(values, weight):
    """Per-row values [B] averaged with row weights; equals values.mean() when every weight is 1."""
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def reconstruction_loss(pred, clean, teacher_terrain=None, return_components=False, row_weight=None, seen=None):
    """``row_weight`` [B] scales whole samples (e.g. near-gap emphasis); None keeps every row at 1.
    ``seen``: optional recently-seen cells, see ``targets``."""
    y, mask = targets(clean, teacher_terrain, seen)
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
    warps each one straight to the current pose with nearest sampling. One warp per snapshot, so sub-cell motion
    never accumulates rounding or interpolation loss (a repeatedly warped, thresholded grid lost a single cell
    after a 0.05 m step).
    """

    def __init__(self, num_envs, device, window_s, min_capture_period_s=0.07):
        if not (window_s > 0 and window_s == window_s and window_s < 60):
            raise ValueError("window_s must be finite and in (0, 60)")
        self.window = float(window_s)
        k = int(-(-window_s // min_capture_period_s)) + 2
        self.masks = torch.zeros(num_envs, k, 187, dtype=torch.bool, device=device)
        self.poses = torch.zeros(num_envs, k, 3, device=device)
        self.stamps = torch.full((num_envs, k), float("-inf"), device=device)
        self.next = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, done):
        self.stamps[done] = float("-inf")
        self.masks[done] = False
        self.next[done] = 0

    @torch.no_grad()
    def update(self, rows, visible, pose_xyyaw, stamp, received=None):
        """visible [R,187] bool (cells delivered now); ``received`` [R] bool: False for frames the student did not
        get (augmentation drop or a hidden camera) -> nothing new is recorded for that row. Returns the seen mask
        [R,187] over the window (snapshots still inside it, warped to ``pose_xyyaw``)."""
        rows = torch.as_tensor(rows, device=self.masks.device)
        if received is None:
            received = torch.ones(len(rows), dtype=torch.bool, device=rows.device)
        write = rows[received]
        if len(write):
            slot = self.next[write]
            self.masks[write, slot] = visible[received]
            self.poses[write, slot] = pose_xyyaw[received]
            self.stamps[write, slot] = stamp[received]
            self.next[write] = (slot + 1) % self.masks.shape[1]
        r, k = len(rows), self.masks.shape[1]
        age = stamp[:, None] - self.stamps[rows]                       # [R, K]
        live = (age >= 0) & (age < self.window)
        warped = warp_memory(self.masks[rows].reshape(r * k, 187, 1).float(), self.poses[rows].reshape(r * k, 3),
                             pose_xyyaw.repeat_interleave(k, 0), mode="nearest").reshape(r, k, 187) > .5
        return (warped & live[..., None]).any(1)


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
