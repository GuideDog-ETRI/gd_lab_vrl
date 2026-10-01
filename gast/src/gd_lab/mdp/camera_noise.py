"""Student-only camera augmentation on canonical depth/IR tensors."""

import torch


def _uniform(bounds, shape, device):
    lo, hi = bounds
    return lo + torch.rand(shape, device=device) * (hi - lo)


def _patch_mask(n, cameras, h, w, probability, size_range, device):
    half = _uniform(size_range, (n, cameras, 1, 1), device) / 2
    cx = torch.rand(n, cameras, 1, 1, device=device)
    cy = torch.rand(n, cameras, 1, 1, device=device)
    xs = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w)
    ys = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1)
    present = torch.rand(n, cameras, 1, 1, device=device) < probability
    return (xs - cx).abs().le(half) & (ys - cy).abs().le(half) & present


def noisy_camera_tensor(frames, cfg, depth_clip=(0.15, 5.0)):
    """Apply generic D430-like noise to canonical ``[N,C,2,H,W]`` frames."""
    if frames.ndim != 5 or frames.shape[2] != 2:
        raise ValueError("frames must have shape [N,C,2,H,W]")
    n, cameras, _, h, w = frames.shape
    device = frames.device
    lo, hi = depth_clip
    shape = (n, cameras, 1, 1)
    depth = frames[:, :, 0] * (hi - lo) + lo
    ir = frames[:, :, 1]
    base_std = _uniform(cfg.depth_base_std_range, shape, device)
    scale = _uniform(cfg.depth_scale_range, shape, device)
    dropout = _uniform(cfg.dropout_prob_range, shape, device)
    ir_std = _uniform(cfg.ir_noise_std_range, shape, device)
    density = _uniform(cfg.ir_dot_density_range, shape, device)
    intensity = _uniform(cfg.ir_dot_intensity_range, shape, device)
    hotspot_probability = _uniform(cfg.hotspot_prob_range, shape, device)
    hotspot = _patch_mask(n, cameras, h, w, hotspot_probability, cfg.hotspot_size_range, device)
    hotspot_dropout = _uniform(cfg.hotspot_dropout_range, shape, device)
    hotspot_ir_std = _uniform(cfg.hotspot_ir_std_range, shape, device)
    dropout = torch.where(hotspot, hotspot_dropout, dropout)
    ir_std = torch.where(hotspot, hotspot_ir_std, ir_std)
    depth = depth + torch.randn_like(depth) * (base_std + scale * depth.square())
    depth = torch.where(torch.rand_like(depth) < dropout, torch.full_like(depth, hi), depth).clamp(lo, hi)
    dots = (torch.rand_like(ir) < density).to(ir.dtype)
    ir = ir + dots * intensity * (lo / depth).square()
    ir = (ir + torch.randn_like(ir) * ir_std).clamp(0, 1)
    return torch.stack(((depth - lo) / (hi - lo), ir), dim=2)


def augment_student_camera_frames(frames, snapshot, origins, gap_envs, noise_cfg, gap_ghost):
    """One noise switch controls generic noise AND gap ghosts; teacher stays clean."""
    if noise_cfg is None:
        return frames
    noisy = noisy_camera_tensor(frames, noise_cfg)
    return gap_ghost(noisy, snapshot, origins, gap_envs)
