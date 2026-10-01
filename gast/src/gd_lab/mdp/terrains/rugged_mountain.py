"""Irregular multi-directional rugged terrain (heightfield), curriculum-scaled.

Unlike IsaacLab's stock ``random_uniform_terrain`` (whose docstring says
"the difficulty parameter is ignored" -- confirmed in hf_terrains.py), this
one scales its amplitude with ``difficulty`` like the rest of this
repo's terrain set, so terrain_levels actually ramps it instead of handing
every robot the same fixed roughness from level 0.

2026-09-13: added to test/train recovery from a hand-built Mujoco course
where the deployed policy got stuck on unbounded rugged terrain (no such
terrain existed in training at all before this). ``height_range[1]`` is
capped at 0.35m -- the kinematic estimate for this robot (leg segments
0.33m+0.33m, nominal stance 0.52m) beyond which a single-step clearance
starts requiring the leg to fold past what training has ever validated
(see the curriculum discussion this generator was added for).
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def rugged_mountain_terrain(difficulty: float, cfg: RuggedMountainTerrainCfg) -> np.ndarray:
    """Smoothly-interpolated random terrain with a flat spawn platform, amplitude ramped by difficulty.

    Same downsample-then-interpolate approach as ``random_uniform_terrain``
    (avoids single-pixel spikes a foot could catch on by accident rather than
    by the terrain's actual difficulty), but the amplitude is
    ``height_range[0] + difficulty * (height_range[1] - height_range[0])``
    instead of a fixed range, and a flat ``platform_width`` circle is carved
    at the spawn point so the robot doesn't start mid-slope.

    2026-09-15: blended with a directional ridge (a diagonal sinusoid) so
    crossing this terrain means climbing/descending a slope at an angle, not
    just stepping over isotropic random bumps -- the same technique used on
    the hand-built Mujoco test course's mountain heightmap, ported here so
    training actually covers what that course exercises.
    """
    amplitude = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    downsampled_scale = cfg.downsampled_scale if cfg.downsampled_scale is not None else cfg.horizontal_scale
    width_downsampled = max(2, int(cfg.size[0] / downsampled_scale))
    length_downsampled = max(2, int(cfg.size[1] / downsampled_scale))

    height_min = int(-amplitude / cfg.vertical_scale)
    height_max = int(amplitude / cfg.vertical_scale)
    height_range = np.arange(height_min, height_max + 1)
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    x = np.linspace(0, cfg.size[0], width_downsampled)
    y = np.linspace(0, cfg.size[1], length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    x_up = np.linspace(0, cfg.size[0], width_pixels)
    y_up = np.linspace(0, cfg.size[1], length_pixels)
    noise_raw = func(x_up, y_up) * cfg.vertical_scale  # pixel-height units -> meters

    theta = np.deg2rad(cfg.ridge_angle_deg)
    xx_m, yy_m = np.meshgrid(x_up, y_up, indexing="ij")
    phase = 2 * np.pi * (xx_m * np.cos(theta) + yy_m * np.sin(theta)) / cfg.ridge_wavelength
    ridge_raw = np.sin(phase) * amplitude

    combined = cfg.ridge_weight * ridge_raw + (1.0 - cfg.ridge_weight) * noise_raw
    # blending two independent +-amplitude signals doesn't preserve the
    # peak-to-peak range, so renormalize back to it
    max_abs = np.abs(combined).max()
    if max_abs > 1e-9:
        combined *= amplitude / max_abs
    hf_raw = combined / cfg.vertical_scale  # back to pixel-height units

    # flat platform at spawn (center) so the robot doesn't start mid-slope
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale / 2)
    cx, cy = width_pixels // 2, length_pixels // 2
    xx, yy = np.meshgrid(np.arange(width_pixels), np.arange(length_pixels), indexing="ij")
    in_platform = (np.abs(xx - cx) < platform_pixels) & (np.abs(yy - cy) < platform_pixels)
    hf_raw[in_platform] = 0.0

    return np.rint(hf_raw).astype(np.int16)


@configclass
class RuggedMountainTerrainCfg(HfTerrainBaseCfg):
    """Irregular rugged terrain, amplitude ramped by difficulty."""

    function = rugged_mountain_terrain

    height_range: tuple[float, float] = MISSING
    """(min, max) local elevation amplitude in m, interpolated by difficulty.

    Cap the max at or below 0.35m -- see the module docstring for why."""

    downsampled_scale: float | None = None
    """Coarser grid the random heights are sampled on before interpolation (m).
    Bigger than horizontal_scale -> smoother, more mountain-like bumps instead
    of single-pixel noise. Defaults to horizontal_scale if unset."""

    platform_width: float = 1.5
    """Flat circular spawn platform diameter (m); no roughness inside it."""

    ridge_wavelength: float = 1.6
    """Distance (m) between ridge crests. Matches the Mujoco test-course heightmap."""

    ridge_angle_deg: float = 35.0
    """Angle (deg) of the ridge line off the walking (+x) direction."""

    ridge_weight: float = 0.55
    """Blend factor between the directional ridge (1.0) and isotropic noise (0.0)."""
