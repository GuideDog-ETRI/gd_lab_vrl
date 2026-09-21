"""Vision rough-terrain task: the blind task plus the 4 real belly cameras.

This module deliberately holds **only** the camera additions. Scene, terrain,
rewards, curriculum and the play variant come from ``vrl_teacher``, which
extends ``blind_rough``. Thus base changes (terrain families, gait cadence,
payload events, PPO settings) reaches the vision task for free. An earlier
revision copied ``blind_rough`` wholesale instead, and the two drifted apart
badly enough that the copy had to be thrown away -- do not reintroduce that.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
import torch
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from gd_lab.tasks.blind_rough import BlindRoughSceneCfg
from gd_lab.tasks.vrl_teacher import VrlTeacherEnvCfg, apply_vrl_play

# 2026-09-15: the real camera mounts are not a guess -- they're already spec'd
# in the robot's own URDF (assets/robots/rbq10/urdf/rbq10_simple.urdf, "Depth
# Camera" section), 4x Intel D430 (IR+depth only, no RGB imager -- see
# _belly_camera's docstring), matched 1:1 to the Mujoco deploy model's
# BT0-3 belly cameras (rbq.xml sensor_0..3): front_depth_camera0/1 = BT0/BT1,
# hind_depth_camera2/3 = BT2/BT3.
#
# RBQ10_CFG spawns with merge_fixed_joints=True, but that only merges the
# *physics* bodies -- the 4 camera links DO survive as their own Xform prims
# under ".../Robot/trunk/<name>" in the compiled USD. They're plain mount-point
# Xforms though, not UsdGeom.Camera prims, so a sensor cannot attach there
# directly (Camera._initialize_impl requires cam_prim.IsA(UsdGeom.Camera)).
#
# Fix: spawn the camera prim as a *child* of each URDF mount link instead of
# reusing the link's own name. The URDF joint's <origin xyz rpy> already places
# that mount link correctly in trunk's frame, so the child camera needs zero
# additional offset; offset=identity + convention="ros" performs the ROS-style
# (forward=+Z, up=-Y) -> USD camera (forward=-Z, up=+Y) remap on its own.
_BELLY_CAMERA_MOUNTS = (
    "front_depth_camera0",  # BT0
    "front_depth_camera1",  # BT1
    "hind_depth_camera2",  # BT2
    "hind_depth_camera3",  # BT3
)

_BELLY_CAMERA_DEPTH_CLIP = (0.15, 5.0)  # must match _belly_camera's clipping_range


def _belly_camera(mount: str) -> TiledCameraCfg:
    """One of the 4 real D430 belly cameras, depth + rgb.

    TiledCameraCfg (not plain CameraCfg) because it batch-renders across all
    parallel envs in one pass -- the only camera sensor that scales to
    thousands of envs.

    ``data_types`` carries both depth (``distance_to_image_plane``) and
    ``rgb``: the real D430 has no RGB imager at all (2x IR stereo imagers +
    IR projector only), and gd_rbq10_deploy's rbq_mujoco derives each BT
    camera's IR channel as ``cv::cvtColor(color, gray, COLOR_BGR2GRAY)``.
    Rendering rgb here and converting to grayscale downstream reproduces that
    exact convention instead of inventing a different IR proxy.

    FOV/clip/aspect (corrected against Intel's D400 datasheet AND the vendor's
    own ``D430_pattern_cfg`` calibration constants): HD 16:9 depth FOV is
    H87/V58 deg, so this renders 80x45 (16:9) to match the real D430's streamed
    aspect (640x360). The apertures are set explicitly rather than left to the
    width/height auto-ratio so the H87/V58 pair lands exactly at
    focal_length=12.0. Near clip 0.15 m is D430's real Min-Z at 640x360.
    """
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/" + mount + "/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=22.7751,
            vertical_aperture=13.3034,
            clipping_range=(0.15, 5.0),
        ),
        width=80,
        height=45,
    )


def belly_camera_frames(scene, depth_clip: tuple[float, float] = _BELLY_CAMERA_DEPTH_CLIP) -> torch.Tensor:
    """(num_envs, 4, 2, H, W) depth+IR tensor from the 4 belly cameras.

    Depth: ``distance_to_image_plane`` clamped/normalized to [0, 1] over
    ``depth_clip``; non-finite (no-hit) pixels map to 1.0 ("far/clipped",
    matching a real depth sensor past its max range).
    IR: rgb -> grayscale luminance / 255, matching the deploy stack's own
    ``cv::cvtColor(color, gray, COLOR_BGR2GRAY)`` convention.
    """
    lo, hi = depth_clip
    frames = []
    for mount in _BELLY_CAMERA_MOUNTS:
        cam = scene[mount]
        depth = cam.data.output["distance_to_image_plane"][..., 0]
        depth = torch.nan_to_num(depth, nan=hi, posinf=hi, neginf=lo)
        depth_norm = ((depth.clamp(lo, hi) - lo) / (hi - lo)).unsqueeze(1)
        rgb = cam.data.output["rgb"][..., :3].float()
        ir = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        frames.append(torch.cat((depth_norm, ir.unsqueeze(1)), dim=1))
    return torch.stack(frames, dim=1)


@dataclass
class CameraNoiseCfg:
    """Depth/IR sensor-noise domain-randomization ranges for the student.

    ``belly_camera_frames`` renders a noiseless, ideal-pinhole image -- as
    clean as the teacher's own privileged height_scan. A student trained only
    on that has never seen what a real D430 looks like (range-dependent depth
    noise, dropout near reflective/grazing surfaces, IR sensor noise), which is
    a real sim2real gap. ``noisy_camera_frames`` injects that.

    Each call re-samples fresh severities per env, so the student becomes
    robust to "there is noise" rather than tuned to one specific profile.
    """

    depth_base_std_range: tuple[float, float] = (0.0005, 0.003)  # m, noise floor at d=0
    depth_scale_range: tuple[float, float] = (0.0025, 0.005)  # m, coefficient on d^2
    dropout_prob_range: tuple[float, float] = (0.0, 0.10)  # fraction of pixels -> "no return"
    ir_noise_std_range: tuple[float, float] = (0.01, 0.05)  # on the [0,1]-normalized IR

    # The real D430 has an active IR pattern projector (850nm VCSEL) that
    # projects a static dot pattern to add texture for stereo matching, so a
    # real IR frame is not a plain grayscale render -- the speckle dominates
    # flat/textureless surfaces (floors, risers). This is a coarse per-pixel
    # approximation: a fresh random dot mask each call (not spatially fixed
    # across steps as the real emitter would be), brightened at dot locations
    # and attenuated by inverse-square falloff with depth.
    ir_dot_density_range: tuple[float, float] = (0.02, 0.08)  # fraction of pixels lit as a dot
    ir_dot_intensity_range: tuple[float, float] = (0.3, 0.7)  # peak added brightness at depth_clip[0]

    # "Hotspot": a localized patch of much-worse noise, approximating a
    # reflective surface (the aluminum threshold plate at a platform-train gap,
    # a polished stair nosing) where the IR pattern scatters instead of
    # returning. That is exactly where gap/step detection matters most, so
    # uniform-only noise never teaches the student that the most
    # safety-critical patch can also be the least trustworthy one.
    hotspot_prob_range: tuple[float, float] = (0.0, 0.3)  # chance a camera/call has a hotspot
    hotspot_size_range: tuple[float, float] = (0.15, 0.4)  # side length, fraction of image size
    hotspot_dropout_range: tuple[float, float] = (0.5, 0.9)  # dropout prob *inside* the hotspot
    hotspot_ir_std_range: tuple[float, float] = (0.2, 0.5)  # IR noise std *inside* the hotspot


def _sample_range(range_: tuple[float, float], shape: tuple[int, ...], device) -> torch.Tensor:
    lo, hi = range_
    return lo + torch.rand(shape, device=device) * (hi - lo)


def _hotspot_mask(
    num_envs: int, height: int, width: int, prob: torch.Tensor, size_range: tuple[float, float], device
) -> torch.Tensor:
    """(num_envs, H, W) bool mask: at most one random square patch per env,
    present with per-env probability ``prob`` (shape (num_envs,1,1))."""
    half_size = _sample_range(size_range, (num_envs, 1, 1), device) / 2
    cx = torch.rand(num_envs, 1, 1, device=device)
    cy = torch.rand(num_envs, 1, 1, device=device)
    xs = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, width)
    ys = torch.linspace(0.0, 1.0, height, device=device).view(1, height, 1)
    in_patch = (xs - cx).abs().le(half_size) & (ys - cy).abs().le(half_size)
    present = torch.rand(num_envs, 1, 1, device=device) < prob
    return in_patch & present


def noisy_camera_frames(
    scene, cfg: CameraNoiseCfg, depth_clip: tuple[float, float] = _BELLY_CAMERA_DEPTH_CLIP
) -> torch.Tensor:
    """Like ``belly_camera_frames`` but with depth/IR sensor noise injected.

    Noise is added in metric/raw space *before* the clamp+normalize, so a noisy
    reading that overshoots ``depth_clip`` gets clipped exactly like a real
    out-of-range reading would.
    """
    lo, hi = depth_clip
    sample = scene[_BELLY_CAMERA_MOUNTS[0]].data.output["distance_to_image_plane"]
    device, num_envs = sample.device, sample.shape[0]

    base_std = _sample_range(cfg.depth_base_std_range, (num_envs, 1, 1), device)
    scale = _sample_range(cfg.depth_scale_range, (num_envs, 1, 1), device)
    dropout_p = _sample_range(cfg.dropout_prob_range, (num_envs, 1, 1), device)
    ir_std = _sample_range(cfg.ir_noise_std_range, (num_envs, 1, 1), device)
    dot_density = _sample_range(cfg.ir_dot_density_range, (num_envs, 1, 1), device)
    dot_intensity = _sample_range(cfg.ir_dot_intensity_range, (num_envs, 1, 1), device)
    hotspot_prob = _sample_range(cfg.hotspot_prob_range, (num_envs, 1, 1), device)
    hotspot_dropout = _sample_range(cfg.hotspot_dropout_range, (num_envs, 1, 1), device)
    hotspot_ir_std = _sample_range(cfg.hotspot_ir_std_range, (num_envs, 1, 1), device)

    frames = []
    for mount in _BELLY_CAMERA_MOUNTS:
        cam = scene[mount]
        depth = cam.data.output["distance_to_image_plane"][..., 0]
        height, width = depth.shape[-2:]
        depth = torch.nan_to_num(depth, nan=hi, posinf=hi, neginf=lo)

        # One random patch per camera per call, independent of the uniform
        # per-pixel terms below, so a frame can have a generally clean sensor
        # AND one badly-corrupted patch.
        hotspot = _hotspot_mask(num_envs, height, width, hotspot_prob, cfg.hotspot_size_range, device)
        effective_dropout_p = torch.where(hotspot, hotspot_dropout, dropout_p)
        effective_ir_std = torch.where(hotspot, hotspot_ir_std, ir_std)

        # Range-dependent noise (real stereo-depth characteristic) + dropout
        # ("no return" -> far clip, same as a real sensor past its usable range).
        noise_std = base_std + scale * depth.pow(2)
        depth = depth + torch.randn_like(depth) * noise_std
        dropout_mask = torch.rand_like(depth) < effective_dropout_p
        depth = torch.where(dropout_mask, torch.full_like(depth, hi), depth)
        depth_clamped = depth.clamp(lo, hi)
        depth_norm = ((depth_clamped - lo) / (hi - lo)).unsqueeze(1)

        rgb = cam.data.output["rgb"][..., :3].float()
        ir = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0

        # Inverse-square falloff so the projected dots fade on far surfaces
        # like the real emitter would; added before the sensor-noise term so
        # the dots themselves also pick up some of that noise.
        dot_mask = (torch.rand_like(ir) < dot_density).float()
        dot_falloff = (lo / depth_clamped).pow(2)
        ir = ir + dot_mask * dot_intensity * dot_falloff

        ir = (ir + torch.randn_like(ir) * effective_ir_std).clamp(0.0, 1.0)

        frames.append(torch.cat((depth_norm, ir.unsqueeze(1)), dim=1))
    return torch.stack(frames, dim=1)


@configclass
class VisionRoughSceneCfg(BlindRoughSceneCfg):
    """``BlindRoughSceneCfg`` + the 4 real URDF-positioned belly cameras.

    Nothing in ``DreamwaqObservationsCfg``/PPO reads these: they are scene
    sensors only, harvested directly by ``belly_camera_frames``.
    """

    front_depth_camera0 = _belly_camera("front_depth_camera0")
    front_depth_camera1 = _belly_camera("front_depth_camera1")
    hind_depth_camera2 = _belly_camera("hind_depth_camera2")
    hind_depth_camera3 = _belly_camera("hind_depth_camera3")


@configclass
class VisionRoughEnvCfg(VrlTeacherEnvCfg):
    """``VrlTeacherEnvCfg`` + the 4 belly cameras.

    Policy and critic observation terms are unchanged from the blind task, so a
    observation widths are unchanged. A blind actor checkpoint still needs
    explicit migration: the VRL actor additionally consumes 32 terrain features.
    """

    scene: VisionRoughSceneCfg = VisionRoughSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        # Rendered slower than control, not every policy step: matches the
        # realistic onboard perception rate the student will actually see.
        policy_dt = self.decimation * self.sim.dt
        for mount in _BELLY_CAMERA_MOUNTS:
            getattr(self.scene, mount).update_period = 4 * policy_dt


@configclass
class VisionRoughEnvCfg_PLAY(VisionRoughEnvCfg):
    """``VisionRoughEnvCfg`` + the fixed non-curriculum play settings.

    For closed-loop student validation: small scene, fixed moderate difficulty,
    no domain-randomization noise, so checkpoints compare under identical
    conditions instead of whatever the curriculum happened to be at.
    """

    def __post_init__(self):
        super().__post_init__()
        apply_vrl_play(self)
