"""Synchronized clean camera snapshots and camera-visible height observations."""

import torch
from isaaclab.managers import ManagerTermBase

from gd_lab.core.camera_contract import CAMERA_NAMES, load_camera_contract
from gd_lab.core.camera_geometry import calibrated_resample, camera_visible_points, mounted_camera_world_poses
from gd_lab.core.camera_timing import camera_period_steps, camera_refresh_mask


def canonical_camera_snapshot(scene, contract):
    """Capture depth + IR proxy and the exact poses used for those images."""
    depths, frames, positions, rotations, intrinsics = [], [], [], [], []
    lo, hi = contract.depth_clip
    robot = scene["robot"]
    trunk = robot.body_names.index("trunk")
    camera_pos, camera_quat = mounted_camera_world_poses(
        robot.data.body_pos_w[:, trunk], robot.data.body_quat_w[:, trunk], contract
    )
    for camera_index, name in enumerate(CAMERA_NAMES):
        sensor = scene[name]
        data = sensor.data
        target_k = data.intrinsic_matrices.clone()
        target_k[:, 0, 0], target_k[:, 1, 1] = contract.fx, contract.fy
        target_k[:, 0, 2], target_k[:, 1, 2] = contract.width / 2, contract.height / 2
        shape = (contract.height, contract.width)
        # Keep invalid depth as invalid for visibility; map it only in student input.
        _validate_sensor_config(sensor, name, camera_index, contract)
        raw = data.output["distance_to_image_plane"]
        # grid_sample on inf can produce NaN even at valid neighbors. A far
        # sentinel is conservative: it can never validate an in-range point.
        safe = torch.where(torch.isfinite(raw) & (raw > 0), raw, hi)
        depth = calibrated_resample(safe, data.intrinsic_matrices, target_k, shape)[..., 0]
        rgb = calibrated_resample(data.output["rgb"][..., :3], data.intrinsic_matrices, target_k, shape, "bilinear")
        ir = (rgb[..., 0]*0.299 + rgb[..., 1]*0.587 + rgb[..., 2]*0.114) / 255
        depths.append(depth)
        frames.append(torch.stack(((depth.clamp(lo, hi)-lo)/(hi-lo), ir.clamp(0, 1)), 1))
        positions.append(camera_pos[:, camera_index])
        rotations.append(camera_quat[:, camera_index])
        intrinsics.append(target_k)
    return tuple(torch.stack(items, 1) for items in (frames, depths, positions, rotations, intrinsics))

def _validate_sensor_config(sensor, name, camera_index, contract):
    """Check actual TiledCameraCfg mount and optics against the teacher contract."""
    cfg = getattr(sensor, "cfg", None)
    offset, spawn = getattr(cfg, "offset", None), getattr(cfg, "spawn", None)
    if offset is None or spawn is None:
        raise RuntimeError(f"{name} has no inspectable camera mount/optics config")
    if getattr(offset, "convention", None) != "opengl":
        raise RuntimeError(f"{name} mount quaternion convention is not OpenGL")
    expected_pos = torch.as_tensor(contract.positions[camera_index], dtype=torch.float64)
    expected_quat = torch.as_tensor(contract.quaternions_opengl[camera_index], dtype=torch.float64)
    actual_pos = torch.as_tensor(offset.pos, dtype=torch.float64)
    actual_quat = torch.as_tensor(offset.rot, dtype=torch.float64)
    actual_quat = actual_quat / actual_quat.norm().clamp_min(1e-12)
    expected_quat = expected_quat / expected_quat.norm().clamp_min(1e-12)
    if actual_pos.shape != expected_pos.shape or not torch.allclose(actual_pos, expected_pos, atol=1e-6, rtol=0):
        raise RuntimeError(f"{name} configured mount position differs from camera contract")
    if torch.abs(torch.dot(actual_quat, expected_quat)) < 1 - 1e-6:
        raise RuntimeError(f"{name} configured mount rotation differs from camera contract")
    if tuple(spawn.clipping_range) != tuple(contract.depth_clip) or (int(cfg.width), int(cfg.height)) != (contract.width, contract.render_height):
        raise RuntimeError(f"{name} configured clipping or resolution differs from camera contract")
    raw_k = sensor.data.intrinsic_matrices
    render_height, render_width = sensor.data.output["distance_to_image_plane"].shape[-3:-1]
    focal_px = contract.focal_m * render_width / contract.sensor_size_m[0]
    expected_k = raw_k.new_tensor([[focal_px, 0, render_width/2], [0, focal_px, render_height/2], [0, 0, 1]])
    if not torch.allclose(raw_k, expected_k.expand_as(raw_k), atol=1e-3, rtol=1e-4):
        raise RuntimeError(f"{name} renderer intrinsics disagree with configured camera optics")
    if abs(float(spawn.focal_length)/1000-contract.focal_m) > 1e-7 or abs(float(spawn.horizontal_aperture)/1000-contract.sensor_size_m[0]) > 1e-7:
        raise RuntimeError(f"{name} configured optical calibration differs from camera contract")


class CameraVisibleTerrain(ManagerTermBase):
    """[masked height(187), validity(187)], held at the camera/student rate.

    Critic scans are untouched. Snapshot buffers also provide the student
    with the SAME clean images as the target, before student-only corruption.
    Reset environments acquire a fresh snapshot, independently of other rows.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contract = load_camera_contract(env.cfg.camera_profile)
        self.period_steps = camera_period_steps(
            env.cfg.decimation * env.cfg.sim.dt,
            self.contract.policy_dt,
            self.contract.period_steps,
        )
        n = env.num_envs
        self.last_step = torch.full((n,), -100, device=env.device, dtype=torch.long)
        self.observation = torch.zeros(n, 374, device=env.device)
        self.buffers = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -100
        self.observation[ids] = 0

    def __call__(self, env):
        step = env.common_step_counter
        capture_step = getattr(env, "_vrl_camera_capture_step", None)
        if capture_step is None:
            refresh = camera_refresh_mask(self.last_step, step, self.period_steps)
        else:
            # Student transport simulation schedules fresh captures explicitly.
            # Resets still acquire their own snapshot immediately.
            refresh = (self.last_step < 0) | ((step == capture_step) & (self.last_step != step))
        if not refresh.any():
            return self.observation.clone()
        snapshot = canonical_camera_snapshot(env.scene, self.contract)
        _, depths, positions, rotations, intrinsics = snapshot
        scan = env.scene["height_scanner"].data
        points = scan.ray_hits_w
        if points.shape[1] != 187:
            raise ValueError("VRL v2 requires the configured 11x17 height scan")
        visible = torch.zeros(points.shape[:2], dtype=torch.bool, device=env.device)
        for camera in range(4):
            visible |= camera_visible_points(points, positions[:, camera], rotations[:, camera],
                                             intrinsics[:, camera], depths[:, camera], self.contract.depth_clip)
        # Same height offset/clip/scale as the critic, but no critic normalizer:
        # hidden terrain must not influence even the encoder normalization stats.
        height = (scan.pos_w[:, 2, None] - points[..., 2] - 0.5).clamp(-1, 1) * 5
        height = torch.where(visible & torch.isfinite(height), height, 0)
        self.observation[refresh] = torch.cat((height, visible.float()), -1)[refresh]
        self.last_step[refresh] = step
        if capture_step is not None and step == capture_step:
            env._vrl_teacher_terrain_capture_steps = self.last_step.clone()
            env._vrl_teacher_terrain_snapshot = self.observation.clone()
            env._vrl_teacher_terrain_contract = self.contract.manifest()
        if self.buffers is None:
            self.buffers = tuple(x.clone() for x in snapshot)
        else:
            for buffer, current in zip(self.buffers, snapshot, strict=True):
                buffer[refresh] = current[refresh]
        env._vrl_camera_snapshot = self.buffers
        env._vrl_camera_snapshot_steps = self.last_step.clone()
        return self.observation.clone()
