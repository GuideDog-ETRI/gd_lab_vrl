"""Strict capture-time and calibration checks for camera-to-teacher distillation."""

import torch

from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses

_CAMERA_FIELDS = ("profile", "positions", "quaternions_opengl", "sensor_size_m", "focal_m",
                  "width", "height", "source_resolution", "depth_clip")


def _freeze(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def camera_signature(contract):
    get = contract.get if isinstance(contract, dict) else lambda key: getattr(contract, key)
    return tuple(_freeze(get(key)) for key in _CAMERA_FIELDS)


def validate_teacher_camera_capture(env, obs, camera_contract):
    """Fail unless rendered image, teacher Ray map, calibration and pose share a capture step."""
    step = env.common_step_counter
    snapshot = getattr(env, "_vrl_camera_snapshot", None)
    camera_steps = getattr(env, "_vrl_camera_snapshot_steps", None)
    ray_steps = getattr(env, "_vrl_teacher_terrain_capture_steps", None)
    ray_snapshot = getattr(env, "_vrl_teacher_terrain_snapshot", None)
    ray_contract = getattr(env, "_vrl_teacher_terrain_contract", None)
    # Backward-compatible names used by the isolated GAST snapshot.
    ray_steps = ray_steps if ray_steps is not None else getattr(env, "_vrl_teacher_ray_capture_steps", None)
    ray_snapshot = ray_snapshot if ray_snapshot is not None else getattr(env, "_vrl_teacher_ray_snapshot", None)
    ray_contract = ray_contract if ray_contract is not None else getattr(env, "_vrl_teacher_ray_contract", None)
    if snapshot is None or len(snapshot) != 5 or camera_steps is None or not (camera_steps == step).all():
        raise RuntimeError("Student camera frame is stale or missing at the scheduled teacher capture step")
    if ray_steps is None or not (ray_steps == step).all() or ray_snapshot is None:
        raise RuntimeError("Teacher Ray height/visibility target is stale or missing at the camera capture step")
    expected_contract = load_camera_contract(env.cfg.camera_profile)
    if camera_signature(expected_contract) != camera_signature(camera_contract):
        raise RuntimeError("Student camera profile/calibration differs from the environment camera contract")
    if ray_contract is None or camera_signature(ray_contract) != camera_signature(expected_contract):
        raise RuntimeError("Teacher Ray camera contract is missing or differs from the student calibration")
    terrain = obs.get("terrain")
    if terrain is None or terrain.shape[-1] != 374:
        raise RuntimeError("Teacher Ray target must contain 187 masked heights and 187 visibility labels")
    if ray_snapshot.shape != terrain.shape or not torch.equal(terrain, ray_snapshot):
        raise RuntimeError("Teacher terrain observation is not the synchronized Ray capture target")

    frames, depths, positions, rotations, intrinsics = snapshot
    n, cameras = frames.shape[:2]
    if frames.shape[1:] != (4, 2, camera_contract.height, camera_contract.width):
        raise RuntimeError(f"Rendered camera frame shape disagrees with calibration: {tuple(frames.shape)}")
    if depths.shape[1:] != (4, camera_contract.height, camera_contract.width):
        raise RuntimeError("Rendered depth shape disagrees with calibration")
    expected_k = intrinsics.new_tensor([[camera_contract.fx, 0, camera_contract.width / 2],
                                        [0, camera_contract.fy, camera_contract.height / 2], [0, 0, 1]])
    if not torch.allclose(intrinsics, expected_k.expand_as(intrinsics), atol=1e-4, rtol=1e-5):
        raise RuntimeError("Rendered camera intrinsics differ from the teacher/student calibration")
    robot = env.scene["robot"]
    trunk = robot.body_names.index("trunk")
    expected_pos, expected_rot = mounted_camera_world_poses(
        robot.data.body_pos_w[:, trunk], robot.data.body_quat_w[:, trunk], camera_contract)
    if not torch.allclose(positions, expected_pos, atol=1e-4, rtol=1e-5):
        raise RuntimeError("Student camera mount positions differ from the teacher target contract")
    if not torch.allclose(rotations, expected_rot, atol=1e-4, rtol=1e-5):
        raise RuntimeError("Student camera mount orientations differ from the teacher target contract")
    return terrain
