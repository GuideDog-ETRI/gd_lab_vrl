"""Stage 2-A terrain group with rendering-free camera visibility and blackouts.

Same [masked height(187), visibility(187)] layout, scale and capture clock as
CameraVisibleTerrain; visibility comes from raycast_visibility instead of
rendered depth, so no camera needs to exist in the scene.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase

from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses
from gd_lab.core.camera_timing import camera_period_steps, camera_refresh_mask
from gd_lab.core.camera_transport import CameraTransport, CameraTransportConfig, delivery_mask

from .raycast_visibility import contract_intrinsic, raycast_visible_points, warp_mesh_cast
from .terrain_dropout import SCAN_CELLS, _schedule

LEGS = ("FL", "FR", "RL", "RR")
# Capsule radii: thigh (hip->knee) and calf (knee->foot), from the RBQ10 URDF geometry.
THIGH_RADIUS, CALF_RADIUS = 0.05, 0.03


def leg_capsules(robot):
    """(seg_a[N,8,3], seg_b[N,8,3], radius[8]) for the eight leg segments."""
    names = robot.body_names
    missing = [f"{leg}_{part}" for leg in LEGS for part in ("thigh", "calf", "foot") if f"{leg}_{part}" not in names]
    if missing:
        raise ValueError(f"robot is missing leg bodies for occlusion: {missing}")
    pos = robot.data.body_pos_w
    index = lambda leg, part: names.index(f"{leg}_{part}")  # noqa: E731
    starts = [index(leg, part) for leg in LEGS for part in ("thigh", "calf")]
    ends = [index(leg, part) for leg in LEGS for part in ("calf", "foot")]
    radius = pos.new_tensor([THIGH_RADIUS, CALF_RADIUS] * len(LEGS))
    return pos[:, starts], pos[:, ends], radius


def raycast_visible_mask(env, contract, intrinsic, cast, enhanced=True) -> torch.Tensor:
    """N,187 union over the four mounted cameras of raycast-visible scan cells."""
    robot = env.scene["robot"]
    occluders = leg_capsules(robot)
    trunk = robot.body_names.index("trunk")
    body_occluders = None
    if enhanced:
        # URDF rbq10_simple: trunk visual/collision box .8 x .20 x .12 m;
        # foot collision spheres .03 m (conservative proxy, not visual mesh).
        from isaaclab.utils.math import matrix_from_quat
        foot_ids = [robot.body_names.index(f'{leg}_foot') for leg in LEGS]
        body_occluders = (robot.data.body_pos_w[:,trunk],
                          matrix_from_quat(robot.data.body_quat_w[:,trunk]),
                          robot.data.body_pos_w.new_tensor([.4,.10,.06]),
                          robot.data.body_pos_w[:,foot_ids], .03)
    positions, rotations = mounted_camera_world_poses(
        robot.data.body_pos_w[:, trunk], robot.data.body_quat_w[:, trunk], contract)
    points = env.scene["height_scanner"].data.ray_hits_w
    visible = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
    for camera in range(positions.shape[1]):
        visible |= raycast_visible_points(points, positions[:, camera], rotations[:, camera], intrinsic,
                                          (contract.height, contract.width), contract.depth_clip, cast,
                                          occluders=occluders, body_occluders=body_occluders,
                                          conservative=enhanced)
    return visible


class RaycastVisibleTerrainDropout(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contract = load_camera_contract(env.cfg.camera_profile)
        self.period_steps = camera_period_steps(
            env.cfg.decimation * env.cfg.sim.dt, self.contract.policy_dt, self.contract.period_steps)
        self.intrinsic = contract_intrinsic(self.contract, env.device)
        scanner = env.scene["height_scanner"]
        self.cast = warp_mesh_cast(scanner.meshes[scanner.cfg.mesh_prim_paths[0]])
        self.last_step = torch.full((env.num_envs,), -100, device=env.device, dtype=torch.long)
        self.observation = torch.zeros(env.num_envs, 2 * SCAN_CELLS, device=env.device)
        self.blackout = _schedule(cfg, env)
        self.blackout.reset()
        self.episodes = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.transport = CameraTransport(CameraTransportConfig(), env.step_dt, env.cfg.seed+17)
        self.timing_manifest = self.transport.config.manifest(env.step_dt)
        self.next_capture_step = 0
        self.last_call_step = -1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -100
        self.observation[ids] = 0
        self.blackout.reset(env_ids)
        if hasattr(self, 'episodes'):
            ids = slice(None) if env_ids is None else env_ids
            self.episodes[ids] += 1

    def __call__(self, env, start_prob=0.0, duration_steps=(50, 300), episode_prob=0.0) -> torch.Tensor:
        step = env.common_step_counter
        if step != self.last_call_step and step >= self.next_capture_step:
            scan = env.scene["height_scanner"].data
            if scan.ray_hits_w.shape[1] != SCAN_CELLS:
                raise ValueError("VRL v2 requires the configured 11x17 height scan")
            visible = raycast_visible_mask(env, self.contract, self.intrinsic, self.cast)
            height = (scan.pos_w[:, 2, None] - scan.ray_hits_w[..., 2] - 0.5).clamp(-1, 1) * 5
            height = torch.where(visible & torch.isfinite(height), height, 0)
            captured = torch.cat((height, visible.float()), -1)
            self.transport.capture(step, (captured, self.episodes.clone()))
            self.next_capture_step = self.transport.next_capture(step)
        for packet in self.transport.receive(step):
            captured, episodes = packet.payload
            valid = delivery_mask(packet.capture_step, episodes, self.episodes, self.last_step)
            self.observation[valid] = captured[valid]
            self.last_step[valid] = packet.capture_step
        self.last_call_step = step
        return torch.where(self.blackout.mask(step)[:, None], 0.0, self.observation)
