import torch
from torch.nn import functional as F
from gd_lab.gast.geometry import reconstruction_loss
from gd_lab.gast.observations import pose_xyyaw
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses


_CAMERA_FIELDS = ("profile", "positions", "quaternions_opengl", "sensor_size_m", "focal_m",
                  "width", "height", "source_resolution", "depth_clip")


def _camera_signature(contract):
    get = contract.get if isinstance(contract, dict) else lambda key: getattr(contract, key)
    return tuple(get(key) for key in _CAMERA_FIELDS)


def _validate_camera_capture(env, snapshot, contract):
    if len(snapshot) != 5:
        raise RuntimeError("Camera capture must include frames, depth, pose, rotation and intrinsics")
    frames, depths, positions, rotations, intrinsics = snapshot
    if frames.shape[1:] != (4, 2, contract.height, contract.width):
        raise RuntimeError(f"Camera frame shape disagrees with calibration: {tuple(frames.shape)}")
    expected_k = intrinsics.new_tensor([[contract.fx, 0, contract.width / 2],
                                        [0, contract.fy, contract.height / 2], [0, 0, 1]])
    if not torch.allclose(intrinsics, expected_k.expand_as(intrinsics), atol=1e-4, rtol=1e-5):
        raise RuntimeError("Rendered camera intrinsics do not match the teacher camera contract")
    robot = env.scene["robot"]
    trunk = robot.body_names.index("trunk")
    expected_pos, expected_rot = mounted_camera_world_poses(
        robot.data.body_pos_w[:, trunk], robot.data.body_quat_w[:, trunk], contract)
    if not torch.allclose(positions, expected_pos, atol=1e-4, rtol=1e-5):
        raise RuntimeError("Student camera mount positions differ from the teacher contract")
    if not torch.allclose(rotations, expected_rot, atol=1e-4, rtol=1e-5):
        raise RuntimeError("Student camera mount orientations differ from the teacher contract")


class GastDistillation:
    def __init__(self, teacher, student, env, camera_contract, warmup=1000, ramp=4000):
        self.teacher, self.student, self.env = teacher, student, env
        self.camera_contract = camera_contract
        self.ray_contract = load_camera_contract(env.cfg.camera_profile)
        if _camera_signature(self.ray_contract) != _camera_signature(camera_contract):
            raise RuntimeError("GAST student camera calibration differs from the BIVT-Ray teacher contract")
        # The runtime camera snapshot is validated against this contract at
        # every scheduled capture (intrinsics and world-space mount pose).
        self.warmup, self.ramp = warmup, ramp
        self.latent = torch.zeros(env.num_envs, 32, device=env.device)
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stamp = torch.full((env.num_envs,), -100., device=env.device)
        self.metrics = {}
        self.target_gate = None

    @torch.no_grad()
    def capture(self, obs, teacher_action):
        step = self.env.common_step_counter
        ray_steps = getattr(self.env, "_vrl_teacher_ray_capture_steps", None)
        ray_snapshot = getattr(self.env, "_vrl_teacher_ray_snapshot", None)
        ray_contract = getattr(self.env, "_vrl_teacher_ray_contract", None)
        camera_snapshot = getattr(self.env, "_vrl_camera_snapshot", None)
        camera_steps = getattr(self.env, "_vrl_camera_snapshot_steps", None)
        if ray_steps is None or not (ray_steps == step).all():
            raise RuntimeError("Ray teacher terrain target is stale relative to this student image capture")
        if camera_snapshot is None or camera_steps is None or not (camera_steps == step).all():
            raise RuntimeError("Student camera image is stale relative to the teacher target")
        if ray_contract is None or _camera_signature(ray_contract) != _camera_signature(self.ray_contract):
            raise RuntimeError("Ray teacher camera-contract manifest is missing or mismatched")
        if ray_snapshot is None or not torch.allclose(obs["terrain"], ray_snapshot, atol=0, rtol=0):
            raise RuntimeError("Captured teacher terrain differs from the frozen Ray observation")
        _validate_camera_capture(self.env, camera_snapshot, self.camera_contract)
        packed = self.teacher._actor_input(obs, inference=True)
        pose = torch.cat((pose_xyyaw(self.env), self.env.scene['robot'].data.root_quat_w), -1)
        stamp = torch.full_like(self.stamp, step * self.env.step_dt)
        return (packed[:, :-32].clone(), obs['gast_clean'].clone(), obs['terrain'].clone(),
                pose.clone(), stamp, teacher_action.clone())

    @torch.no_grad()
    def actions(self, obs, iteration):
        teacher_action = self.teacher.act_inference(obs)
        age = self.env.common_step_counter*self.env.step_dt-self.stamp
        fresh = (age < .3) & self.ready
        student_action = self.teacher.act_with_terrain_latent(obs, self.latent*fresh[:, None])
        probability = min(1., max(0., (iteration-self.warmup)/max(1,self.ramp)))
        use = torch.rand_like(age) < probability
        self.metrics['student_rollout_fraction'] = use.float().mean().item()
        self.metrics['stale_fraction'] = (~fresh).float().mean().item()
        return torch.where(use[:, None], student_action, teacher_action)

    def reset(self, done):
        self.ready[done] = False
        self.latent[done] = 0
        self.stamp[done] = -100

    def update(self, frames, hidden, rows, extra, age_seconds):
        base, clean, teacher_terrain, pose, stamp, teacher_action = (x[rows] for x in extra)
        frames = frames.clone()
        n = len(rows)
        missing = torch.rand(n, device=frames.device) < .1
        blur = (torch.rand(n, device=frames.device) < .15) & ~missing
        partial = torch.rand(n, 4, device=frames.device) < .08
        flat = frames.reshape(-1, 2, 45, 80)
        blurred = F.avg_pool2d(flat, 9, stride=1, padding=4).reshape_as(frames)
        frames = torch.where(blur[:,None,None,None,None], blurred, frames)
        absent = partial | missing[:, None]
        frames[:, :, 0] = torch.where(absent[...,None,None], 1., frames[:, :, 0])
        frames[:, :, 1] = torch.where(absent[...,None,None], 0., frames[:, :, 1])
        quality_target = torch.where(blur, .2, 1.) * (~missing).float()
        latent, memory, spatial, logits, gate = self.student.encode(frames, hidden, pose, age_seconds, (~missing).float())
        # Bad-input supervision asks for the learned zero-terrain behavior.
        self.target_gate = (quality_target[:,None] * max(0., 1-age_seconds/.3)).detach()
        actual = self.teacher.actor(torch.cat((base, latent), -1))
        action_loss = F.mse_loss(actual, teacher_action)
        geometry_loss, height_loss, visibility_loss, visible_fraction = reconstruction_loss(
            spatial, clean, teacher_terrain, return_components=True)
        quality_loss = F.binary_cross_entropy_with_logits(logits[:,0], quality_target)
        self.latent[rows] = latent.detach()
        self.ready[rows] = True
        self.stamp[rows] = stamp
        self.metrics.update(action_mse=action_loss.item(), spatial_loss=geometry_loss.item(),
            height_visible_loss=height_loss.item(), visibility_bce=visibility_loss.item(),
            teacher_visible_fraction=visible_fraction.item(), quality_loss=quality_loss.item(),
            quality_gate=gate.mean().item())
        return latent, memory, action_loss+.5*geometry_loss+.1*quality_loss
