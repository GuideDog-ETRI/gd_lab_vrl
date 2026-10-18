"""UPDATE134B: GAST teacher terrain with raycast camera visibility and a per-cell IR channel.

One term owns the whole chain (no BIVT term stacked under NoisyTerrain):
  finite GT hit & raycast-visible (4 mounted cameras, leg capsules) -> height noise + episode bias
  -> 3% dropout -> blackout (whole sensor: IR too) -> invalid cells height 0 / valid 0 / IR 0.
Capture on the camera contract clock and hold in between, so height, mask and IR share one capture pose.

IR (cell-level proxy, never uses gap/terrain labels):
  I = clip(a + g * rho(x, y) * max(0, n.l) * (r0 / r)^0.5 + N(0, sigma), 0, 1), level matched to the rosbag IR
  r, l: nearest camera that sees the cell -> GT hit; n: GT scan surface normal (finite differences);
  rho: world-fixed material field per env and episode (dark patches, bright patches, low-reflectance bands);
  g: per-episode exposure. Encoding: available IR = 0.05 + 0.95 I, unavailable = 0. IR exists only on
  depth-valid cells (limitation of this proxy).

The 'terrain' group stays [height 187, valid 187] (latent gate and critic unchanged); the GAST history
carries [height, valid, IR] per frame: 8 x (3*187 + age) = 4496.
"""
import math
import os

import torch

from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses
from gd_lab.core.camera_timing import camera_period_steps, camera_refresh_mask
from gd_lab.gast.geometry import warp_memory
from gd_lab.gast.observations import NoisyTerrain, TerrainHistory, pose_xyyaw
from gd_lab.gast.temporal import HISTORY_OFFSETS
from gd_lab.teachers.bivt.raycast_terrain import leg_capsules
from gd_lab.teachers.bivt.raycast_visibility import contract_intrinsic, raycast_visible_points, warp_mesh_cast

CELLS = 187
CHANNELS = 3
FRAME = CHANNELS * CELLS + 1  # + age
IR_FLOOR = .05
IR_AMBIENT, IR_SIGMA, IR_DROP = .05, .03, .03
# Calibrated to the real RealSense IR of the noen rosbag (2026-10-02, 4 cameras, 44 frames, depth-valid pixels):
# mean .34, p10 .22, p50 .34, p90 .46, no saturation; auto-exposure keeps it nearly flat with range
# (.35 at 0.4-0.7 m, .28 at 0.7-1.0 m), so the fall-off is (r0/r)^0.5, not 1/r^2.
IR_GAIN, IR_R0, IR_FALLOFF = float(os.environ.get('GAST_IR_GAIN', '0.57')), .6, .5
RHO_BASE, RHO_DARK, RHO_BRIGHT, BAND_FACTOR = (.55, .65), .15, .75, .5
DARK_Z, BRIGHT_Z = 1.28, -1.04   # ~10% dark, ~15% bright of a unit-normal field
WAVES = 4


def scan_normals(points):
    """N,187,3 GT world hit points of the 11x17 grid -> N,187,3 world unit normals (z >= 0).

    Cross product of the two grid tangents, so it does not depend on the grid's axis order or yaw.
    """
    grid = torch.nan_to_num(points, nan=0., posinf=0., neginf=0.).reshape(-1, 11, 17, 3)
    t_col = torch.gradient(grid, dim=2)[0]
    t_row = torch.gradient(grid, dim=1)[0]
    n = torch.linalg.cross(t_col, t_row).reshape(-1, CELLS, 3)
    n = torch.where(n[..., 2:3] < 0, -n, n)
    norm = n.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(n)
    up[..., 2] = 1
    return torch.where(norm > 1e-9, n / norm.clamp_min(1e-9), up)


class MaterialField:
    """World-fixed reflectance per env, redrawn each episode, independent of terrain family/labels."""

    def __init__(self, n, device):
        self.k = torch.zeros(n, WAVES, 2, device=device)
        self.phase = torch.zeros(n, WAVES, device=device)
        self.base = torch.zeros(n, 1, device=device)
        self.band_dir = torch.zeros(n, 2, device=device)
        self.band_phase = torch.zeros(n, 1, device=device)
        self.band_freq = torch.zeros(n, 1, device=device)
        self.gain = torch.ones(n, 1, device=device)
        self.reset()

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        n = self.k[ids].shape[0]
        dev = self.k.device
        angle = torch.rand(n, WAVES, device=dev) * 2 * math.pi
        wavelength = .4 + 1.6 * torch.rand(n, WAVES, device=dev)          # 0.4-2.0 m blobs
        self.k[ids] = torch.stack((angle.cos(), angle.sin()), -1) * (2 * math.pi / wavelength)[..., None]
        self.phase[ids] = torch.rand(n, WAVES, device=dev) * 2 * math.pi
        self.base[ids] = RHO_BASE[0] + (RHO_BASE[1] - RHO_BASE[0]) * torch.rand(n, 1, device=dev)
        band = torch.rand(n, device=dev) * 2 * math.pi
        self.band_dir[ids] = torch.stack((band.cos(), band.sin()), -1)
        self.band_phase[ids] = torch.rand(n, 1, device=dev) * 2 * math.pi
        self.band_freq[ids] = 2 * math.pi / (1.5 + 2.5 * torch.rand(n, 1, device=dev))
        self.gain[ids] = .9 + .2 * torch.rand(n, 1, device=dev)

    def __call__(self, xy):
        """N,P,2 world xy -> N,P reflectance."""
        waves = torch.sin(torch.einsum('npd,nwd->npw', xy, self.k) + self.phase[:, None])
        field = waves.sum(-1) / math.sqrt(WAVES / 2)
        rho = self.base.expand_as(field).clone()
        rho = torch.where(field > DARK_Z, torch.full_like(rho, RHO_DARK), rho)
        rho = torch.where(field < BRIGHT_Z, torch.full_like(rho, RHO_BRIGHT), rho)
        band = torch.sin((xy * self.band_dir[:, None]).sum(-1) * self.band_freq + self.band_phase) > .9
        return torch.where(band, rho * BAND_FACTOR, rho)


class RayIrHistory(TerrainHistory):
    def __init__(self, env):
        super().__init__(env)
        self.frames = torch.zeros(env.num_envs, 49, CHANNELS * CELLS, device=env.device)

    def update(self, env, terrain):
        """terrain: N, 3*187 = [height, valid, IR]."""
        step = env.common_step_counter
        pose = pose_xyyaw(env)
        capture = step - self.last_capture >= self.capture_every
        if capture.any():
            for value in (self.frames, self.poses, self.steps):
                value[capture, 1:] = value[capture, :-1].clone()
            self.frames[capture, 0] = terrain[capture]
            self.poses[capture, 0] = pose[capture]
            self.steps[capture, 0] = step
            self.last_capture[capture] = step
        index = list(HISTORY_OFFSETS)
        grids = self.frames[:, index].reshape(-1, CHANNELS, CELLS).transpose(1, 2)
        ir_available = (grids[..., 2:3] > 0).float()
        warped = warp_memory(torch.cat((grids, ir_available), -1), self.poses[:, index].reshape(-1, 3),
                             pose[:, None].expand(-1, 8, -1).reshape(-1, 3))
        valid = warped[..., 1] > .999
        ir = torch.where(valid & (warped[..., 3] > .999), warped[..., 2], 0)
        packed = torch.cat((torch.where(valid, warped[..., 0], 0), valid.float(), ir), -1)
        packed = packed.reshape(-1, 8, CHANNELS * CELLS)
        packed[:, -1] = terrain
        age = ((step - self.steps[:, index]).clamp(0, 10**6).float() * env.step_dt).clamp(0, 5)
        age[:, -1] = 0
        return torch.cat((packed, age[..., None]), -1).flatten(1)


class RaycastIrTerrain(NoisyTerrain):
    def __init__(self, cfg, env):
        self.contract = load_camera_contract(env.cfg.camera_profile)
        self.period_steps = camera_period_steps(
            env.cfg.decimation * env.cfg.sim.dt, self.contract.policy_dt, self.contract.period_steps)
        self.intrinsic = contract_intrinsic(self.contract, env.device)
        scanner = env.scene['height_scanner']
        self.cast = warp_mesh_cast(scanner.meshes[scanner.cfg.mesh_prim_paths[0]])
        self.capture_step = torch.full((env.num_envs,), -100, device=env.device, dtype=torch.long)
        self.held = torch.zeros(env.num_envs, CHANNELS * CELLS, device=env.device)
        self.materials = MaterialField(env.num_envs, env.device)
        self._ir_debug = os.environ.get('GAST_IR_DEBUG', '0') == '1'
        # [visible cells, scan cells, IR available, IR sum, dark-material visible cells]
        self.ir_counts = torch.zeros(5, device=env.device, dtype=torch.float64)
        super().__init__(cfg, env)
        self.mapper_aug = None
        if getattr(env.cfg, "mapper_aug_enabled", False):
            from gd_lab.gast.mapper_aug import MapperAugCfg, MapperAugmentation
            self.mapper_aug = MapperAugmentation(env.num_envs, env.device, MapperAugCfg(
                enabled=True, ramp_steps=env.cfg.mapper_aug_ramp_steps,
                clean_fraction=env.cfg.mapper_aug_clean_fraction, seed=env.cfg.mapper_aug_seed))
        self.history = RayIrHistory(env)
        self.history.reset()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if hasattr(self, 'capture_step'):
            self.capture_step[ids] = -100
            self.held[ids] = 0
            self.materials.reset(env_ids)
        if hasattr(self, "unknown_run"):
            ids = slice(None) if env_ids is None else env_ids
            self.unknown_run[ids] = 0
            self.extra_unknown_run[ids] = 0
            self.held_extra_unknown[ids] = False
        if getattr(self, "mapper_aug", None) is not None:
            self.mapper_aug.reset(env_ids)

    def _capture(self, env, strength):
        scan = env.scene['height_scanner'].data
        if scan.ray_hits_w.shape[1] != CELLS:
            raise ValueError('GAST Ray+IR requires the configured 11x17 height scan')
        points = scan.ray_hits_w
        z = points[..., 2]
        finite = torch.isfinite(points).all(-1)
        robot = env.scene['robot']
        trunk = robot.body_names.index('trunk')
        positions, rotations = mounted_camera_world_poses(
            robot.data.body_pos_w[:, trunk], robot.data.body_quat_w[:, trunk], self.contract)
        occluders = leg_capsules(robot)
        visible = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
        best_r = torch.full(points.shape[:2], float('inf'), device=points.device)
        best_l = torch.zeros_like(points)
        for camera in range(positions.shape[1]):
            seen = raycast_visible_points(points, positions[:, camera], rotations[:, camera], self.intrinsic,
                                          (self.contract.height, self.contract.width), self.contract.depth_clip,
                                          self.cast, occluders=occluders)
            to_camera = positions[:, camera, None] - points
            r = torch.nan_to_num(to_camera.norm(dim=-1), nan=float('inf'), posinf=float('inf'))
            closer = seen & (r < best_r)
            best_r = torch.where(closer, r, best_r)
            best_l = torch.where(closer[..., None], to_camera / r.clamp_min(1e-6)[..., None], best_l)
            visible |= seen
        visible &= finite
        # Height: same scale/offset as clean_terrain; noise chain of NoisyTerrain.
        height_z = z
        if self.mapper_aug is not None:
            from gd_lab.mdp.terrain_families import terrain_family_gate
            height_z = self.mapper_aug.geometry(points, scan.pos_w[:, 2], env.scene.env_origins[:, 2],
                pose_xyyaw(env)[:, 2], terrain_family_gate(env, ("platform_gap", "gap_field")) == 0,
                env.common_step_counter, self.cast)
        h = ((scan.pos_w[:, 2, None] - height_z - .5).clamp(-1, 1))
        h = torch.where(visible, h, 0.)
        h = h + strength * (torch.randn_like(h) * .01 + self.bias)
        valid = visible & (torch.rand_like(h) > .03 * strength)
        # IR from geometry and the world-fixed material field only.
        normals = scan_normals(points)
        cosine = (normals * best_l).sum(-1).clamp_min(0)
        rho = self.materials(points[..., :2].nan_to_num(0.))
        falloff = (IR_R0 / best_r.clamp(.15, 1e3)) ** IR_FALLOFF
        intensity = IR_AMBIENT + IR_GAIN * self.materials.gain * rho * cosine * falloff
        intensity = (intensity + torch.randn_like(intensity) * IR_SIGMA).clamp(0, 1)
        if self._ir_debug and visible.any():
            q = torch.quantile(intensity[visible][:200000].float(), torch.tensor([.1, .5, .9], device=h.device))
            print(f'[GAST_IR_DEBUG] step={env.common_step_counter} mean={intensity[visible].mean():.3f} '
                  f'p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f} sat={(intensity[visible] >= .99).float().mean():.3f} '
                  f'dark<.1={(intensity[visible] < .1).float().mean():.3f}', flush=True)
        ir_ok = valid & (torch.rand_like(intensity) > IR_DROP)
        ir = torch.where(ir_ok, IR_FLOOR + (1 - IR_FLOOR) * intensity, 0.)
        height = torch.where(valid, h.clamp(-1, 1) * 5, 0.)
        dark = visible & (rho <= RHO_DARK + 1e-6)
        self.ir_counts += torch.stack((visible.sum().double(), visible.new_tensor(visible.numel(), dtype=torch.float64),
                                       ir_ok.sum().double(), ir.sum().double(), dark.sum().double()))
        if self.mapper_aug is not None:
            height, valid, ir = self.mapper_aug.unknown(height, valid, ir, points, pose_xyyaw(env)[:, 2])
        return torch.cat((height, valid.float(), ir), -1)

    def __call__(self, env):
        step = env.common_step_counter
        if self.last_step == step and self.cached is not None:
            return self.cached.clone()
        strength = min(1., .1 + (step + self.step_offset) / 200000.)
        refresh = camera_refresh_mask(self.capture_step, step, self.period_steps)
        if refresh.any():
            fresh = self._capture(env, strength)
            self.held[refresh] = fresh[refresh]
            self.capture_step[refresh] = step
            if getattr(env, '_aug149_diagnostics', False):
                if not hasattr(self, 'held_extra_unknown'):
                    self.held_extra_unknown = torch.zeros_like(self.held[:, :CELLS], dtype=torch.bool)
                if self.mapper_aug is not None:
                    self.held_extra_unknown[refresh] = self.mapper_aug.removed[refresh]
        blackout = self.blackout.mask(step)
        full = torch.where(blackout[:, None], 0., self.held)
        valid = full[:, CELLS:2 * CELLS].bool()
        if getattr(env, "_aug149_diagnostics", False):
            if not hasattr(self, "unknown_run"):
                self.unknown_run = torch.zeros_like(valid, dtype=torch.long)
                self.longest_unknown_steps = torch.zeros((), dtype=torch.long, device=valid.device)
                self.extra_unknown_run = torch.zeros_like(self.unknown_run)
                self.longest_extra_unknown_steps = self.longest_unknown_steps.clone()
            self.unknown_run = torch.where(valid, 0, self.unknown_run + 1)
            self.longest_unknown_steps = torch.maximum(self.longest_unknown_steps, self.unknown_run.max())
            self.extra_unknown_run = torch.where(self.held_extra_unknown, self.extra_unknown_run+1, 0)
            self.longest_extra_unknown_steps = torch.maximum(self.longest_extra_unknown_steps, self.extra_unknown_run.max())
            if not torch.isfinite(full).all():
                raise FloatingPointError("UPDATE149B nonfinite actor frame")
        self.diagnostic_counts += torch.stack((blackout.sum(), blackout.new_tensor(env.num_envs, dtype=torch.long),
                                              valid.sum(), valid.new_tensor(valid.numel(), dtype=torch.long),
                                              full.new_tensor(strength * env.num_envs))).double()
        self.cached = full[:, :2 * CELLS].clone()
        self.last_step = step
        env._gast_history = self.history.update(env, full)
        return self.cached.clone()
