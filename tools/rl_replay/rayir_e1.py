"""UPDATE136: E1 input for the UPDATE134B Ray+IR teacher.

The arm map (G / GV / L0D ...) still supplies [height, valid]; this adds the third history channel, IR,
with the training proxy of gd_lab.gast.rayir (update134_rayir, IR_GAIN .57) unchanged:
  I = clip(a + g * rho(x, y) * max(0, n.l) * (r0 / r)^0.5 + N(0, sigma), 0, 1); IR = .05 + .95 I, else 0.
IR exists only where the arm cell is valid AND one of the 4 mounted cameras sees the GT hit (leg capsules
occlude), and drops 3% i.i.d., as in training. U136_IR=zero feeds IR = 0 everywhere (ablation).
The material field is world-fixed per env and redrawn at each reset, seeded by the E1 seed.
"""
import math
import os

import torch

from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses
from gd_lab.gast.geometry import warp_memory
from gd_lab.gast.observations import TerrainHistory, pose_xyyaw
from gd_lab.gast.temporal import HISTORY_OFFSETS
from gd_lab.teachers.bivt.raycast_terrain import leg_capsules
from gd_lab.teachers.bivt.raycast_visibility import contract_intrinsic, raycast_visible_points, warp_mesh_cast

CELLS = 187
CHANNELS = 3
IR_FLOOR = .05
IR_AMBIENT, IR_SIGMA, IR_DROP = .05, .03, .03
IR_GAIN, IR_R0, IR_FALLOFF = .57, .6, .5
RHO_BASE, RHO_DARK, RHO_BRIGHT, BAND_FACTOR = (.55, .65), .15, .75, .5
DARK_Z, BRIGHT_Z = 1.28, -1.04
WAVES = 4


def scan_normals(points):
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
    def __init__(self, n, device, generator):
        self.g = generator
        self.k = torch.zeros(n, WAVES, 2, device=device)
        self.phase = torch.zeros(n, WAVES, device=device)
        self.base = torch.zeros(n, 1, device=device)
        self.band_dir = torch.zeros(n, 2, device=device)
        self.band_phase = torch.zeros(n, 1, device=device)
        self.band_freq = torch.zeros(n, 1, device=device)
        self.gain = torch.ones(n, 1, device=device)
        self.reset()

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.k.device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        n = self.k[ids].shape[0]
        if n == 0:
            return
        angle = self._rand(n, WAVES) * 2 * math.pi
        wavelength = .4 + 1.6 * self._rand(n, WAVES)
        self.k[ids] = torch.stack((angle.cos(), angle.sin()), -1) * (2 * math.pi / wavelength)[..., None]
        self.phase[ids] = self._rand(n, WAVES) * 2 * math.pi
        self.base[ids] = RHO_BASE[0] + (RHO_BASE[1] - RHO_BASE[0]) * self._rand(n, 1)
        band = self._rand(n) * 2 * math.pi
        self.band_dir[ids] = torch.stack((band.cos(), band.sin()), -1)
        self.band_phase[ids] = self._rand(n, 1) * 2 * math.pi
        self.band_freq[ids] = 2 * math.pi / (1.5 + 2.5 * self._rand(n, 1))
        self.gain[ids] = .9 + .2 * self._rand(n, 1)

    def __call__(self, xy):
        waves = torch.sin(torch.einsum('npd,nwd->npw', xy, self.k) + self.phase[:, None])
        field = waves.sum(-1) / math.sqrt(WAVES / 2)
        rho = self.base.expand_as(field).clone()
        rho = torch.where(field > DARK_Z, torch.full_like(rho, RHO_DARK), rho)
        rho = torch.where(field < BRIGHT_Z, torch.full_like(rho, RHO_BRIGHT), rho)
        band = torch.sin((xy * self.band_dir[:, None]).sum(-1) * self.band_freq + self.band_phase) > .9
        return torch.where(band, rho * BAND_FACTOR, rho)


class CellIr:
    """Per-cell IR for the current GT scan; computed on each call (32 envs)."""

    def __init__(self, env, seed):
        self.mode = os.environ.get('U136_IR', 'native')
        assert self.mode in ('native', 'zero'), self.mode
        self.contract = load_camera_contract(env.cfg.camera_profile)
        self.intrinsic = contract_intrinsic(self.contract, env.device)
        scanner = env.scene['height_scanner']
        self.cast = warp_mesh_cast(scanner.meshes[scanner.cfg.mesh_prim_paths[0]])
        self.g = torch.Generator(device=env.device)
        self.g.manual_seed(int(seed) * 7919 + 136)
        self.materials = MaterialField(env.num_envs, env.device, self.g)
        self.counts = torch.zeros(3, dtype=torch.float64, device=env.device)  # valid, ir available, ir sum

    def reset(self, ids):
        if len(ids):
            self.materials.reset(ids)

    def __call__(self, env, valid):
        """valid: N,187 bool (arm map validity) -> N,187 IR channel."""
        if self.mode == 'zero':
            return torch.zeros(valid.shape, device=valid.device)
        points = env.scene['height_scanner'].data.ray_hits_w
        finite = torch.isfinite(points).all(-1)
        robot = env.scene['robot']
        trunk = robot.body_names.index('trunk')
        positions, rotations = mounted_camera_world_poses(
            robot.data.body_pos_w[:, trunk], robot.data.body_quat_w[:, trunk], self.contract)
        occluders = leg_capsules(robot)
        seen_any = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
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
            seen_any |= seen
        normals = scan_normals(points)
        cosine = (normals * best_l).sum(-1).clamp_min(0)
        rho = self.materials(points[..., :2].nan_to_num(0.))
        falloff = (IR_R0 / best_r.clamp(.15, 1e3)) ** IR_FALLOFF
        intensity = IR_AMBIENT + IR_GAIN * self.materials.gain * rho * cosine * falloff
        noise = torch.randn(intensity.shape, generator=self.g, device=intensity.device) * IR_SIGMA
        intensity = (intensity + noise).clamp(0, 1)
        keep = torch.rand(intensity.shape, generator=self.g, device=intensity.device) > IR_DROP
        ok = valid & seen_any & finite & keep
        ir = torch.where(ok, IR_FLOOR + (1 - IR_FLOOR) * intensity, 0.)
        self.counts += torch.stack((valid.sum().double(), ok.sum().double(), ir.sum().double()))
        return ir

    def summary(self):
        v, a, s = self.counts.tolist()
        return dict(mode=self.mode, valid_cells=v, ir_available_fraction_of_valid=a / max(v, 1),
                    ir_mean_where_available=s / max(a, 1))


class RayIrHistory(TerrainHistory):
    """Copy of gd_lab.gast.rayir.RayIrHistory (update134_rayir): 8 x (3*187 + age) = 4496."""

    def __init__(self, env):
        super().__init__(env)
        self.frames = torch.zeros(env.num_envs, 49, CHANNELS * CELLS, device=env.device)

    def update(self, env, terrain, pose_override=None):
        step = env.common_step_counter
        pose = pose_xyyaw(env) if pose_override is None else pose_override
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
