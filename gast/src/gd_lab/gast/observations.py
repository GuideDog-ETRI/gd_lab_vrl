"""No-render teacher observations; camera snapshots only in student task."""
import torch
from isaaclab.managers import ManagerTermBase
from gd_lab.mdp.terrain_families import terrain_family_gate
from gd_lab.teachers.bivt.blackout import BlackoutSchedule
from gd_lab.gast.geometry import warp_memory
from gd_lab.gast.temporal import HISTORY_OFFSETS

def pose_xyyaw(env):
    data = env.scene['robot'].data
    w, x, y, z = data.root_quat_w.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return torch.cat((data.root_pos_w[:, :2], yaw[:, None]), -1)

class TerrainHistory:
    def __init__(self, env):
        n, device = env.num_envs, env.device
        self.frames = torch.zeros(n, 49, 374, device=device)
        self.poses = torch.zeros(n, 49, 3, device=device)
        self.times = torch.full((n, 49), -100., device=device)
        self.last_capture = torch.full((n,), -100., device=device)
    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.frames[ids] = 0
        self.times[ids] = -100
        self.last_capture[ids] = -100
    def update(self, env, terrain):
        now = env.common_step_counter * env.step_dt
        pose = pose_xyyaw(env)
        capture = now-self.last_capture >= .1-1e-6
        if capture.any():
            for value in (self.frames, self.poses, self.times):
                value[capture, 1:] = value[capture, :-1].clone()
            self.frames[capture, 0] = terrain[capture]
            self.poses[capture, 0] = pose[capture]
            self.times[capture, 0] = now
            self.last_capture[capture] = now
        index = list(HISTORY_OFFSETS)
        grids = self.frames[:, index].reshape(-1, 2, 187).transpose(1, 2)
        warped = warp_memory(grids, self.poses[:, index].reshape(-1, 3),
            pose[:, None].expand(-1, 8, -1).reshape(-1, 3))
        valid = warped[..., 1] > .999
        packed = torch.cat((torch.where(valid, warped[..., 0], 0), valid.float()), -1).reshape(-1, 8, 374)
        packed[:, -1] = terrain
        age = (now-self.times[:, index]).clamp(0, 5)
        age[:, -1] = 0
        return torch.cat((packed, age[..., None]), -1).flatten(1)

def history_observation(env):
    env._gast_terrain_term(env)
    return env._gast_history.clone()


def clean_terrain(env):
    scan = env.scene['height_scanner'].data
    z = scan.ray_hits_w[..., 2]
    valid = torch.isfinite(z)
    h = (scan.pos_w[:, 2, None] - z - .5).clamp(-1, 1) * 5
    h = torch.where(valid, h, 0.)
    # All current non-gap terrain families contain solid ground, including stairs.
    gap_family = terrain_family_gate(env, ('platform_gap', 'gap_field')) == 0
    gap = gap_family[:, None] & valid & (z < env.scene.env_origins[:, 2, None] - .3)
    return torch.cat((h, valid.float(), gap.float(), valid.float()), -1)


class NoisyTerrain(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._gast_terrain_term = self
        self.blackout = BlackoutSchedule(env.num_envs, env.device, .001, (50, 300), .05)
        self.blackout.reset()
        self.bias = torch.zeros(env.num_envs, 1, device=env.device)
        self.last_step = -1
        self.cached = None
        self.history = TerrainHistory(env)
        self.reset()

    def reset(self, env_ids=None):
        self.blackout.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.bias[ids] = torch.randn_like(self.bias[ids]) * .015
        self.last_step = -1

        if hasattr(self, 'history'):
            self.history.reset(env_ids)

    def __call__(self, env):
        step = env.common_step_counter
        if self.last_step == step and self.cached is not None:
            return self.cached.clone()
        clean = clean_terrain(env)
        h, valid = clean[:, :187] / 5, clean[:, 187:374].bool()
        # Ramp observation corruption over the first 200k control steps.
        strength = min(1., .1 + step / 200000.)
        h = h + strength * (torch.randn_like(h) * .01 + self.bias)
        valid = valid & (torch.rand_like(h) > .03 * strength)
        blackout = self.blackout.mask(step)
        valid = valid & ~blackout[:, None]
        self.cached = torch.cat((torch.where(valid, h.clamp(-1, 1)*5, 0), valid.float()), -1)
        self.last_step = step
        env._gast_history = self.history.update(env, self.cached)
        return self.cached.clone()


class StudentTerrain(NoisyTerrain):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from gd_lab.mdp.camera_observations import CameraVisibleTerrain
        self.snapshots = CameraVisibleTerrain(cfg, env)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'snapshots'):
            self.snapshots.reset(env_ids)

    def __call__(self, env):
        self.snapshots(env)
        # Clean oracle teacher labels. No random blackout of the frozen expert.
        if self.last_step != env.common_step_counter:
            self.cached = clean_terrain(env)[:, :374]
            env._gast_history = self.history.update(env, self.cached)
            self.last_step = env.common_step_counter
        return self.cached.clone()
