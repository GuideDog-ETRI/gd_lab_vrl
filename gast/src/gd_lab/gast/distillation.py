import torch
from torch.nn import functional as F
from gd_lab.gast.geometry import reconstruction_loss
from gd_lab.gast.observations import pose_xyyaw


class GastDistillation:
    def __init__(self, teacher, student, env, warmup=1000, ramp=4000):
        self.teacher, self.student, self.env = teacher, student, env
        self.warmup, self.ramp = warmup, ramp
        self.latent = torch.zeros(env.num_envs, 32, device=env.device)
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stamp = torch.full((env.num_envs,), -100., device=env.device)
        self.metrics = {}
        self.target_gate = None

    @torch.no_grad()
    def capture(self, obs):
        packed = self.teacher._actor_input(obs, inference=True)
        pose = torch.cat((pose_xyyaw(self.env), self.env.scene['robot'].data.root_quat_w), -1)
        stamp = torch.full_like(self.stamp, self.env.common_step_counter*self.env.step_dt)
        return packed[:, :-32].clone(), obs['gast_clean'].clone(), pose.clone(), stamp

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
        base, clean, pose, stamp = (x[rows] for x in extra)
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
        expected = self.teacher.actor(torch.cat((base, self.teacher_latent[rows]*self.target_gate), -1))
        actual = self.teacher.actor(torch.cat((base, latent), -1))
        action_loss = F.mse_loss(actual, expected)
        geometry_loss = reconstruction_loss(spatial, clean)
        quality_loss = F.binary_cross_entropy_with_logits(logits[:,0], quality_target)
        self.latent[rows] = latent.detach()
        self.ready[rows] = True
        self.stamp[rows] = stamp
        self.metrics.update(action_mse=action_loss.item(), spatial_loss=geometry_loss.item(),
            quality_loss=quality_loss.item(), quality_gate=gate.mean().item())
        return latent, memory, action_loss+.5*geometry_loss+.1*quality_loss
