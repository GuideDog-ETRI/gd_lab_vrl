"""Capture-aligned supervision and student-driven rollout; actor remains frozen."""

import torch
from torch.nn import functional as F

from gd_lab.students.gavd.model import spatial_loss


class AttentionDistillation:
    def __init__(self, teacher, student, num_envs, device, warmup=1000, ramp=4000):
        self.teacher, self.student = teacher, student
        self.warmup, self.ramp = warmup, ramp
        self.latent = torch.zeros(num_envs, 32, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.metrics = {}

    @torch.no_grad()
    def capture(self, obs):
        packed = self.teacher._actor_input(obs, inference=True)
        return packed[:, :-32].clone(), self.teacher.actor(packed).clone(), obs["terrain"].clone()

    @torch.no_grad()
    def actions(self, obs, iteration):
        teacher_actions = self.teacher.act_inference(obs)
        probability = min(1., max(0., (iteration - self.warmup) / max(1, self.ramp)))
        student_actions = self.teacher.act_with_terrain_latent(obs, self.latent)
        use_student = (torch.rand_like(self.ready, dtype=torch.float) < probability) & self.ready
        self.metrics["student_rollout_fraction"] = use_student.float().mean().item()
        return torch.where(use_student[:, None], student_actions, teacher_actions)

    def reset(self, done):
        self.ready[done] = False
        self.latent[done] = 0

    def update(self, frames, hidden, rows, extra, age_seconds):
        base, actions, terrain = (x[rows] for x in extra)
        frames = frames.clone()
        dropped = torch.rand(frames.shape[:2], device=frames.device) < .08
        frames[:, :, 0] = torch.where(dropped[..., None, None], 1., frames[:, :, 0])
        frames[:, :, 1] = torch.where(dropped[..., None, None], 0., frames[:, :, 1])
        timed_hidden = torch.cat((hidden[:, :63], torch.full_like(hidden[:, 63:64], min(age_seconds, 1.))), -1)
        latent, memory, spatial = self.student.encode(frames, timed_hidden)
        predicted_action = self.teacher.actor(torch.cat((base, latent), -1))
        action_loss = F.mse_loss(predicted_action, actions)
        geometry_loss = spatial_loss(spatial, terrain)
        self.latent[rows] = latent.detach()
        self.ready[rows] = True
        self.metrics.update(action_mse=action_loss.item(), spatial_loss=geometry_loss.item(),
                            age_ms=age_seconds * 1000, view_drop_fraction=dropped.float().mean().item())
        return latent, memory, action_loss + .5 * geometry_loss
