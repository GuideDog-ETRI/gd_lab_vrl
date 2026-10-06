"""Capture-aligned supervision and student-driven rollout; actor remains frozen."""

import torch
from torch.nn import functional as F

from gd_lab.students.gap_focus import row_mse, weighted_mean
from gd_lab.students.gavd.model import spatial_loss


class AttentionDistillation:
    def __init__(self, teacher, student, num_envs, device, warmup=1000, ramp=4000, stale_student_rollout=False):
        self.teacher, self.student = teacher, student
        self.warmup, self.ramp = warmup, ramp
        # False (legacy): a stale/missing student latent hands the env back to the privileged teacher.
        # True: the selected env keeps the student policy and walks the blind (zero-latent) route,
        # as deployment does when the camera stream stalls (and as the GAST loop already does).
        self.stale_student_rollout = bool(stale_student_rollout)
        self.latent = torch.zeros(num_envs, 32, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stamp = torch.full((num_envs,), -100.0, device=device)
        self.metrics = {}

    @torch.no_grad()
    def capture(self, obs, teacher_action, base_actor=None, teacher_terrain=None):
        packed = self.teacher._actor_input(obs, inference=True)
        base = packed[:, :-32] if base_actor is None else base_actor
        terrain = obs["terrain"] if teacher_terrain is None else teacher_terrain
        if terrain.shape[-1] != 374:
            raise RuntimeError("GAVD requires the synchronized 187-height + 187-visibility teacher map")
        return base.clone(), teacher_action.clone(), terrain.clone()

    @torch.no_grad()
    def actions(self, obs, iteration, current_time_seconds):
        teacher_actions = self.teacher.act_inference(obs)
        probability = min(1., max(0., (iteration - self.warmup) / max(1, self.ramp)))
        age = current_time_seconds - self.stamp
        fresh = self.ready & (age >= 0) & (age < 0.3)
        student_actions = self.teacher.act_with_terrain_latent(obs, self.latent * fresh[:, None])
        use_student = torch.rand_like(self.ready, dtype=torch.float) < probability
        if not self.stale_student_rollout:
            use_student &= fresh
        self.metrics["student_rollout_fraction"] = use_student.float().mean().item()
        self.metrics["stale_student_rollout_fraction"] = (use_student & ~fresh).float().mean().item()
        self.metrics["stale_fraction"] = (~fresh).float().mean().item()
        return torch.where(use_student[:, None], student_actions, teacher_actions)

    def reset(self, done):
        self.ready[done] = False
        self.latent[done] = 0
        self.stamp[done] = -100.0

    def update(self, frames, hidden, rows, extra, age_seconds, capture_time_seconds=None, row_weight=None,
               near_gap=None):
        """``row_weight`` [B] near-gap emphasis (None = 1); ``near_gap`` [B] bool for the gap Top-5 metric."""
        base, actions, terrain = (x[rows] for x in extra)
        frames = frames.clone()
        dropped = torch.rand(frames.shape[:2], device=frames.device) < .08
        frames[:, :, 0] = torch.where(dropped[..., None, None], 1., frames[:, :, 0])
        frames[:, :, 1] = torch.where(dropped[..., None, None], 0., frames[:, :, 1])
        timed_hidden = torch.cat((hidden[:, :63], torch.full_like(hidden[:, 63:64], min(age_seconds, 1.))), -1)
        latent, memory, spatial = self.student.encode(frames, timed_hidden)
        predicted_action = self.teacher.actor(torch.cat((base, latent), -1))
        weight = torch.ones(len(rows), device=frames.device) if row_weight is None else row_weight
        per_row = row_mse(predicted_action, actions)
        action_loss = weighted_mean(per_row, weight)
        geometry_loss, height_loss, visibility_loss, visible_fraction = spatial_loss(
            spatial, terrain, return_components=True, row_weight=row_weight)
        with torch.no_grad():
            gap = torch.zeros_like(weight, dtype=torch.bool) if near_gap is None else near_gap
            self.near_gap_action_sse = float(per_row[gap].sum())
            self.near_gap_rows = int(gap.sum())
        self.latent[rows] = latent.detach()
        self.ready[rows] = True
        # Production supplies the packet timestamp on the simulation clock.
        # Keep direct CPU/unit callers compatible with the pre-alignment API.
        self.stamp[rows] = 0.0 if capture_time_seconds is None else capture_time_seconds
        self.metrics.update(action_mse=action_loss.item(), spatial_loss=geometry_loss.item(),
                            height_visible_loss=height_loss.item(), visibility_bce=visibility_loss.item(),
                            teacher_visible_fraction=visible_fraction.item(),
                            age_ms=age_seconds * 1000, view_drop_fraction=dropped.float().mean().item())
        return latent, memory, action_loss + .5 * geometry_loss
