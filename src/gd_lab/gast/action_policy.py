"""UPDATE19 student-owned action policies and execution-time supervision (CPU-testable)."""
import copy
import math
import torch
from torch import nn
from gd_lab.gast.geometry import near_gap_rows, weighted_mean


def validate_options(action_target, student_policy, gate_target, action_loss_weight, latent_loss_weight):
    if action_target not in ('capture', 'current') or student_policy not in ('frozen', 'copy', 'residual'):
        raise ValueError('invalid action target or student policy')
    if gate_target not in ('gated', 'ungated_latent'):
        raise ValueError('invalid gate target')
    for value in (action_loss_weight, latent_loss_weight):
        if not math.isfinite(value) or value < 0:
            raise ValueError('loss weights must be finite and nonnegative')


class StudentActionPolicy(nn.Module):
    """Actor input retains the teacher's [base, 32D executed latent] contract.

    age is current execution time minus capture time, capped at 1 s; gate is
    the delivered effective gate (including freshness). Residual uses both.
    """
    def __init__(self, teacher_actor, mode):
        super().__init__()
        if mode not in ('copy', 'residual'):
            raise ValueError('student-owned policy must be copy or residual')
        self.mode = mode
        self.actor = copy.deepcopy(teacher_actor)
        self.actor.requires_grad_(mode == 'copy')
        if mode == 'residual':
            linear = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
            self.residual = nn.Sequential(nn.Linear(linear[0].in_features + 2, 64), nn.ELU(),
                                          nn.Linear(64, linear[-1].out_features))
            self.residual.to(device=linear[0].weight.device, dtype=linear[0].weight.dtype)
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, base, latent, age, gate):
        packed = torch.cat((base, latent), -1)
        action = self.actor(packed)
        if self.mode == 'residual':
            context = torch.cat((packed, age.reshape(-1, 1).clamp(0, 1), gate.reshape(-1, 1)), -1)
            action = action + self.residual(context)
        return action


class ExecutionCache:
    """Differentiable latest delivery; packet timestamps are never relabelled.

    The caller performs delivery_mask before put(), and calls reset at episode
    boundaries. Values survive BPTT boundaries, but their graph does not.
    """
    def __init__(self, n, device):
        self.latent = torch.zeros(n, 32, device=device)
        self.gate = torch.zeros(n, 1, device=device)
        self.stamp = torch.full((n,), -100., device=device)
        self.ready = torch.zeros(n, dtype=torch.bool, device=device)

    def put(self, rows, latent, gate, stamp):
        self.latent = self.latent.index_copy(0, rows, latent)
        self.gate = self.gate.index_copy(0, rows, gate)
        self.stamp[rows] = stamp
        self.ready[rows] = True

    def reset(self, done):
        self.latent = torch.where(done[:, None], torch.zeros_like(self.latent), self.latent)
        self.gate = torch.where(done[:, None], torch.zeros_like(self.gate), self.gate)
        self.ready[done] = False
        self.stamp[done] = -100.

    def detach(self):
        self.latent = self.latent.detach()
        self.gate = self.gate.detach()

    def inputs(self, now):
        age = now - self.stamp
        fresh = (age < .3) & self.ready
        return self.latent * fresh[:, None], age, self.gate * fresh[:, None]


def execution_action_loss(policy, teacher_actor, base, latent, age, gate, expected,
                          use_student, clean, gap_weight, gap_action_weight):
    """Same prediction is returned for execution and the supervised action loss."""
    actual = (teacher_actor(torch.cat((base, latent), -1)) if policy is None
              else policy(base, latent, age, gate))
    near = near_gap_rows(clean)
    weight = 1 + (max(gap_weight, gap_action_weight) - 1) * near.float()
    if use_student.any():
        loss = weighted_mean((actual[use_student] - expected[use_student]).square().mean(-1),
                             weight[use_student])
    else:
        loss = actual.sum() * 0.
    executed = torch.where(use_student[:, None], actual, expected).detach()
    return executed, loss


def restore_action_policy(policy, saved, config):
    """Legacy -> experiment starts fresh Adam; experiment resume must match exactly."""
    previous = saved.get('action_distillation')
    if previous is not None and previous != config:
        raise ValueError('action distillation resume contract differs; use an explicit new experiment')
    if previous is not None and policy is not None:
        policy.load_state_dict(saved['student_policy_state'], strict=True)
    return previous is not None or config == dict(action_target='capture', student_policy='frozen',
        gate_target='gated', action_loss_weight=1., latent_loss_weight=1.)
