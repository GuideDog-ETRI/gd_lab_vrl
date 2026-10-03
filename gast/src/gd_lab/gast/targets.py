"""Pure supervision-target helpers, kept independent of Isaac Sim imports."""

import torch


def gated_teacher_action(teacher, base_actor, teacher_latent, gate):
    return teacher.actor(torch.cat((base_actor, teacher_latent * gate), -1))
