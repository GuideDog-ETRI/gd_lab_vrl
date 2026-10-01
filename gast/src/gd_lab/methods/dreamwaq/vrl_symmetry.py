"""Optional mirror augmentation including the stored visibility mask."""

import torch

from gd_lab.methods.dreamwaq.symmetry import compute_symmetric_states


def mirror_vrl_observations(obs=None, actions=None, env=None):
    base = obs.exclude("terrain") if obs is not None else None
    augmented, actions_aug = compute_symmetric_states(base, actions, env)
    if obs is not None:
        terrain = obs["terrain"]
        if terrain.shape[-1] != 374:
            raise ValueError("VRL mirror expects height+mask, 2x11x17")
        mirrored = terrain.reshape(-1, 2, 11, 17).flip(-2).reshape_as(terrain)
        augmented["terrain"] = torch.cat((terrain, mirrored), 0)
    return augmented, actions_aug
