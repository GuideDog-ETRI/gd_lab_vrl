"""Gap tiles start facing a crossing; other tiles retain their original commands."""

import math

import torch
from isaaclab.envs.mdp import reset_root_state_uniform

from gd_lab.mdp.commands_pulse import UniformThresholdVelocityCommand
from gd_lab.mdp.platform_gap_terms import boarding_family_mask


class BoardingVelocityCommand(UniformThresholdVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[boarding_family_mask(self._env)[ids]]
        if not len(ids):
            return
        # Keep heading along world +/-X; no lateral command that can bypass the gap.
        heading = self.robot.data.heading_w[ids]
        self.heading_target[ids] = torch.where(heading.cos() >= 0, 0.0, math.pi)
        self.is_heading_env[ids] = True
        # The generic initial 0.12 m/s range cannot reliably bring the rear
        # feet across within a 20 s episode with standing windows. Boarding
        # has its own 0.20 m/s floor and 0.30..0.60 m/s evolving ceiling.
        upper = min(0.6, max(0.3, self.cfg.ranges.lin_vel_x[1]))
        self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(0.2, upper)
        self.vel_command_b[ids, 1] = 0.0
        self._pulse_active[ids] = False


def reset_boarding_root(env, env_ids, pose_range, velocity_range):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    gap = boarding_family_mask(env)[ids]
    if (~gap).any():
        reset_root_state_uniform(env, ids[~gap], pose_range, velocity_range)
    ids = ids[gap]
    reverse = torch.rand(len(ids), device=env.device) < 0.5
    for selection, heading in ((ids[~reverse], 0.0), (ids[reverse], math.pi)):
        if len(selection):
            pose = dict(pose_range, x=(-0.3, 0.3), y=(-0.3, 0.3), yaw=(heading - 0.1, heading + 0.1))
            reset_root_state_uniform(env, selection, pose, velocity_range)
