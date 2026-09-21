"""Boarding-only rewards and curriculum; no privileged information enters the student."""

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from gd_lab.mdp.platform_gap_math import (
    boarding_crossing_candidates,
    boarding_drop_cost,
    boarding_support_height,
)
from gd_lab.mdp.rewards.posture import base_height_l2_clamped, slope_aligned_pitch_l2
from gd_lab.mdp.terrain_curriculums import terrain_levels_vel_cmd_aware
from gd_lab.mdp.terrain_families import terrain_family_gate


def boarding_family_mask(env):
    return terrain_family_gate(env, ("platform_gap",)) == 0


def _support(env):
    return boarding_support_height(env.scene["height_scanner"].data.ray_hits_w[..., 2], env.scene.env_origins[:, 2])


def platform_gap_foot_drop(env, asset_cfg: SceneEntityCfg):
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    return boarding_drop_cost(feet, _support(env)) * boarding_family_mask(env)


def platform_gap_base_height(env, target_height, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    usual = base_height_l2_clamped(env, target_height, asset_cfg, sensor_cfg)
    gap = (env.scene[asset_cfg.name].data.root_pos_w[:, 2] - _support(env) - target_height).square()
    return torch.where(boarding_family_mask(env), gap, usual)


def platform_gap_pitch(env, sensor_cfg, asset_cfg):
    usual = slope_aligned_pitch_l2(env, sensor_cfg, asset_cfg)
    # Two horizontal decks with a void are not a downhill plane.
    gap = env.scene[asset_cfg.name].data.projected_gravity_b[:, 0].square()
    return torch.where(boarding_family_mask(env), gap, usual)


class PlatformGapCrossing(ManagerTermBase):
    """One bonus per direction/episode, paid only on a supported upright landing."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.stable_steps = torch.zeros(env.num_envs, 2, dtype=torch.long, device=env.device)
        self.achieved = torch.zeros_like(self.paid)
        self.bypassed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.stable_steps[ids] = 0
        self.achieved[ids] = False
        self.bypassed[ids] = False

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, hold_time: float = 0.1):
        robot = env.scene[asset_cfg.name]
        terrain_cfg = env.scene.terrain.cfg.terrain_generator
        gap_cfg = terrain_cfg.sub_terrains["platform_gap"]
        feet = robot.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
        boundary = gap_cfg.gap_center_offset + gap_cfg.gap_width_range[1] / 2 + 0.06
        candidate = boarding_crossing_candidates(feet, boundary, terrain_cfg.size[1] / 2)
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        supported = (force > 10.0).sum(dim=1) >= 2
        upright = robot.data.projected_gravity_b[:, 2] < -0.7
        healthy = ~env.termination_manager.terminated
        feet_above_pit = (robot.data.body_pos_w[:, asset_cfg.body_ids, 2] > _support(env)[:, None] - 0.22).all(dim=1)
        valid = boarding_family_mask(env) & supported & upright & healthy & feet_above_pit
        # Leaving the tile sideways invalidates the episode: no reward for walking around a slot.
        outside = (feet[..., 1].abs() >= terrain_cfg.size[1] / 2 - 0.15).any(dim=1)
        self.bypassed |= outside
        candidate &= (valid & ~self.bypassed)[:, None]
        self.stable_steps = torch.where(candidate, self.stable_steps + 1, 0)
        completed = self.stable_steps * env.step_dt >= hold_time
        event = completed & ~self.paid
        self.paid |= event
        self.achieved |= event
        # RewardManager multiplies all terms by dt: compensate for this discrete event.
        return event.sum(dim=1).float() / env.step_dt


def platform_gap_levels(env, env_ids, min_moving_time: float = 2.0):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    mask = boarding_family_mask(env)[ids]
    ordinary = ids[~mask]
    if len(ordinary):
        terrain_levels_vel_cmd_aware(env, ordinary, min_moving_time)
    gap_ids = ids[mask]
    if len(gap_ids):
        term = env.reward_manager.get_term_cfg("platform_gap_crossing").func
        crossed = term.achieved[gap_ids].any(dim=1) & ~term.bypassed[gap_ids]
        command = env.command_manager.get_term("base_velocity")
        eligible = command.cmd_moving_time[gap_ids] > min_moving_time
        failed = env.termination_manager.terminated[gap_ids]
        up = eligible & crossed & ~failed
        down = eligible & ~up & (failed | (command.cmd_dist_integral[gap_ids] > 2.5))
        env.scene.terrain.update_env_origins(gap_ids, up, down)
    return env.scene.terrain.terrain_levels.float().mean()
