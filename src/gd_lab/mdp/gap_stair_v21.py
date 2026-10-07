"""v2.1 (main 30k-update training, user decisions 2026-10-07) on top of v2.

- Speed: gaps and stairs must be crossed up to 1.2 m/s, with 0.8-1.0 m/s the usual pace. Gap tiles (v2: 0.2-0.6
  m/s) and half of the stair-tile resamples (v2: generic random commands) get a forward command from
  ``V21_SPEED_MIX``, walking straight up/down the stairs. The ceiling ramps from 0.6 to 1.2 m/s over
  GD_LAB_V21_SPEED_RAMP_STEPS env steps of this process (default 300k ~ 3k updates; 0 = full range at once).
- Gap hop: both hind feet airborne near a gap (the MuJoCo "hind legs hop over the gap" and forward dives).
- Overspeed: forward speed above command + 0.2 m/s on gap and stair tiles (the dives followed a 0.7-1.0 m/s surge).
- Stair stall: a forward command on the stair band but < 0.05 m/s progress for more than 2 s.
- Gap foothold margin halved (-0.5 -> -0.25): it seems to push the hop-over.
Observations unchanged (a v2 checkpoint resumes as is). gast/src keeps a byte-identical copy.
"""

from __future__ import annotations

import math
import os

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from gd_lab.mdp.gap_stair_v2 import _StairGeometry, _gap_mask, add_v2_terms, chebyshev_frame
from gd_lab.mdp.gap_stair_v2_math import V21_SPEED_MIX, sample_v21_speed, speed_ceiling  # noqa: F401
from gd_lab.mdp.platform_gap_commands import BoardingVelocityCommand
from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
from gd_lab.mdp.platform_gap_terms import boarding_family_mask
from gd_lab.mdp.terrain_families import family_column_masks

V21_STAIR_FORWARD_PROB = 0.5
V21_WEIGHTS = {"gap_foothold_margin": -0.25, "gap_hind_hop": -1.0, "overspeed": -1.0, "stair_stall": -0.5}
CONTACT_N = 10.0


class V21VelocityCommand(BoardingVelocityCommand):
    """Boarding command (gap tiles face their crossing) with the v2.1 speeds, plus straight stair runs."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stairs = None
        self.ramp_steps = int(os.environ.get("GD_LAB_V21_SPEED_RAMP_STEPS", 300_000))
        print(f"[INFO] v2.1 command: speed ceiling ramp steps={self.ramp_steps}", flush=True)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        ceiling = speed_ceiling(int(self._env.common_step_counter), self.ramp_steps)
        gap = ids[boarding_family_mask(self._env)[ids]]
        if len(gap):
            self.vel_command_b[gap, 0] = sample_v21_speed(len(gap), ceiling, self.device)
        if self._stairs is None:
            self._stairs = _StairGeometry(self._env)
        col = self._env.scene.terrain.terrain_types.long()[ids]
        stair = ids[self._stairs.is_stair[col] & (torch.rand(len(ids), device=self.device) < V21_STAIR_FORWARD_PROB)]
        if len(stair):
            rel = self.robot.data.root_pos_w[stair, :2] - self._env.scene.env_origins[stair, :2]
            _, outward = chebyshev_frame(rel)
            inward = torch.rand(len(stair), device=self.device) < 0.5  # up or down, whichever way the tile goes
            yaw = torch.atan2(outward[:, 1], outward[:, 0]) + inward.float() * math.pi
            self.heading_target[stair] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            self.is_heading_env[stair] = True
            self.vel_command_b[stair, 0] = sample_v21_speed(len(stair), ceiling, self.device)
            self.vel_command_b[stair, 1] = 0.0
            self._pulse_active[stair] = False
            self.is_standing_env[stair] = False


class GapHindHop(ManagerTermBase):
    """Per second: both hind feet airborne while the base is within ``reach`` m of a gap slot on a gap tile."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.geometry = GapTileGeometry(env)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, reach: float = 0.6):
        slots, _ = self.geometry.for_envs(env)
        x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        near = ((x[:, None] > slots[..., 0] - reach) & (x[:, None] < slots[..., 1] + reach)).any(-1)
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        hind_air = (force < CONTACT_N).all(-1)
        return (near & hind_air & _gap_mask(env)).float()


def _gap_or_stair(env) -> torch.Tensor:
    cols = family_column_masks(env)
    names = [n for n in cols if n == "platform_gap" or n.startswith("pyramid_stairs")]
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    col = env.scene.terrain.terrain_types.long()
    for name in names:
        mask |= torch.isin(col, cols[name])
    return mask


def overspeed(env, margin: float = 0.2):
    """Per second: forward body speed above the forward command + margin (m/s) on gap and stair tiles."""
    command = env.command_manager.get_command("base_velocity")[:, 0]
    speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    excess = (speed - command - margin).clamp(min=0) * (command > 0)
    return excess * _gap_or_stair(env)


class StairStall(ManagerTermBase):
    """Per second once a forward command (> 0.3 m/s) on the stair band has made < 0.05 m/s progress for > 2 s."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stairs = _StairGeometry(env)
        self.timer = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.timer[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_s: float = 2.0):
        robot = env.scene["robot"]
        col = env.scene.terrain.terrain_types.long()
        rel = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        cheb, _ = chebyshev_frame(rel)
        band = self.stairs.is_stair[col] & (cheb > self.stairs.inner[col] + 0.15) & (cheb < self.stairs.outer[col] - 0.10)
        command = env.command_manager.get_command("base_velocity")[:, 0]
        stuck = band & (command > 0.3) & (robot.data.root_lin_vel_b[:, 0] < 0.05)
        self.timer = torch.where(stuck, self.timer + env.step_dt, torch.zeros_like(self.timer))
        return (self.timer > hold_s).float()


def add_v21_terms(cfg) -> None:
    """v2 terms + the v2.1 command and terms (shared by the BIVT-Ray and GAST v2.1 teachers)."""
    from isaaclab.managers import RewardTermCfg

    add_v2_terms(cfg)
    cfg.commands.base_velocity.class_type = V21VelocityCommand
    cfg.rewards.gap_foothold_margin.weight = V21_WEIGHTS["gap_foothold_margin"]
    hind = SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"])
    cfg.rewards.gap_hind_hop = RewardTermCfg(func=GapHindHop, weight=V21_WEIGHTS["gap_hind_hop"],
                                             params={"sensor_cfg": hind})
    cfg.rewards.overspeed = RewardTermCfg(func=overspeed, weight=V21_WEIGHTS["overspeed"], params={})
    cfg.rewards.stair_stall = RewardTermCfg(func=StairStall, weight=V21_WEIGHTS["stair_stall"], params={})


def add_student_v21_env(cfg) -> None:
    """Distillation env of a v2.1 teacher: the v2 student env (26 cm gaps, stair disturbance) + v2.1 commands."""
    from gd_lab.mdp.gap_stair_v2 import add_student_v2_env

    add_student_v2_env(cfg)
    cfg.commands.base_velocity.class_type = V21VelocityCommand
