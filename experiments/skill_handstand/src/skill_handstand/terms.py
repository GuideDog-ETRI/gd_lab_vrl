"""Observation, reward, assist and curriculum terms for the handstand skill (stage S1).

Rewards are gated by the external mode command, never by a clock: request on -> handstand terms,
request off (before ENTER, after EXIT / ABORT) -> upright four-leg terms.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .commands import SkillModeCommand, pitch_from_gravity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CMD = "skill_mode"


def _cmd(env: "ManagerBasedRLEnv") -> SkillModeCommand:
    return env.command_manager.get_term(CMD)


def _target_gravity(pitch: float, device) -> torch.Tensor:
    """Body-frame gravity direction of a trunk pitched nose-down by ``pitch``."""
    return torch.tensor([math.sin(pitch), 0.0, -math.cos(pitch)], device=device)


# -- observations ------------------------------------------------------------------------------------------------
def skill_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _cmd(env).command


def skill_walk(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """S2: stepping requested (1) while in the handstand; the velocity command stays 0 in S2."""
    return _cmd(env).walk.unsqueeze(1)


def skill_curriculum(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Critic-only: current target pitch (/ pi/2) and assist scale; the assist wrench is part of the dynamics."""
    c = _cmd(env)
    v = torch.tensor([c.target_pitch / (math.pi / 2), c.assist_scale], device=env.device)
    return v.expand(env.num_envs, 2)


# -- rewards -----------------------------------------------------------------------------------------------------
def hs_orientation(env: "ManagerBasedRLEnv", sigma: float = 0.25) -> torch.Tensor:
    """Request on: trunk orientation toward the current target pitch."""
    c = _cmd(env)
    pg = env.scene["robot"].data.projected_gravity_b
    err = (pg - _target_gravity(c.target_pitch, env.device)).square().sum(dim=1)
    return torch.exp(-err / sigma) * c.request().float()


def hs_front_support(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Request on and past half the target pitch: both front feet down, both hind feet up."""
    c = _cmd(env)
    front, hind = c.feet_contact()
    pitch = pitch_from_gravity(env.scene["robot"].data.projected_gravity_b)
    gate = c.request() & (pitch > 0.5 * c.target_pitch) & ~c.walking()   # stepping lifts a front foot by design
    return (front.all(dim=1) & ~hind.any(dim=1) & gate).float()


def hs_hind_contact(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Request on and pitched more than 30 deg: number of hind feet still on the ground."""
    c = _cmd(env)
    _, hind = c.feet_contact()
    pitch = pitch_from_gravity(env.scene["robot"].data.projected_gravity_b)
    gate = c.request() & (pitch > math.radians(30.0))
    return hind.float().sum(dim=1) * gate.float()


def hs_front_load(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Request on and still below the target: share of the vertical ground force carried by the front feet (v3).
    Shifting the weight onto the front feet is the prerequisite of the lift-off that v2 never found from a stand."""
    c = _cmd(env)
    f = c._contact.data.net_forces_w[:, :, 2].clamp_min(0.0)
    front, hind = f[:, c._front_ids].sum(dim=1), f[:, c._hind_ids].sum(dim=1)
    share = front / (front + hind + 1.0)
    pitch = pitch_from_gravity(env.scene["robot"].data.projected_gravity_b)
    return share * (c.request() & (pitch < c.target_pitch - 0.17)).float()


def hs_hold(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Request on: inside the success window (pitch within tolerance, hind feet up from 60 deg targets)."""
    c = _cmd(env)
    return (c.hold_timer > 0).float()


def hs_upright(env: "ManagerBasedRLEnv", sigma: float = 0.25) -> torch.Tensor:
    """Request off: level trunk with all four feet on the ground."""
    c = _cmd(env)
    pg = env.scene["robot"].data.projected_gravity_b
    level = torch.exp(-(pg - _target_gravity(0.0, env.device)).square().sum(dim=1) / sigma)
    front, hind = c.feet_contact()
    feet = (front.float().sum(dim=1) + hind.float().sum(dim=1)) / 4.0
    return 0.5 * (level + level * feet) * (~c.request()).float()


def hs_still_when_idle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """No request yet in this episode: trunk motion (stand still until ENTER)."""
    c = _cmd(env)
    d = env.scene["robot"].data
    motion = d.root_lin_vel_b[:, :2].square().sum(dim=1) + 0.2 * d.root_ang_vel_b.square().sum(dim=1)
    return motion * (~c.had_request()).float()


def torque_saturation(env: "ManagerBasedRLEnv", fraction: float = 0.9,
                      hip_limit: float = 93.6, knee_limit: float = 126.0) -> torch.Tensor:
    """Number of joints whose commanded torque exceeds ``fraction`` of the motor limit."""
    robot = env.scene["robot"]
    if not hasattr(env, "_hs_torque_limit"):
        names = robot.data.joint_names
        env._hs_torque_limit = torch.tensor([knee_limit if "KNEE" in n else hip_limit for n in names],
                                            device=env.device)
    return (robot.data.computed_torque.abs() > fraction * env._hs_torque_limit).float().sum(dim=1)


# -- S2b: overshoot / swing height ---------------------------------------------------------------------------------
def hs_overshoot(env: "ManagerBasedRLEnv", margin: float = 0.17) -> torch.Tensor:
    """Request on: pitch beyond the target by more than ``margin`` (rad); S2 v1 tipped over the top in MuJoCo."""
    c = _cmd(env)
    pitch = pitch_from_gravity(env.scene["robot"].data.projected_gravity_b)
    return (pitch - c.target_pitch - margin).clamp_min(0.0) * c.request().float()


def hs_pitch_rate_near_vertical(env: "ManagerBasedRLEnv", from_deg: float = 70.0) -> torch.Tensor:
    """Request on and past ``from_deg``: squared body pitch rate (arrive at the vertical slowly)."""
    c = _cmd(env)
    d = env.scene["robot"].data
    pitch = pitch_from_gravity(d.projected_gravity_b)
    return d.root_ang_vel_b[:, 1].square() * (c.request() & (pitch > math.radians(from_deg))).float()


def hs_swing_height(env: "ManagerBasedRLEnv", target: float = 0.05, radius: float = 0.03) -> torch.Tensor:
    """Walking: lifted front feet rewarded up to ``target`` m above the ground (flat terrain)."""
    c = _cmd(env)
    robot = env.scene["robot"]
    if not hasattr(env, "_hs_front_body_ids"):
        env._hs_front_body_ids = robot.find_bodies(["FL_foot", "FR_foot"], preserve_order=True)[0]
    h = (robot.data.body_pos_w[:, env._hs_front_body_ids, 2] - env.scene.env_origins[:, 2:3] - radius).clamp(0.0, target)
    front, _ = c.feet_contact()
    return ((h / target) * (~front).float()).sum(dim=1) * c.walking().float()


# -- S2 stepping in place --------------------------------------------------------------------------------------------
def hs_step_air_time(env: "ManagerBasedRLEnv", min_air: float = 0.15, max_air: float = 0.4) -> torch.Tensor:
    """Walking: on each front-foot touchdown, reward the swing time above ``min_air`` (capped at ``max_air``)."""
    c = _cmd(env)
    first = c._contact.compute_first_contact(env.step_dt)[:, c._front_ids]
    air = c._contact.data.last_air_time[:, c._front_ids]
    return ((air.clamp(max=max_air) - min_air) * first).sum(dim=1) * c.walking().float()


def hs_front_alternation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Walking: +1 with exactly one front foot down (single support), -1 with both front feet in the air."""
    c = _cmd(env)
    front, hind = c.feet_contact()
    n = front.float().sum(dim=1)
    return ((n == 1).float() - (n == 0).float()) * c.walking().float() * (~hind.any(dim=1)).float()


def hs_stay_in_place(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Walking with a zero velocity command: horizontal trunk velocity (world) and yaw rate."""
    c = _cmd(env)
    d = env.scene["robot"].data
    return (d.root_lin_vel_w[:, :2].square().sum(dim=1) + 0.2 * d.root_ang_vel_w[:, 2].square()) * c.walking().float()


def hs_walk_drift(env: "ManagerBasedRLEnv", deadband: float = 0.05) -> torch.Tensor:
    """Walking with a zero velocity command: horizontal distance from where the walk started, beyond ``deadband`` m (S2c)."""
    c = _cmd(env)
    d = (env.scene["robot"].data.root_pos_w[:, :2] - c.walk_start_xy).norm(dim=1)
    return (d - deadband).clamp_min(0.0) * c.walking().float()


def target_step_excess(env: "ManagerBasedRLEnv", scale: float = 0.25, max_step_deg: float = 10.0) -> torch.Tensor:
    """Joint-target change per policy step beyond ``max_step_deg`` (rad, summed): the drives follow abrupt targets
    with saturated torque (MuJoCo S1: knees up to 36 deg per 10 ms)."""
    am = env.action_manager
    step = (am.action - am.prev_action).abs() * scale
    return (step - math.radians(max_step_deg)).clamp_min(0.0).sum(dim=1)


# -- assist wrench (event, every policy step) ----------------------------------------------------------------------
def assist_wrench(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None, kp: float = 150.0, kd: float = 15.0,
                  lift_fraction: float = 0.3) -> None:
    """Virtual trunk wrench toward the commanded orientation, scaled by the curriculum (0 at the last levels).

    Torque (body frame) = scale * (kp * (target x g_b) - kd * w_b): rotates the body-frame gravity toward the target
    (handstand pitch while requested, level otherwise). Lift = scale * lift_fraction * m g upward.
    """
    c = _cmd(env)
    robot = env.scene["robot"]
    scale = c.assist_scale
    trunk = env._hs_trunk_id if hasattr(env, "_hs_trunk_id") else robot.find_bodies("trunk")[0]
    env._hs_trunk_id = trunk
    n = env.num_envs
    if scale <= 0.0:
        if getattr(env, "_hs_assist_on", False):
            robot.set_external_force_and_torque(torch.zeros(n, 1, 3, device=env.device),
                                                torch.zeros(n, 1, 3, device=env.device), body_ids=trunk)
            env._hs_assist_on = False
        return
    pg = robot.data.projected_gravity_b
    req = c.request().float().unsqueeze(1)
    target = req * _target_gravity(c.target_pitch, env.device) + (1 - req) * _target_gravity(0.0, env.device)
    torque = scale * (kp * torch.cross(target, pg, dim=1) - kd * robot.data.root_ang_vel_b)
    if not hasattr(env, "_hs_total_mass"):
        env._hs_total_mass = robot.root_physx_view.get_masses().sum(dim=1).to(env.device)
    lift = scale * lift_fraction * env._hs_total_mass * 9.81
    force = -lift.unsqueeze(1) * pg   # world +z expressed in the body frame
    robot.set_external_force_and_torque(force.unsqueeze(1), torque.unsqueeze(1), body_ids=trunk)
    env._hs_assist_on = True


# -- start in a handstand (reset event, v2) ------------------------------------------------------------------------
_POSES = None


def _start_poses(device):
    global _POSES
    if _POSES is None:
        import json
        from pathlib import Path
        table = json.loads((Path(__file__).with_name("start_poses.json")).read_text())
        _POSES = dict(order=table["joint_order"],
                      pitch=torch.tensor([r["pitch_deg"] for r in table["poses"]], device=device, dtype=torch.float),
                      q=torch.tensor([r["joint_pos"] for r in table["poses"]], device=device),
                      z=torch.tensor([r["base_z"] for r in table["poses"]], device=device))
    return _POSES


def start_in_handstand(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, fraction: float = 0.3,
                       request_range: tuple[float, float] = (3.0, 10.0)) -> None:
    """A ``fraction`` of resets start in a statically balanced front-feet pose (URDF statics table, COM over the
    front feet) with the request already on, so the policy experiences holding and returning from the start.
    Only poses up to the current target pitch are used (none while the target is below the table's 45 deg)."""
    if env_ids is None or len(env_ids) == 0:
        return
    c = _cmd(env)
    P = _start_poses(env.device)
    allowed = (P["pitch"] <= math.degrees(c.target_pitch) + 0.1).nonzero().squeeze(1)
    if allowed.numel() == 0:
        return
    ids = env_ids[torch.rand(len(env_ids), device=env.device) < fraction]
    if ids.numel() == 0:
        return
    row = allowed[torch.randint(0, allowed.numel(), (ids.numel(),), device=env.device)]
    robot = env.scene["robot"]
    pitch = torch.deg2rad(P["pitch"][row])
    yaw = torch.empty(ids.numel(), device=env.device).uniform_(-math.pi, math.pi)
    # q = yaw(z) * pitch(y), (w, x, y, z)
    cy, sy, cp, sp = torch.cos(yaw / 2), torch.sin(yaw / 2), torch.cos(pitch / 2), torch.sin(pitch / 2)
    quat = torch.stack([cy * cp, -sy * sp, cy * sp, sy * cp], dim=1)
    pos = env.scene.env_origins[ids].clone()
    pos[:, 2] += P["z"][row]
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=env.device), env_ids=ids)
    names = robot.data.joint_names
    cols = [P["order"].index(n) for n in names]
    q = P["q"][row][:, cols]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    # the mode command: request on from t = 0, released after request_range seconds (applied by the command reset,
    # which Isaac runs after the reset events)
    c.pending_pose_request[ids] = torch.empty(ids.numel(), device=env.device).uniform_(*request_range)


# -- curriculum ------------------------------------------------------------------------------------------------------
def handstand_levels(env: "ManagerBasedRLEnv", env_ids) -> dict:
    """Promote (target pitch, assist) when the success EMA passes the threshold; never demote."""
    c = _cmd(env)
    cfg = c.cfg
    if (c.level < len(c.levels) - 1 and c.episodes_at_level >= cfg.min_episodes_per_level
            and c.success_ema >= cfg.promote_rate):
        c.level += 1
        c.episodes_at_level = 0
        c.success_ema = 0.0
        print(f"[HANDSTAND] level {c.level}: target {c.levels[c.level][0]} deg, assist {c.levels[c.level][1]}",
              flush=True)
    return {"level": float(c.level), "target_pitch_deg": float(c.levels[c.level][0]),
            "assist_scale": float(c.levels[c.level][1]), "success_ema": float(c.success_ema)}
