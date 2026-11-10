"""External mode command for the handstand skill, sampled like an operator / gdm signal.

command = [request, abort, time since the last change / 3 s (clipped to 1)]
  request 1  -> go to (and hold) the handstand
  request 0  -> four-leg stand (before ENTER, after EXIT, and after ABORT)
  abort   1  -> the last change was an ABORT (fast return); stays 1 until the episode ends

The term also owns the S1 curriculum state (target pitch, assist scale) and the per-episode success bookkeeping,
because both depend on the request schedule.
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FRONT_FEET = ("FL_foot", "FR_foot")
HIND_FEET = ("RL_foot", "RR_foot")


def pitch_from_gravity(pg: torch.Tensor) -> torch.Tensor:
    """Nose-down pitch (rad) from the body-frame gravity direction; 0 = level, +pi/2 = handstand."""
    return torch.atan2(pg[:, 0], -pg[:, 2])


class SkillModeCommand(CommandTerm):
    cfg: "SkillModeCommandCfg"

    def __init__(self, cfg: "SkillModeCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        inf = float("inf")
        self._cmd = torch.zeros(n, 3, device=dev)
        self.t_on = torch.full((n,), inf, device=dev)
        self.t_off = torch.full((n,), inf, device=dev)
        self.is_abort = torch.zeros(n, dtype=torch.bool, device=dev)
        self.abort_kind = torch.zeros(n, dtype=torch.long, device=dev)   # 0 none, 1 while entering, 2 while holding
        # curriculum (shared by all envs)
        self.level = 0
        self.levels = [tuple(x) for x in cfg.levels]
        self.success_ema = 0.0
        self.episodes_at_level = 0
        # per-episode bookkeeping
        self.hold_timer = torch.zeros(n, device=dev)
        self.best_hold = torch.zeros(n, device=dev)
        self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.returned = torch.zeros(n, dtype=torch.bool, device=dev)
        # set by the start_in_handstand reset event: the episode began in a handstand (excluded from the curriculum)
        self.started_in_pose = torch.zeros(n, dtype=torch.bool, device=dev)
        # Isaac applies reset events BEFORE the command manager resets, so the event only leaves the request
        # duration here (NaN = none) and _resample_command applies it.
        self.pending_pose_request = torch.full((n,), float("nan"), device=dev)
        # S2 walk window inside the request (walk flag = stepping requested; velocity command stays 0 in S2)
        self.t_walk_on = torch.full((n,), float("inf"), device=dev)
        self.t_walk_off = torch.full((n,), float("inf"), device=dev)
        self.walk = torch.zeros(n, device=dev)
        self.walk_start_xy = torch.zeros(n, 2, device=dev)
        self.metrics["front_steps"] = torch.zeros(n, device=dev)
        self.metrics["walk_drift_m"] = torch.zeros(n, device=dev)
        self._robot = env.scene["robot"]
        self._contact = env.scene.sensors["contact_forces"]
        self._front_ids = self._contact.find_bodies(list(FRONT_FEET), preserve_order=True)[0]
        self._hind_ids = self._contact.find_bodies(list(HIND_FEET), preserve_order=True)[0]
        self.metrics["request_on"] = torch.zeros(n, device=dev)
        # per-episode maxima (CommandTerm.reset logs the mean over finished envs, then zeroes them)
        self.metrics["max_pitch_deg"] = torch.zeros(n, device=dev)
        self.metrics["best_hold_s"] = torch.zeros(n, device=dev)

    def __str__(self) -> str:
        return f"SkillModeCommand: levels={self.levels}"

    # -- curriculum state ---------------------------------------------------
    @property
    def target_pitch(self) -> float:
        return math.radians(self.levels[self.level][0])

    @property
    def assist_scale(self) -> float:
        return float(self.levels[self.level][1])

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    # -- contacts / pose helpers (also used by the rewards) -----------------
    def feet_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        f = self._contact.data.net_forces_w[:, :, 2]
        return f[:, self._front_ids] > 1.0, f[:, self._hind_ids] > 1.0

    def walking(self) -> torch.Tensor:
        return self.walk > 0.5

    def request(self) -> torch.Tensor:
        return self._cmd[:, 0] > 0.5

    def aborted(self) -> torch.Tensor:
        return self._cmd[:, 1] > 0.5

    def had_request(self) -> torch.Tensor:
        t = self._env.episode_length_buf.float() * self._env.step_dt
        return t >= self.t_on

    # -- CommandTerm API -------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._account_finished(ids)
        n = len(ids)
        cfg = self.cfg
        u = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)  # noqa: E731
        t_on = u(*cfg.enter_time_range)
        hold = u(*cfg.request_duration_range)
        kind = torch.zeros(n, dtype=torch.long, device=self.device)
        r = torch.rand(n, device=self.device)
        kind[r < cfg.abort_entering_prob] = 1
        kind[(r >= cfg.abort_entering_prob) & (r < cfg.abort_entering_prob + cfg.abort_holding_prob)] = 2
        t_off = t_on + hold
        t_off = torch.where(kind == 1, t_on + u(*cfg.abort_entering_after), t_off)
        t_off = torch.where(kind == 2, t_on + u(*cfg.abort_holding_after), t_off)
        none = torch.rand(n, device=self.device) < cfg.no_request_prob
        t_on[none] = float("inf")
        t_off[none] = float("inf")
        kind[none] = 0
        self.t_on[ids], self.t_off[ids] = t_on, t_off
        self.abort_kind[ids] = kind
        self.is_abort[ids] = kind > 0
        for buf in (self.hold_timer, self.best_hold):
            buf[ids] = 0.0
        self.reached[ids] = False
        self.returned[ids] = False
        pend = self.pending_pose_request[ids]
        pose = ~torch.isnan(pend)
        self.started_in_pose[ids] = pose
        if pose.any():
            pids = ids[pose]
            self.t_on[pids] = 0.0
            self.t_off[pids] = pend[pose]
            self.abort_kind[pids] = 0
            self.is_abort[pids] = False
            self.pending_pose_request[pids] = float("nan")
        # walk window (S2): after the robot is up, for a while, inside the request window
        self.t_walk_on[ids] = float("inf")
        self.t_walk_off[ids] = float("inf")
        if cfg.walk_prob > 0:
            walk = (torch.rand(n, device=self.device) < cfg.walk_prob) & torch.isfinite(self.t_on[ids]) \
                & (self.abort_kind[ids] != 1)
            delay = torch.where(self.started_in_pose[ids], u(*cfg.walk_delay_from_pose), u(*cfg.walk_delay))
            on_t = self.t_on[ids] + delay
            off_t = torch.minimum(on_t + u(*cfg.walk_duration), self.t_off[ids])
            self.t_walk_on[ids] = torch.where(walk, on_t, torch.full_like(on_t, float("inf")))
            self.t_walk_off[ids] = torch.where(walk, off_t, torch.full_like(off_t, float("inf")))

    def _update_command(self):
        t = self._env.episode_length_buf.float() * self._env.step_dt
        on = (t >= self.t_on) & (t < self.t_off)
        after = t >= self.t_off
        last = torch.where(after, self.t_off, torch.where(t >= self.t_on, self.t_on, torch.zeros_like(t)))
        self._cmd[:, 0] = on.float()
        self._cmd[:, 1] = (after & self.is_abort).float()
        self._cmd[:, 2] = ((t - last) / 3.0).clamp(0.0, 1.0)
        walk = on & (t >= self.t_walk_on) & (t < self.t_walk_off)
        start = walk & (self.walk < 0.5)
        if start.any():
            self.walk_start_xy[start] = self._robot.data.root_pos_w[start, :2]
        self.walk = walk.float()
        self._track(on, after)

    def _update_metrics(self):
        self.metrics["request_on"] = self._cmd[:, 0].clone()
        pitch = torch.rad2deg(pitch_from_gravity(self._robot.data.projected_gravity_b))
        on = self._cmd[:, 0] > 0.5
        self.metrics["max_pitch_deg"] = torch.where(on, torch.maximum(self.metrics["max_pitch_deg"], pitch),
                                                    self.metrics["max_pitch_deg"])
        self.metrics["best_hold_s"] = self.best_hold.clone()
        walking = self.walk > 0.5
        if walking.any():
            first = self._contact.compute_first_contact(self._env.step_dt)[:, self._front_ids].float().sum(dim=1)
            self.metrics["front_steps"] += first * walking.float()
            drift = (self._robot.data.root_pos_w[:, :2] - self.walk_start_xy).norm(dim=1)
            self.metrics["walk_drift_m"] = torch.where(walking, torch.maximum(self.metrics["walk_drift_m"], drift),
                                                       self.metrics["walk_drift_m"])

    # -- success bookkeeping ---------------------------------------------------
    def _track(self, on: torch.Tensor, after: torch.Tensor):
        pg = self._robot.data.projected_gravity_b
        pitch = pitch_from_gravity(pg)
        front, hind = self.feet_contact()
        ok = (pitch - self.target_pitch).abs() < self.cfg.success_pitch_tol
        if self.target_pitch >= math.radians(60):
            ok &= ~hind.any(dim=1)
        ok &= on
        dt = self._env.step_dt
        self.hold_timer = torch.where(ok, self.hold_timer + dt, torch.zeros_like(self.hold_timer))
        self.best_hold = torch.maximum(self.best_hold, self.hold_timer)
        self.reached |= self.hold_timer >= self.cfg.success_hold_s
        upright = pg[:, 2] < -0.95
        self.returned |= after & upright & front.all(dim=1) & hind.all(dim=1)

    def _account_finished(self, ids: torch.Tensor):
        """Episodes that asked for a handstand and were not aborted while entering feed the curriculum."""
        valid = torch.isfinite(self.t_on[ids]) & (self.abort_kind[ids] != 1) & ~self.started_in_pose[ids]
        t = self._env.episode_length_buf[ids].float() * self._env.step_dt
        # every episode that reached ENTER counts; one that fell before holding long enough is a failure (counting only
        # the survivors of the first 2 s inflated the rate and promoted v1 to 90 deg / assist 0.6 at a 72 % fall rate)
        valid &= t >= self.t_on[ids]
        k = int(valid.sum())
        if k == 0:
            return
        rate = float(self.reached[ids][valid].float().mean())
        a = min(1.0, k / self.cfg.ema_episodes)
        self.success_ema = (1 - a) * self.success_ema + a * rate
        self.episodes_at_level += k


@configclass
class SkillModeCommandCfg(CommandTermCfg):
    class_type: type = SkillModeCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)   # resampled only at reset
    enter_time_range: tuple[float, float] = (0.5, 3.0)
    request_duration_range: tuple[float, float] = (6.0, 13.0)     # includes ~2-3 s to get up
    abort_entering_prob: float = 0.15
    abort_entering_after: tuple[float, float] = (0.4, 2.0)
    abort_holding_prob: float = 0.10
    abort_holding_after: tuple[float, float] = (3.0, 9.0)
    no_request_prob: float = 0.15
    success_pitch_tol: float = 0.26          # 15 deg
    success_hold_s: float = 2.0
    ema_episodes: float = 2000.0
    # (target pitch deg, assist scale); promoted when the success EMA passes promote_rate
    levels: list = MISSING
    promote_rate: float = 0.7
    # S2 walk window (walk_prob 0 = S1 behaviour)
    walk_prob: float = 0.0
    walk_delay: tuple[float, float] = (2.5, 4.0)
    walk_delay_from_pose: tuple[float, float] = (0.5, 2.0)
    walk_duration: tuple[float, float] = (3.0, 8.0)
    min_episodes_per_level: int = 4000
