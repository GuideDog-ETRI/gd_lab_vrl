#!/usr/bin/env python3
"""E2 trial: one model x one condition on the GAST-Narrow v2 training env, every env pinned to one test cell.

  apptainer exec ... python scripts/gast_narrow/e2/e2_trial.py --plan plan.json --model model_v2.pt --out <dir>

``plan.json`` (written by run_e2.py) holds the task (v2 or v24 env), the condition (LiDAR mode, rear Mid-360,
payload, pushes, stair disturbance, seed, seconds) and the list of cells. A cell pins its envs to one (terrain row,
column) and, for the retention terrain, to one command; narrow cells keep the v2 gdm-like command and pin the
requested margin m (and, on the v24 task, the passage direction: forward or tail first); turn cells (v24) pin the
margin, the turn direction, the target angle and the turn rate, with no pause window.
Each env contributes its FIRST episode only (no curriculum, the terrain levels stay fixed). The model must be
v2-shaped (convert_models.py --to v2): 6566 / 1748 are widened with zero columns, so they act exactly as trained.

Result: <out>/result.json = {"condition", "cells": [cell + tile info], "rows": [one dict per env]}.
"""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gast"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--plan", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
from vrl_runtime import prepare_vrl_runtime  # noqa: E402
prepare_vrl_runtime(args)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import gd_lab  # noqa: E402,F401
from gd_lab.core import registry  # noqa: E402
from gd_lab.core.experiments import training_arm_overrides  # noqa: E402
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking  # noqa: E402

plan = json.loads(Path(args.plan).read_text())
cond = plan["condition"]
cells = plan["cells"]
SEED = int(plan["seed"])
SECONDS = float(plan["seconds"])
TASK_KIND = plan.get("task", "v2")  # v2: the GAST-Narrow v2 env (44 columns); v24: v2.4 (58 columns, turn tiles)
TASKS = {"v2": ("GastNarrowV2", "gd_lab.gast_narrow.tasks_v2:GastNarrowV2TeacherCfg",
                "gd_lab.gast_narrow.model:GastNarrowV2RunnerCfg"),
         "v24": ("GastNarrowV24", "gd_lab.gast_narrow.tasks_v24:GastNarrowV24TeacherCfg",
                 "gd_lab.gast_narrow.tasks_v24:GastNarrowV24RunnerCfg")}
task_name, env_entry, agent_entry = TASKS[TASK_KIND]
TASK = registry.register_task(task=task_name, robot="Rbq10", method="Dreamwaq", env_cfg=env_entry, agent_cfg=agent_entry)
from gd_lab.gast_narrow import geometry, mdp_v2  # noqa: E402
from gd_lab.gast_narrow.state import narrow_state  # noqa: E402
from gd_lab.mdp.terrain_families import family_column_masks  # noqa: E402

V24 = TASK_KIND == "v24"
if V24:
    from gd_lab.gast_narrow import mdp_turn  # noqa: E402
STAIR_OUT_R, STAIR_IN_R = float(plan.get("stair_outer_r", 3.6)), float(plan.get("stair_inner_r", 1.0))
GAP_LEAVE_R = float(plan.get("gap_leave_r", 3.8))  # Chebyshev radius at which a gap env leaves its 8 m tile
LIDAR_ON = {"both": (True, True), "ouster": (True, False), "mid360": (False, True), "off": (False, False)}
LIDAR_ON[False] = LIDAR_ON["off"]  # YAML 1.1 reads a bare `off` as false
UP_OUTWARD = ("_inv",)  # *_inv tiles: the pit is in the centre, walking outward climbs; the others descend outward
MODE_TURN = 3


def stair_motion_outward(family, direction):
    """'up'/'down' on a stair/slope family -> walk outward (True) or inward (False) from the tile centre."""
    inv = any(tag in family for tag in UP_OUTWARD)
    if direction in ("outward", "inward"):
        return direction == "outward"
    return (direction == "up") == inv


# ---------------------------------------------------------------- env config
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
for item in training_arm_overrides("4"):  # training timing/gains (train_teacher applies them via Hydra)
    key, value = item.split("=", 1)
    if not key.startswith("env."):
        continue
    *path, leaf = key[4:].split(".")
    target = env_cfg
    for part in path:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    value = yaml.safe_load(value)
    if isinstance(target, dict):
        target[leaf] = value
    else:
        setattr(target, leaf, value)
num_envs = sum(int(c["envs"]) for c in cells)
env_cfg.scene.num_envs = num_envs
env_cfg.seed = SEED
env_cfg.scene.terrain.terrain_generator.seed = SEED  # same tiles (widths, gaps) for every model
env_cfg.episode_length_s = SECONDS
env_cfg.curriculum.terrain_levels = None               # levels stay where the cells put them
if not cond.get("push", False):
    env_cfg.events.push_robot = None
if not cond.get("stair_disturbance", False):
    env_cfg.rewards.stair_handle_disturbance.params["rate_hz"] = 0.0  # never starts a hip-handle force
payload = cond.get("payload_kg")
if payload is not None:
    env_cfg.events.randomize_payload.params["payload_masses"] = (float(payload),)
    env_cfg.events.randomize_payload.params["payload_probabilities"] = (1.0,)


# Per-env cell spec (filled after the env exists), read by the spawn event and the command hook.
class Spec:
    pass


spec = Spec()


def e2_spawn(env, env_ids):
    """Reset event after reset_narrow_v2: inward stair starts at the tile edge; narrow margin m pinned."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not hasattr(spec, "mode") or not len(ids):
        return
    command = env.command_manager.get_term("base_velocity")
    if hasattr(command, "_e2_yaw"):
        command._e2_yaw[ids] = float("nan")  # new episode: the first command resample picks the straight heading
    inward = ids[(spec.mode[ids] == 2) & ~spec.outward[ids]]
    if len(inward):
        robot = env.scene["robot"]
        n = len(inward)
        side = torch.randint(0, 4, (n,), device=env.device)
        along = (torch.rand(n, device=env.device) * 2 - 1) * 1.0
        r = STAIR_OUT_R + 0.1
        x = torch.where(side == 0, r, torch.where(side == 1, -r, along))
        y = torch.where(side == 2, r, torch.where(side == 3, -r, along))
        yaw = torch.where(side == 0, math.pi, torch.where(side == 1, 0.0, torch.where(side == 2, -math.pi / 2,
                                                                                         math.pi / 2)))
        pose = robot.data.default_root_state[inward, :7].clone()
        origin = env.scene.env_origins[inward]
        pose[:, 0], pose[:, 1] = origin[:, 0] + x, origin[:, 1] + y
        top = torch.stack((pose[:, 0], pose[:, 1], origin[:, 2] + 5.0), -1).contiguous()
        down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(n, 3).contiguous()
        dist = spec.cast(top, down, 20.0)
        ground = torch.where(torch.isfinite(dist), top[:, 2] - dist, origin[:, 2])
        pose[:, 2] = ground + pose[:, 2] + 0.03   # default root height above the local ground at the edge
        pose[:, 3], pose[:, 4:6], pose[:, 6] = torch.cos(yaw / 2), 0.0, torch.sin(yaw / 2)
        robot.write_root_pose_to_sim(pose, env_ids=inward)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=inward)
        state = getattr(env, "_narrow_state", None)
        if state is not None:
            state.reset_counter += 1
    pinned = ids[~torch.isnan(spec.margin[ids]) & (spec.mode[ids] != MODE_TURN)]
    if len(pinned):
        state = narrow_state(env)
        episode = mdp_v2.narrow_episode(env)
        m = spec.margin[pinned]
        width = state.width[pinned]
        dead_end = state.kind[pinned] == geometry.KINDS["dead_end"]
        threshold = mdp_v2.ROBOT_WIDTH + 2 * m
        episode.margin[pinned] = m
        episode.blocked[pinned] = (dead_end | (width < threshold)) & state.narrow[pinned]
        episode.slack[pinned] = torch.where(state.narrow[pinned], width - threshold, torch.zeros_like(m))
    if V24:
        e2_spawn_v24(env, ids)


def e2_spawn_v24(env, ids):
    """v2.4 tiles. Turn cells: the margin, turnable (from the ACTUAL tile radius, the training rule), the target
    angle and the direction replace reset_turn's draws (the start pose stays as reset_turn placed it). Narrow cells:
    the passage direction replaces reset_narrow_back's 25 % draw (spawn yaw turned back where it differs)."""
    state = narrow_state(env)
    episode = mdp_v2.narrow_episode(env)
    turn = mdp_turn.turn_episode(env)
    tids = ids[spec.mode[ids] == MODE_TURN]
    if len(tids):
        m = spec.margin[tids]
        slack = state.width[tids] - (mdp_turn.R_SWEEP + m)
        episode.margin[tids] = m
        episode.blocked[tids] = False
        episode.slack[tids] = slack
        turn.turnable[tids] = slack >= 0
        turn.target[tids] = spec.turn_angle[tids]
        turn.direction[tids] = spec.turn_dir[tids]
    nids = ids[spec.mode[ids] == 0]
    if len(nids):
        back = mdp_turn.narrow_back(env)
        eligible = torch.zeros(len(nids), dtype=torch.bool, device=env.device)
        for k in mdp_turn.BACK_KINDS:
            eligible |= state.kind[nids] == k
        want = spec.back[nids] & eligible & ~episode.blocked[nids]
        flip = nids[want != back[nids]]
        if len(flip):
            robot = env.scene["robot"]
            pose = torch.cat((robot.data.root_pos_w[flip], robot.data.root_quat_w[flip]), -1).clone()
            w, z = pose[:, 3].clone(), pose[:, 6].clone()  # yaw-only quaternion: yaw + pi (as reset_narrow_back)
            pose[:, 3], pose[:, 6] = -z, w
            robot.write_root_pose_to_sim(pose, env_ids=flip)
            state.reset_counter += 1
        back[nids] = want


env_cfg.events.e2_spawn = EventTermCfg(func=e2_spawn, mode="reset")
env_cfg.commands.base_velocity.debug_vis = False  # marker USDs come from S3 (hangs on offline compute nodes)

env = gym.make(TASK, cfg=env_cfg)
ensure_prev_prev_action_tracking(env.unwrapped)
raw = env.unwrapped
device = raw.device
terrain = raw.scene.terrain
gen = terrain.cfg.terrain_generator
families = family_column_masks(raw)

# ---------------------------------------------------------------- cells -> envs
n = num_envs
spec.mode = torch.zeros(n, dtype=torch.long, device=device)      # 0 narrow / free, 1 gap, 2 radial (stairs, slopes)
spec.speed = torch.zeros(n, device=device)
spec.backward = torch.zeros(n, dtype=torch.bool, device=device)
spec.outward = torch.ones(n, dtype=torch.bool, device=device)
spec.margin = torch.full((n,), float("nan"), device=device)
spec.back = torch.zeros(n, dtype=torch.bool, device=device)        # v24 narrow cells: tail first
spec.turn_dir = torch.ones(n, device=device)                        # v24 turn cells: +1 left, -1 right
spec.turn_angle = torch.zeros(n, device=device)                     # rad
spec.turn_rate = torch.zeros(n, device=device)                      # rad/s
lidar_term = raw.observation_manager._group_obs_class_term_cfgs["narrow_lidar"][0].func
N_SENSORS = lidar_term.sensor_on.shape[1]  # 2 (Ouster, Mid-360) on v2; 3 (+ rear Mid-360) on v24


def lidar_flags(mode, rear):
    """Condition / cell LiDAR mode (+ the rear Mid-360 switch on v24) -> per-sensor on flags."""
    flags = LIDAR_ON[mode]
    if N_SENSORS == 3:
        flags = flags + (bool(rear) and mode not in ("off", False),)
    return flags


spec.lidar = torch.tensor([lidar_flags(cond.get("lidar", "both"), cond.get("rear", True))],
                          device=device).expand(n, N_SENSORS).clone()
cell_of = torch.zeros(n, dtype=torch.long, device=device)
levels = torch.zeros(n, dtype=torch.long, device=device)
types_ = torch.zeros(n, dtype=torch.long, device=device)
narrow_meta = getattr(gen, "narrow_tile_metadata", None)
gap_meta = getattr(gen, "gap_tile_metadata", None)
cell_info = []
start = 0
for index, cell in enumerate(cells):
    k = int(cell["envs"])
    ids = torch.arange(start, start + k, device=device)
    start += k
    fam = cell["family"]
    cols = families[fam].tolist()
    info = dict(cell, index=index)
    if cell["suite"] in ("narrow", "dead_end") and "width" in cell:
        # tile of this family with the width closest to the target (row varies, the fixed seed fixes the widths)
        best = min(((abs(narrow_meta[r][c]["width"] - cell["width"]), r, c) for r in range(gen.num_rows)
                    for c in cols if narrow_meta[r][c] is not None))
        _, row, col = best
        rows_, cols_ = [row] * k, [col] * k
        info.update(row=row, col=col, actual_width=narrow_meta[row][col]["width"])
    elif cell["suite"] == "turn":
        # tile with the free radius closest to the target, outside the +/-TURN_BAND draw band of r = R_SWEEP + m
        # (the training never asks for a turn there); turnable follows from the actual radius as in training.
        threshold = mdp_turn.R_SWEEP + float(cell["margin"])
        tiles = [(r, c, narrow_meta[r][c]["width"]) for r in range(gen.num_rows) for c in cols
                 if narrow_meta[r][c] is not None]
        clear = [t for t in tiles if abs(t[2] - threshold) >= mdp_turn.TURN_BAND]
        row, col, radius = min(clear or tiles, key=lambda t: abs(t[2] - cell["radius"]))
        rows_, cols_ = [row] * k, [col] * k
        info.update(row=row, col=col, actual_radius=radius, slack=radius - threshold, turnable=radius >= threshold,
                    ambiguous=not clear)
    else:
        row = int(cell["level"])
        rows_ = [row] * k
        cols_ = [cols[i % len(cols)] for i in range(k)]
        info.update(row=row, cols=sorted(set(cols_)))
        if narrow_meta is not None and narrow_meta[row][cols[0]] is not None:
            info["actual_width"] = narrow_meta[row][cols[0]]["width"]
        if fam == "platform_gap" and gap_meta is not None:
            widths = [float(s[1] - s[0]) for c in sorted(set(cols_)) for s in gap_meta[row][c]["slots"]]
            info["gap_widths"] = widths
    levels[ids] = torch.tensor(rows_, device=device)
    types_[ids] = torch.tensor(cols_, device=device)
    cell_of[ids] = index
    if cell["suite"] == "gap":
        spec.mode[ids] = 1
        spec.speed[ids] = float(cell["speed"])
        spec.backward[ids] = cell["direction"] == "backward"
    elif cell["suite"] in ("stairs", "terrain"):
        spec.mode[ids] = 2
        spec.speed[ids] = float(cell["speed"])
        spec.outward[ids] = stair_motion_outward(fam, cell.get("direction", "outward"))
    elif cell["suite"] == "turn":
        spec.mode[ids] = MODE_TURN
        spec.turn_dir[ids] = -1.0 if cell["direction"] == "right" else 1.0
        spec.turn_angle[ids] = math.radians(float(cell["angle_deg"]))
        spec.turn_rate[ids] = float(cell["rate"])
    elif cell["suite"] in ("narrow", "dead_end"):
        spec.back[ids] = cell.get("direction", "forward") == "backward"
    if "margin" in cell:
        spec.margin[ids] = float(cell["margin"])
    if "lidar" in cell:
        spec.lidar[ids] = torch.tensor(lidar_flags(cell["lidar"], cell.get("rear", cond.get("rear", True))), device=device)
    cell_info.append(info)
assert start == n
assert V24 or not bool(((spec.mode == MODE_TURN) | spec.back).any()), "turn / backward cells need task: v24"

terrain.terrain_levels[:] = levels
terrain.terrain_types[:] = types_
raw.scene.env_origins[:] = terrain.terrain_origins[levels, types_]

# ---------------------------------------------------------------- hooks: LiDAR mode, commands
spec.cast = lidar_term.cast  # warp ray cast on the terrain mesh (inward stair spawn height)
_lidar_reset = lidar_term.reset


def lidar_reset(env_ids=None):
    _lidar_reset(env_ids)
    ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=device, dtype=torch.long)
    lidar_term.sensor_on[ids] = spec.lidar[ids]
    lidar_term.loss_step[ids] = -1  # no late loss: the condition fixes the sensors


lidar_term.reset = lidar_reset
cmd = raw.command_manager.get_term("base_velocity")
_resample = cmd._resample_command


def resample(self, env_ids):
    _resample(env_ids)
    ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    gap = ids[spec.mode[ids] == 1]
    if len(gap):  # Boarding already locked the heading to +/-x; fixed speed, direction by the sign of vx
        self.vel_command_b[gap, 0] = torch.where(spec.backward[gap], -spec.speed[gap], spec.speed[gap])
        self.vel_command_b[gap, 1] = 0.0
        self.backward[gap] = spec.backward[gap]
        self.is_heading_env[gap] = True
        self.is_standing_env[gap] = False
        self._pulse_active[gap] = False
    radial = ids[spec.mode[ids] == 2]
    if len(radial):
        rel = self.robot.data.root_pos_w[radial, :2] - raw.scene.env_origins[radial, :2]
        ax, ay = rel[:, 0].abs(), rel[:, 1].abs()
        along_x = ax >= ay
        out = torch.stack((torch.where(along_x, torch.sign(rel[:, 0]), 0.0),
                           torch.where(along_x, 0.0, torch.sign(rel[:, 1]))), -1)
        out = torch.where(out.abs().sum(-1, keepdim=True) > 0, out, torch.tensor([1.0, 0.0], device=device))
        yaw = torch.atan2(out[:, 1], out[:, 0]) + (~spec.outward[radial]).float() * math.pi
        if not hasattr(self, "_e2_yaw"):
            self._e2_yaw = torch.full((n,), float("nan"), device=device)
        keep = ~torch.isnan(self._e2_yaw[radial])  # keep the episode's first heading (straight run)
        yaw = torch.where(keep, self._e2_yaw[radial], yaw)
        self._e2_yaw[radial] = yaw
        self.heading_target[radial] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        self.is_heading_env[radial] = True
        self.vel_command_b[radial, 0] = spec.speed[radial]
        self.vel_command_b[radial, 1] = 0.0
        self.backward[radial] = False
        self.is_standing_env[radial] = False
        self._pulse_active[radial] = False


cmd._resample_command = types.MethodType(resample, cmd)
if V24:  # turn cells: the cell's turn rate, no pause window (the training draws 0.3..0.8 rad/s, pauses on 20 %)
    _reset_turn = cmd._reset_turn

    def reset_turn_pinned(ids):
        _reset_turn(ids)
        t = ids[spec.mode[ids] == MODE_TURN]
        if len(t):
            cmd.turn_rate[t] = spec.turn_rate[t]
            cmd.pause_at[t] = 1e6

    cmd._reset_turn = reset_turn_pinned

venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
agent_cfg.num_steps_per_env = 1  # evaluation never fills the PPO rollout buffer (100 steps x 31k envs = 53 GB)
from gd_lab.rl.runner import DreamwaqRunner  # noqa: E402
runner = DreamwaqRunner(venv, agent_cfg.to_dict(), log_dir=None, device=device)
saved = torch.load(args.model, map_location=device, weights_only=False)
runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
policy = runner.get_inference_policy(device=device)

# ---------------------------------------------------------------- per-episode recording
recorded = torch.zeros(n, dtype=torch.bool, device=device)
armed = {"on": False}
rows = [None] * n
radius_max = torch.zeros(n, device=device)
radius_min = torch.full((n,), 99.0, device=device)
passed_r = torch.zeros(n, dtype=torch.bool, device=device)
pass_t = torch.full((n,), -1.0, device=device)
err_v = torch.zeros(n, device=device)
err_w = torch.zeros(n, device=device)
moving_t = torch.zeros(n, device=device)
pitch_max = torch.zeros(n, device=device)
nose_up_max = torch.zeros(n, device=device)  # signed: front up (deg), asin(-g_x) as backward_nose_up
nose_t10 = torch.zeros(n, device=device)
nose_t15 = torch.zeros(n, device=device)
reach_t = torch.full((n,), -1.0, device=device)  # turn cells: episode time at which the target angle was reached
_reset_idx = raw._reset_idx


def term(name):
    return raw.reward_manager.get_term_cfg(name).func


def record_and_reset(self, env_ids):
    ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if armed["on"]:
        ids = ids[~recorded[ids]]
    if armed["on"] and len(ids):
        L = lambda t: t[ids].detach().cpu().tolist()  # noqa: E731
        base = L(self.termination_manager.get_term("base_contact"))
        timeout = L(self.termination_manager.time_outs)
        dur = L((self.episode_length_buf.float() * self.step_dt))
        sums = {k: L(v) for k, v in self.reward_manager._episode_sums.items()
                if k in ("platform_gap_foot_drop", "undesired_contacts", "stair_push_fall", "narrow_wall_proximity")}
        crossing = term("platform_gap_crossing")
        achieved, bypassed = L(crossing.achieved.any(dim=1)), L(crossing.bypassed)
        tracker = term("platform_gap_monitor").tracker
        outcomes = L(tracker.outcomes)
        clean_paid = L(tracker.paid.any(dim=1))
        mon = term("narrow_monitor")
        nar = {k: L(getattr(mon, k)) for k in ("passed", "held", "breached", "touch_time", "min_clearance")}
        ep = mdp_v2.narrow_episode(self)
        st = narrow_state(self)
        blocked, slack, margin, width = L(ep.blocked), L(ep.slack), L(ep.margin), L(st.width)
        extra = {k: L(v) for k, v in (("r_max", radius_max), ("r_min", radius_min), ("passed_r", passed_r),
                                         ("pass_t", pass_t), ("err_v", err_v), ("err_w", err_w),
                                         ("moving_t", moving_t), ("pitch_max_deg", pitch_max),
                                         ("nose_up_max_deg", nose_up_max), ("nose_up_t10", nose_t10),
                                         ("nose_up_t15", nose_t15))}
        if V24:  # turn tracker (mdp_turn): meaningful on turn cells, zeros elsewhere; backward = tail first
            turn = mdp_turn.turn_episode(self)
            drift_done = self.termination_manager.get_term("turn_drift")
            extra.update({k: L(v) for k, v in (
                ("turnable", turn.turnable), ("turned_deg", torch.rad2deg(turn.turned * turn.direction)),
                ("drift_max", turn.drift_max), ("reached", turn.reached), ("reach_t", reach_t),
                ("target_deg", torch.rad2deg(turn.target)), ("direction", turn.direction), ("drift_done", drift_done),
                ("backward", mdp_turn.narrow_back(self)))})
        cells_ = L(cell_of)
        for j, env_id in enumerate(ids.tolist()):
            row = {"env": env_id, "cell": cells_[j], "base_contact": bool(base[j]), "timeout": bool(timeout[j]),
                   "duration": round(dur[j], 3), "gap_achieved": bool(achieved[j]), "gap_bypassed": bool(bypassed[j]),
                   "gap_clean_paid": bool(clean_paid[j]), "gap_attempts": [int(v) for v in outcomes[j]],
                   "narrow_passed": bool(nar["passed"][j]), "narrow_held": bool(nar["held"][j]),
                   "narrow_breached": bool(nar["breached"][j]), "touch_time": round(nar["touch_time"][j], 3),
                   "min_clearance": round(min(nar["min_clearance"][j], 9.0), 4), "blocked": bool(blocked[j]),
                   "slack": round(slack[j], 4), "margin": round(margin[j], 4), "width": round(width[j], 4)}
            row.update({k: (round(v[j], 4) if isinstance(v[j], float) else v[j]) for k, v in extra.items()})
            row.update({f"sum_{k}": round(v[j], 4) for k, v in sums.items()})
            rows[env_id] = row
        recorded[ids] = True
    return _reset_idx(env_ids)


raw._reset_idx = types.MethodType(record_and_reset, raw)

obs, _ = venv.reset()
# One unrecorded step at the new tiles, then reset again: the first reset after moving the origins can carry the
# previous pose's contact history (base_contact on step 1, ~5 % of episodes in the first E2 run).
with torch.inference_mode():  # the step leaves inference tensors in the env buffers, so reset inside it too
    venv.step(policy(obs))
    obs, _ = venv.reset()
armed["on"] = True
dt = raw.step_dt
max_steps = int(SECONDS / dt) + 10
robot = raw.scene["robot"]
t0 = time.time()
step = 0
for step in range(max_steps):
    with torch.inference_mode():
        obs, _, _, _ = venv.step(policy(obs))
    live = ~recorded
    rel = robot.data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2]
    r = torch.maximum(rel[:, 0].abs(), rel[:, 1].abs())
    t_ep = raw.episode_length_buf.float() * dt
    fresh = t_ep <= dt * 1.5  # a reset this step: start the trackers over
    radius_max = torch.where(fresh, r, torch.maximum(radius_max, r))
    radius_min = torch.where(fresh, r, torch.minimum(radius_min, r))
    reach = torch.where(spec.outward, r >= STAIR_OUT_R, r <= STAIR_IN_R) & (spec.mode == 2)
    new = live & reach & ~passed_r
    pass_t = torch.where(new, t_ep, pass_t)
    passed_r |= new
    c = cmd.vel_command_b
    v = robot.data.root_lin_vel_b
    w = robot.data.root_ang_vel_b[:, 2]
    moving = live & (c[:, :2].norm(dim=-1) > 0.1) & (spec.mode > 0) & (spec.mode != MODE_TURN) & ~passed_r
    if V24:  # turn cells: tracking error of the turn command until the target angle is reached
        turn = mdp_turn.turn_episode(raw)
        moving |= live & (spec.mode == MODE_TURN) & (c[:, 2].abs() > 0.1) & ~turn.reached
        reach_t = torch.where(live & turn.reached & (reach_t < 0), t_ep, reach_t)
    err_v += moving.float() * (v[:, :2] - c[:, :2]).norm(dim=-1) * dt
    err_w += moving.float() * (w - c[:, 2]).abs() * dt
    moving_t += moving.float() * dt
    g = robot.data.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(g[:, 0].abs(), -g[:, 2]))
    pitch_max = torch.where(live, torch.maximum(pitch_max, pitch), pitch_max)
    nose = torch.rad2deg(torch.asin((-g[:, 0]).clamp(-1.0, 1.0)))
    nose_up_max = torch.where(live, torch.maximum(nose_up_max, nose), nose_up_max)
    nose_t10 += (live & (nose > 10.0)).float() * dt
    nose_t15 += (live & (nose > 15.0)).float() * dt
    # Decided before the time-out: a gap env leaving its tile (later slots / falls would be on a neighbour tile),
    # a stair/slope env at the far end of its flight. Record now and reset (the reset episode is not used).
    decided = ~recorded & (((spec.mode == 1) & (r >= GAP_LEAVE_R)) | ((spec.mode == 2) & passed_r))
    if V24:  # a turn env is decided by the pass (clean turn) or breach event; a hold runs to the time-out
        mon = term("narrow_monitor")
        decided |= ~recorded & (spec.mode == MODE_TURN) & (mon.passed | mon.breached)
    if bool(decided.any()):
        with torch.inference_mode():  # env buffers are inference tensors (written inside venv.step)
            record_and_reset(raw, decided.nonzero().flatten())
    if bool(recorded.all()):
        break
missing = (~recorded).nonzero().flatten().tolist()
if missing:  # still running at the end (should not happen: episodes time out at SECONDS)
    with torch.inference_mode():
        raw.termination_manager.time_outs[missing] = True
        record_and_reset(raw, torch.tensor(missing, device=device))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
result = {"schema": "e2_trial_v1", "task": TASK_KIND, "condition": cond, "seed": SEED, "seconds": SECONDS, "num_envs": n,
          "model": str(args.model), "steps": step + 1, "wall_s": round(time.time() - t0, 1), "censored": len(missing),
          "cells": cell_info, "rows": rows}
(out / "result.json").write_text(json.dumps(result))
print(f"[E2] wrote {out/'result.json'} envs={n} steps={step + 1} wall={time.time() - t0:.0f}s censored={len(missing)}",
      flush=True)
env.close()
app.close()
