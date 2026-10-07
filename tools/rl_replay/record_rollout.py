"""Record a policy rollout for offline reward analysis (tools/rl_replay/viewer.html plays it back).

Per control step (100 Hz) and env: base pose and velocity, every body position (for drawing the robot),
joint positions/velocities, the command, the policy's mean action and std, the critic value V(s), every
reward term (weighted, per step, as the RewardManager adds it), foot contacts, the 11x17 height-scan points
and the terrain latent. After the rollout it adds, per env, the discounted return-to-go G_t, its split by
reward term (G_t^k = sum_i gamma^i r^k_{t+i}), the TD error delta_t and the GAE advantage A_t -- the numbers
that say why PPO pushes the policy toward the action it took.

  PYTHONPATH=src python tools/rl_replay/record_rollout.py --headless \
      --task Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0 --checkpoint <model.pt> --num_envs 8 --seconds 4 \
      --out logs/rl_replay/gast2000.json [--vx 0.8] [--stochastic] [--student_view] [--live <file.ndjson>]

--student_view renders the four belly cameras and stores what they see as world points (the student's view),
next to the ground-truth scan and the policy's own terrain input. --live also streams every step as one JSON
line while the simulator runs, so the viewer can follow it live (G_t and A_t there are provisional).

The JSON is self-contained (meta + per-env arrays, rounded to 4 decimals).
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="the task the policy was trained on")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seconds", type=float, default=4.0, help="recorded window (3-4 s shows a gap or a stair run)")
parser.add_argument("--warmup_seconds", type=float, default=1.0, help="walk this long before recording")
parser.add_argument("--vx", type=float, default=None, help="fix the forward command (m/s); default: the task's commands")
parser.add_argument("--stochastic", action="store_true", help="sample actions (training behaviour) instead of the mean")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", required=True)
parser.add_argument("--student_view", action="store_true", help="render the belly cameras and record their depth as points")
parser.add_argument("--cloud_envs", type=int, default=2, help="envs that keep camera points (they are large)")
parser.add_argument("--cloud_stride", type=int, default=4, help="keep every n-th depth pixel per axis")
parser.add_argument("--live", default=None, help="also stream each step as an NDJSON line to this file")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.student_view or "Raycast" in args_cli.task or "Vision" in args_cli.task:
    args_cli.enable_cameras = True  # BIVT-Ray's camera-visible terrain needs the cameras too
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import hashlib
import importlib
import json
import os
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import gd_lab  # noqa: F401  (registers the tasks)
if args_cli.task.startswith("Gd-Gast"):
    import gd_lab.gast.tasks  # noqa: F401
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


def action_std(policy, mean):
    """The Gaussian policy's std without sampling (rsl_rl keeps either ``std`` or ``log_std``)."""
    if getattr(policy, "std", None) is not None:
        return policy.std.detach().expand_as(mean)
    if getattr(policy, "log_std", None) is not None:
        return policy.log_std.detach().exp().expand_as(mean)
    return torch.zeros_like(mean)


def _round(t, digits=4):
    return torch.round(t.float() * 10**digits).div(10**digits).cpu().tolist()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    cameras = args_cli.enable_cameras
    if cameras:
        from gd_lab.core.camera_contract import DEFAULT_CAMERA_PROFILE
        from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
        if not hasattr(env_cfg, "camera_profile"):
            env_cfg.camera_profile = DEFAULT_CAMERA_PROFILE
        configure_vrl_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    runner = _resolve(agent_cfg.class_name)(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.alg.policy
    policy.eval()

    robot = base.scene["robot"]
    rewards = base.reward_manager
    term_names = list(rewards.active_terms)
    scanner = base.scene.sensors.get("height_scanner") if hasattr(base.scene, "sensors") else None
    contacts = base.scene.sensors.get("contact_forces") if hasattr(base.scene, "sensors") else None
    foot_ids = [robot.body_names.index(n) for n in robot.body_names if n.endswith("_foot")]
    contact_foot_ids = ([contacts.body_names.index(robot.body_names[i]) for i in foot_ids]
                        if contacts is not None else [])
    dt = base.step_dt
    gamma = float(agent_cfg.algorithm.gamma)
    lam = float(agent_cfg.algorithm.lam)
    student_view = None
    if cameras and args_cli.student_view:
        from gd_lab.core.camera_contract import load_camera_contract
        from gd_lab.core.camera_geometry import depth_pixels_world
        from gd_lab.mdp.camera_observations import canonical_camera_snapshot
        contract = load_camera_contract(env_cfg.camera_profile)
        k = min(args_cli.cloud_envs, base.num_envs)

        def student_view():
            """World points of the four depth images (what the student sees), one flat [x,y,z,...] list per env."""
            _, depths, positions, rotations, intrinsics = canonical_camera_snapshot(base.scene, contract)
            pts = depth_pixels_world(depths[:k], positions[:k], rotations[:k], intrinsics[:k])
            s = args_cli.cloud_stride
            pts, dep = pts[:, :, ::s, ::s].reshape(k, -1, 3), depths[:k, :, ::s, ::s].reshape(k, -1)
            lo, hi = contract.depth_clip
            ok = torch.isfinite(dep) & (dep > lo) & (dep < hi * 0.999) & torch.isfinite(pts).all(-1)
            return [torch.round(pts[e][ok[e]] * 1000).div(1000).flatten().cpu().tolist() for e in range(k)]

    def set_command():
        if args_cli.vx is None:
            return
        cmd = base.command_manager.get_term("base_velocity")
        cmd.vel_command_b[:, 0] = args_cli.vx
        cmd.vel_command_b[:, 1:] = 0.0

    def env_step_payload(e, i):
        """One env's data at step i, for the live stream (same keys as the final file)."""
        row = {key: _round(rec[key][i][e]) for key in rec if key not in ("done", "foot_contact")}
        row["done"] = int(rec["done"][i][e])
        row["foot_contact"] = rec["foot_contact"][i][e].int().tolist()
        if clouds:
            row["cloud"] = clouds[i][e] if e < len(clouds[i]) else []
        return row

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(int(round(args_cli.warmup_seconds / dt))):
            set_command()
            obs, _, _, _ = env.step(policy.act_inference(obs))

    steps = int(round(args_cli.seconds / dt))
    n = base.num_envs
    rec = {k: [] for k in ("base_pos", "base_quat", "base_lin_vel_b", "command", "bodies", "joint_pos", "joint_vel",
                           "action_mean", "action_std", "action", "value", "terms", "reward", "done", "foot_contact",
                           "scan", "scan_z", "latent", "terrain_obs")}
    clouds = []
    live = None
    if args_cli.live:
        Path(args_cli.live).parent.mkdir(parents=True, exist_ok=True)
        live = open(args_cli.live, "w")
        live.write(json.dumps({"meta": {"task": args_cli.task, "checkpoint": os.path.abspath(args_cli.checkpoint),
            "dt": dt, "gamma": gamma, "lam": lam, "steps_planned": steps, "num_envs": n, "stochastic": args_cli.stochastic,
            "vx": args_cli.vx, "term_names": term_names, "body_names": list(robot.body_names),
            "joint_names": list(robot.joint_names), "foot_names": [robot.body_names[i] for i in foot_ids], "live": True}},
            separators=(",", ":")) + "\n")
        live.flush()
    with torch.inference_mode():
        for _ in range(steps):
            set_command()
            mean = policy.act_inference(obs)
            action = policy.act(obs) if args_cli.stochastic else mean
            std = action_std(policy, mean)
            value = policy.evaluate(obs).squeeze(-1)
            latent = policy.terrain_latent(obs) if hasattr(policy, "terrain_latent") else torch.zeros(n, 0, device=mean.device)
            cmd = base.command_manager.get_command("base_velocity")
            rec["base_pos"].append(robot.data.root_pos_w.clone())
            rec["base_quat"].append(robot.data.root_quat_w.clone())
            rec["base_lin_vel_b"].append(robot.data.root_lin_vel_b.clone())
            rec["command"].append(cmd.clone())
            rec["bodies"].append(robot.data.body_pos_w.clone())
            rec["joint_pos"].append(robot.data.joint_pos.clone())
            rec["joint_vel"].append(robot.data.joint_vel.clone())
            rec["action_mean"].append(mean.clone())
            rec["action_std"].append(std.expand_as(mean).clone())
            rec["action"].append(action.clone())
            rec["value"].append(value.clone())
            rec["latent"].append(latent.clone())
            # What the policy itself was given: its 'terrain' group (11x17 height *5 + validity), e.g. the
            # camera-visible cells for BIVT-Ray or the noisy/blacked-out full grid for GAST.
            if "terrain" in obs.keys():
                terrain_in = obs["terrain"]
            elif "gast_history" in obs.keys():  # GAST teacher: the newest of its 8 terrain frames (same encoding)
                terrain_in = obs["gast_history"].reshape(n, 8, -1)[:, -1, :374]
            else:
                terrain_in = torch.zeros(n, 0, device=mean.device)
            rec["terrain_obs"].append(terrain_in.clone())
            rec["scan"].append(scanner.data.ray_hits_w.clone() if scanner is not None else torch.zeros(n, 0, 3, device=mean.device))
            rec["scan_z"].append(scanner.data.pos_w[:, 2].clone() if scanner is not None else torch.zeros(n, device=mean.device))
            if student_view is not None:
                clouds.append(student_view())
            obs, reward, done, _ = env.step(action)
            # RewardManager keeps each weighted term as a rate; the per-step contribution is rate * dt.
            rec["terms"].append(rewards._step_reward.clone() * dt)
            rec["reward"].append(reward.clone())
            rec["done"].append(done.clone())
            if contacts is not None:
                forces = contacts.data.net_forces_w[:, contact_foot_ids].norm(dim=-1)
                rec["foot_contact"].append(forces > 1.0)
            else:
                rec["foot_contact"].append(torch.zeros(n, len(foot_ids), dtype=torch.bool, device=mean.device))
            if live is not None:
                i = len(rec["reward"]) - 1
                live.write(json.dumps({"t": i, "envs": [env_step_payload(e, i) for e in range(n)]}, separators=(",", ":")) + "\n")
                live.flush()
        bootstrap = policy.evaluate(obs).squeeze(-1)

    stacked = {k: torch.stack(v, 1) for k, v in rec.items()}  # [n, T, ...]
    r, v, d = stacked["reward"], stacked["value"], stacked["done"].float()
    terms = stacked["terms"]
    returns = torch.zeros_like(r)
    term_returns = torch.zeros_like(terms)
    adv = torch.zeros_like(r)
    delta = torch.zeros_like(r)
    next_value, next_return, next_term_return, next_adv = bootstrap, bootstrap, torch.zeros_like(terms[:, 0]), torch.zeros_like(bootstrap)
    for t in reversed(range(steps)):
        alive = 1.0 - d[:, t]
        delta[:, t] = r[:, t] + gamma * next_value * alive - v[:, t]
        adv[:, t] = delta[:, t] + gamma * lam * alive * next_adv
        returns[:, t] = r[:, t] + gamma * alive * next_return
        term_returns[:, t] = terms[:, t] + gamma * alive[:, None] * next_term_return
        next_value, next_return, next_term_return, next_adv = v[:, t], returns[:, t], term_returns[:, t], adv[:, t]

    sha = hashlib.sha256(Path(args_cli.checkpoint).read_bytes()).hexdigest()
    out = {
        "meta": {"task": args_cli.task, "checkpoint": os.path.abspath(args_cli.checkpoint), "checkpoint_sha256": sha,
                 "dt": dt, "gamma": gamma, "lam": lam, "steps": steps, "num_envs": n,
                 "stochastic": args_cli.stochastic, "vx": args_cli.vx, "term_names": term_names,
                 "body_names": list(robot.body_names), "joint_names": list(robot.joint_names),
                 "foot_names": [robot.body_names[i] for i in foot_ids],
                 "notes": {"terms": "weighted reward per step (rate*dt), summing to 'reward'",
                           "scan": "height-scanner hits (privileged ground truth, 11x17)",
                           "terrain_obs": "the policy's terrain input: 187 heights (scan_z - z - 0.5, clipped +-1, x5) + 187 validity",
                           "cloud": "student view: belly-camera depth pixels as world points, flat [x,y,z,...]",
                           "term_returns": "discounted return-to-go of each term; their sum is 'return' "
                                           "(the bootstrap value V(s_T) is in 'return' only)",
                           "advantage": "GAE(gamma, lam) with the critic value", "delta": "TD error"}},
        "envs": [],
    }
    for e in range(n):
        env_out = {k: _round(stacked[k][e]) for k in stacked if k not in ("done", "foot_contact")}
        env_out["done"] = stacked["done"][e].cpu().int().tolist()
        env_out["foot_contact"] = stacked["foot_contact"][e].cpu().int().tolist()
        env_out["return"] = _round(returns[e])
        env_out["term_returns"] = _round(term_returns[e])
        env_out["advantage"] = _round(adv[e])
        env_out["delta"] = _round(delta[e])
        if clouds and e < len(clouds[0]):
            env_out["cloud"] = [c[e] for c in clouds]
        out["envs"].append(env_out)
    Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out).write_text(json.dumps(out, separators=(",", ":")))
    if live is not None:
        live.write(json.dumps({"end": True, "out": args_cli.out}) + "\n")
        live.close()
    print(f"[INFO] wrote {args_cli.out}: {n} envs x {steps} steps, {len(term_names)} reward terms", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
