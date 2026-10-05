"""Single-GPU BAVRL trainer. Explicit invocation only; never starts on import."""

import argparse
import os
import sys
import signal
from pathlib import Path

from isaaclab.app import AppLauncher

from gd_lab.core.experiments import training_arm_overrides
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Gd-Vrl-Rbq10-Dreamwaq-Vision-v0")
parser.add_argument("--teacher", required=True)
parser.add_argument("--teacher-agent", required=True)
parser.add_argument("--output", required=True, help="New, non-existing run directory")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--horizon", type=int, default=16)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None)
parser.add_argument("--delay-ms", type=float, default=100.)
parser.add_argument("--drop-prob", type=float, default=.05)
parser.add_argument("--top5-criteria", default=os.environ.get("GD_LAB_TOP5_CRITERIA_FILE") or
                    str(Path(__file__).resolve().parents[1] / "configs/online_top5.json"))
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if (min(args.num_envs, args.iterations, args.horizon, args.epochs, args.batch_size) < 1
        or not 0 <= args.drop_prob <= 1 or not 0 <= args.delay_ms <= 200):
    parser.error("Invalid count, dropout probability or camera delay (0..200 ms)")
args.enable_cameras = True
sys.argv = [sys.argv[0]] + training_arm_overrides(os.environ.get("TRAIN_ARM", "4")) + hydra_args
runtime = prepare_vrl_runtime(args)
app = AppLauncher(args).app
verify_vrl_runtime(runtime)

import json
import random

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from torch.utils.tensorboard import SummaryWriter

import gd_lab  # noqa: F401
from gd_lab.deploy.metadata import capture_context
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.residuals.bavrl import BAVRL, ResidualConfig, load_blind_teacher
from gd_lab.residuals.bavrl.ppo import ResidualPPO, gae
from gd_lab.residuals.bavrl.online_quality import OnlineQuality
from gd_lab.rl.online_rollout import install_episode_collector
from gd_lab.mdp.terrain_families import family_column_masks
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras


def check_contract(saved, current):
    """Fail closed on actuator/observation mismatches, not just tensor dimensions."""
    for key in ("policy_dt", "joint_names", "default_joint_pos", "terms", "actor_history_steps", "history_initialization"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Blind teacher deployment contract mismatch: {key}")
    for key in ("type", "scale", "offset", "clip", "previous_action", "gains", "soft_margin_deg"):
        if saved.get("action", {}).get(key) != current.get("action", {}).get(key):
            raise ValueError(f"Blind teacher action contract mismatch: {key}")


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    from gd_lab.core.camera_contract import require_training_camera_profile
    require_training_camera_profile(env_cfg.camera_profile)  # legacy calibration only with explicit opt-in
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.device:
        env_cfg.sim.device = args.device
    configure_vrl_cameras(env_cfg)
    env_cfg.sim.render_interval = env_cfg.decimation
    env = gym.make(args.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    writer = SummaryWriter(str(output))
    try:
        obs = env.get_observations()
        teacher, teacher_hash = load_blind_teacher(args.teacher, args.teacher_agent, obs)
        context = capture_context(env, teacher)
        source = torch.load(args.teacher, map_location="cpu", weights_only=True)
        check_contract(source["infos"]["gd_lab"]["deploy_context"], context)
        model = BAVRL(teacher, ResidualConfig(camera_profile=env_cfg.camera_profile)).to(env.device)
        learner = ResidualPPO(model)
        start = 0
        if args.resume:
            state = torch.load(args.resume, map_location=env.device, weights_only=True)
            model.restore(state, teacher_hash, learner.optimizer)
            start = state["iteration"]
        columns = {name: ids.cpu().tolist() for name, ids in family_column_masks(env.unwrapped).items()}
        quality_records = install_episode_collector(env, columns)
        quality = OnlineQuality(output, args.top5_criteria)
        stopping = False

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
            print('[INFO] Stop requested; saving after the current complete iteration.', flush=True)

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        def save_checkpoint(path):
            state = model.checkpoint(teacher_hash, learner.optimizer)
            state.update(iteration=iteration+1, deploy_context=context)
            target = Path(path)
            temporary = target.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(target)
        dump_yaml(str(output / "env.yaml"), env_cfg)
        (output / "run.json").write_text(json.dumps({"args": vars(args), "teacher_sha256": teacher_hash,
                                                     "deploy_context": context}, indent=2))
        n, device, dt = env.num_envs, env.device, env.unwrapped.step_dt
        frames = torch.zeros(n, 4, 2, 45, 80, device=device)
        terrain = torch.zeros(n, 374, device=device)
        hidden = torch.zeros(n, 64, device=device)
        memory = hidden.clone()
        vision_age = torch.zeros(n, device=device)
        previous = torch.zeros(n, 12, device=device)
        delivered = torch.full((n,), -100000, device=device, dtype=torch.long)
        captured = delivered.clone()
        episode = torch.zeros(n, device=device, dtype=torch.long)
        available = torch.zeros(n, device=device, dtype=torch.bool)
        pending = []
        for iteration in range(start, start + args.iterations):
            steps = []
            for _ in range(args.horizon):
                tick = env.unwrapped.common_step_counter
                snapshot = getattr(env.unwrapped, "_vrl_camera_snapshot", None)
                snapshot_steps = getattr(env.unwrapped, "_vrl_camera_snapshot_steps", None)
                if snapshot is None or snapshot_steps is None:
                    available[:] = False
                    pending.clear()
                stamps = captured.clone() if snapshot_steps is None else snapshot_steps.clone()
                fresh = (stamps > captured) if snapshot is not None else torch.zeros_like(available)
                expired = ~available | ((tick-delivered)*dt > model.config.stale_seconds)
                hidden[expired] = memory[expired] = 0
                if fresh.any():
                    captured = stamps.clone()
                    # Capture all targets before transport; never pair delayed image with current terrain.
                    if rng.random() >= args.drop_prob:
                        pending.append((tick + rng.randint(0, round(args.delay_ms/1000/dt)),
                                        snapshot[0].clone(), obs["terrain"].clone(),
                                        stamps, episode.clone(), fresh))
                remaining = []
                for packet in pending:
                    due, images, target, stamp, ep, mask = packet
                    if due > tick:
                        remaining.append(packet)
                        continue
                    rows = mask & (ep == episode) & (stamp > delivered)
                    frames[rows], terrain[rows] = images[rows], target[rows]
                    hidden[rows] = memory[rows]  # Input memory stays fixed while a frame is held.
                    vision_age[rows] = (tick-stamp[rows]).float()*dt
                    delivered[rows], available[rows] = stamp[rows], True
                pending = remaining
                age = (tick-delivered).float()*dt
                record = {"obs": obs.clone(), "frames": frames.clone(), "hidden": hidden.clone(),
                          "previous": previous.clone(), "age": age.clone(), "available": available.clone(),
                          "terrain": terrain.clone(), "supervised": available.clone(), "vision_age": vision_age.clone()}
                with torch.no_grad():
                    base, dist, value, next_memory, _, good = model(obs, frames, hidden, previous, age, available, vision_age)
                    raw = dist.sample()
                    action, delta = model.compose(base, raw, previous, good)
                    memory = next_memory.detach()
                    record.update(raw=raw, log_prob=dist.log_prob(raw).sum(-1), value=value)
                obs, reward, done, extras = env.step(action)
                done = done.bool()
                previous = delta.detach()
                with torch.no_grad():
                    critic_obs = teacher.critic_obs_normalizer(teacher.get_critic_obs(obs))
                    next_value = model.critic(critic_obs).squeeze(-1)
                    # Isaac auto-reset does not expose final observations: match RSL's
                    # pre-step V timeout approximation, never bootstrap from reset obs.
                    timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                    next_value = torch.where(done, torch.where(timeout, value, 0.), next_value)
                record.update(reward=reward, done=done, next_value=next_value)
                steps.append(record)
                episode[done] += 1
                for tensor in (hidden, memory, previous, frames, terrain):
                    tensor[done] = 0
                available[done] = False
                delivered[done] = captured[done] = -100000
            advantage, returns = gae(torch.stack([s["reward"] for s in steps]),
                                     torch.stack([s["value"] for s in steps]),
                                     torch.stack([s["next_value"] for s in steps]),
                                     torch.stack([s["done"] for s in steps]))
            for index, record in enumerate(steps):
                record["advantage"], record["return"] = advantage[index], returns[index]
            batch = {key: torch.cat([s[key] for s in steps], dim=0) for key in steps[0]}
            metrics = []
            for _ in range(args.epochs):
                for ids in torch.randperm(n*args.horizon, device=device).split(args.batch_size):
                    metrics.append(learner.update({key: value[ids] for key, value in batch.items()}))
            for key in metrics[0]:
                writer.add_scalar("BAVRL/"+key, sum(m[key] for m in metrics)/len(metrics), iteration+1)
            writer.add_scalar("BAVRL/reward", batch["reward"].mean().item(), iteration+1)
            print(f"BAVRL {iteration+1}: reward={batch['reward'].mean().item():.4f}", flush=True)
            completed = list(quality_records)
            quality_records.clear()
            report = quality.update(iteration+1, completed, save_checkpoint)
            candidate = report['candidate']
            score = None if candidate is None else candidate['score']
            writer.add_scalar('Quality/completed_episodes', len(completed), iteration+1)
            writer.add_scalar('Quality/top5_saved', int(report['saved']), iteration+1)
            if score is not None:
                writer.add_scalar('Quality/S_online', score, iteration+1)
            print(f"Quality {iteration+1}: episodes={len(completed)} S_online={score} "
                  f"saved={report['saved']} gates={report['failed_gates']}", flush=True)
            if (iteration+1) % 50 == 0 or iteration == start+args.iterations-1 or stopping:
                save_checkpoint(output / f"bavrl_{iteration+1}.pt")
            if stopping:
                print(f'[INFO] Stopped with checkpoint at {iteration+1}.', flush=True)
                break
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
