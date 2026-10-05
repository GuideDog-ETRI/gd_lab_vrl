"""Gap evaluation grid for a BIVT-Ray teacher (spec gate C, teacher side).

!! INCOMPLETE and UNVERIFIED: teacher only, written without Isaac on this PC, never executed. !!
!! It has no paired GAST student path and no flat/stairs regression set, so it can NOT produce !!
!! the gate-C verdict. It writes measurements only and never creates or satisfies the training !!
!! gate (--baseline_gate, schema gap_baseline_gate_v1); that record needs the missing parts. !!
!! Codex must run the smoke below on the server and fix API drift before trusting any number. !!

Every cell fixes gap width and height offset (one env is built per (width, height) pair), pins the
forward command to one speed, puts every env on a platform_gap column, and runs one episode per
env while the GapMonitor tracker (the same code the training arms use) records attempts,
intrusion depth and slot dwell/contact. No PPO update, no optimizer step.

Not covered: the paired GAST student (needs the student policy loader) and the "no-gap" flat/stairs
regression set. Both stay on the gate-C checklist.

Smoke:  python scripts/eval_gap_cases.py --headless --checkpoint <17206_top1.pt> --smoke
Full:   python scripts/eval_gap_cases.py --headless --checkpoint <ckpt> --out logs/gap_eval/teacher_17206.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Gap evaluation grid (teacher).")
parser.add_argument("--task", default="Gd-VrlGapFinetuneBaselineRaycast-Rbq10-Dreamwaq-v0",
                    help="Baseline arm: adds the monitor but no reward change.")
parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint (17206_top1.pt or a fine-tuned one).")
parser.add_argument("--widths", default="0.05,0.10,0.15,0.20,0.24,0.30")
parser.add_argument("--heights", default="0.0,0.08,0.16", help="Deck height offset magnitude; the sign is random per tile.")
parser.add_argument("--speeds", default="0.3,0.6,0.8,1.0")
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episode_steps", type=int, default=2000, help="20 s at 100 Hz.")
parser.add_argument("--out", default="logs/gap_eval/gap_eval.jsonl")
parser.add_argument("--smoke", action="store_true", help="One cell, 200 steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False  # raycast teacher, camera-free
sys.argv = [sys.argv[0]] + hydra_args

runtime_root = prepare_vrl_runtime(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
verify_vrl_runtime(runtime_root)

import importlib

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import gd_lab  # noqa: F401  (registers the tasks)
import gd_lab.teachers.bivt  # noqa: F401  (registers the gap tasks)
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.mdp.platform_gap_attempts import EpisodeArchive
from gd_lab.mdp.terrain_families import family_column_masks

MONITOR = "platform_gap_monitor"


def _floats(text):
    return [float(item) for item in text.split(",") if item]


def build_env(width, height):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    generator = env_cfg.scene.terrain.terrain_generator
    sub = generator.sub_terrains["platform_gap"]
    sub.gap_width_range, sub.height_offset_range = (width, width), (height, height)
    generator.difficulty_range = (0.0, 0.0)  # every row has exactly this width/height
    for name in list(vars(env_cfg.curriculum)):  # no level or command-range changes during evaluation
        setattr(env_cfg.curriculum, name, None)
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (1.0e6, 1.0e6)
    command.rel_standing_envs = 0.0
    if hasattr(command, "pulse_prob"):
        command.pulse_prob = 0.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    module_name, attr = agent_cfg.class_name.split(":")
    runner = getattr(importlib.import_module(module_name), attr)(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    return env, runner


def place_on_gap_tiles(env):
    unwrapped = env.unwrapped
    terrain = unwrapped.scene.terrain
    columns = family_column_masks(unwrapped)["platform_gap"]
    terrain.terrain_types[:] = columns[torch.arange(unwrapped.num_envs, device=unwrapped.device) % len(columns)]
    terrain.terrain_levels[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def pin_speed(env, speed):
    command = env.unwrapped.command_manager.get_term("base_velocity")
    original = getattr(command, "_unpinned_resample", None) or command._resample_command
    command._unpinned_resample = original  # never stack wrappers when the speed changes

    def resample(env_ids):
        original(env_ids)
        command.vel_command_b[env_ids, 0] = speed
        command.vel_command_b[env_ids, 1] = 0.0
        command.is_standing_env[env_ids] = False

    command._resample_command = resample
    resample(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device))


def run_cell(env, runner, speed, seed, steps):
    """One episode per env: the FIRST episode of every env is archived right before its reset."""
    unwrapped = env.unwrapped
    monitor = unwrapped.reward_manager.get_term_cfg(MONITOR).func
    torch.manual_seed(seed)
    env.reset()
    pin_speed(env, speed)
    monitor.tracker.reset(None)
    archive = EpisodeArchive(monitor.tracker)
    archive.install(monitor)  # reset wipes the tracker; the archive snapshots first
    policy = runner.get_inference_policy(device=unwrapped.device)
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    seen = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
    terminated = timed_out = 0
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, dones, extras = env.step(policy(obs))
            done = dones.bool()
            time_outs = extras.get("time_outs", torch.zeros_like(done)).bool()
            fresh = done & ~seen  # later episodes of an env are ignored, like in the archive
            terminated += int((fresh & ~time_outs).sum())
            timed_out += int((fresh & time_outs).sum())
            seen |= done
    stats, records = archive.finish()
    return {"terminated": terminated, "timed_out": timed_out, "steps": steps,
            "truncated_envs": archive.truncated, **stats}, records


def main():
    widths, heights, speeds, seeds = (_floats(args_cli.widths), _floats(args_cli.heights), _floats(args_cli.speeds),
                                      [int(s) for s in _floats(args_cli.seeds)])
    steps = args_cli.episode_steps
    if args_cli.smoke:
        widths, heights, speeds, seeds, steps = widths[:1], heights[:1], speeds[:1], seeds[:1], 200
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", buffering=1) as handle:
        for width in widths:
            for height in heights:
                env, runner = build_env(width, height)
                place_on_gap_tiles(env)
                for speed in speeds:
                    for seed in seeds:
                        cell, records = run_cell(env, runner, speed, seed, steps)
                        row = {"checkpoint": args_cli.checkpoint, "width": width, "height": height, "speed": speed,
                               "seed": seed, "num_envs": args_cli.num_envs, **cell, "records": records}
                        handle.write(json.dumps(row) + "\n")
                        print(f"[EVAL] w={width} h={height} v={speed} s={seed}: " +
                              json.dumps({k: v for k, v in row.items() if k != "records"}), flush=True)
                env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
