"""Train an RSL-RL agent on a gd_lab vision-RL task (Stage 1 teacher: the blind
DreamWaQ policy plus the privileged terrain encoder).

Regenerated from scripts/train.py rather than kept as an old fork, so the
deployment-contract capture and the TRAIN_ARM experiment overrides stay in step
with the blind trainer. Only the default task and the camera flag differ."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

from gd_lab.core.experiments import training_arm_overrides

parser = argparse.ArgumentParser(description="Train an RSL-RL agent on a gd_lab task.")
parser.add_argument("--task", type=str, default="Gd-Vrl-Rbq10-Dreamwaq-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU / multi-node training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The actor terrain target is masked by rendered camera visibility.
args_cli.enable_cameras = True

# Explicit CLI overrides take precedence over the selected arm defaults.
train_arm = os.environ.get("TRAIN_ARM")
try:
    arm_overrides = training_arm_overrides(train_arm)
    arm_overrides = [item.replace("agent.experiment_name=blind_", "agent.experiment_name=vision_") for item in arm_overrides]
except ValueError as exc:
    parser.error(str(exc))
sys.argv = [sys.argv[0]] + arm_overrides + hydra_args
if train_arm is not None:
    print(f"[INFO] Training arm: {train_arm} (configs/experiment/arm_{train_arm}.yaml)")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
from datetime import datetime

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import gd_lab  # noqa: F401  (registers the tasks)
from gd_lab.core.paths import LOG_ROOT
from gd_lab.deploy.metadata import capture_context
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _resolve(path: str) -> type[OnPolicyRunner]:
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.distributed:
        if args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training requires a GPU device.")
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging run in: {log_dir}")
    env_cfg.log_dir = log_dir

    configure_vrl_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ensure_prev_prev_action_tracking(env.unwrapped)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class = _resolve(agent_cfg.class_name)
    runner: OnPolicyRunner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # A capture failure must not cost a training run; export refuses later
    # rather than guessing.
    try:
        runner.deploy_context = capture_context(env, runner.alg.policy)
    except (ValueError, AttributeError, KeyError, TypeError) as exc:
        print(f"[WARN] Deployment metadata unavailable; checkpoints will not carry it: {exc}")
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
