"""Run a trained vision-RL gd_lab policy in a play environment and export it for deploy.

Separate from ``scripts/play.py`` (the blind, 2-input export contract) by
design -- see ``gd_lab.deploy.export_vrl``'s module docstring for why the two
must not be mixed. Everything else about this script is identical to
``play.py``; only the export call differs (``export_policy_vrl``, which also
needs ``terrain_latent_dim``).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Play a trained gd_lab policy.")
parser.add_argument("--task", type=str, default="Gd-Vrl-Rbq10-Dreamwaq-Play-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to simulate.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--no-export", action="store_true", default=False, help="Skip the JIT/ONNX export.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

runtime_root = prepare_vrl_runtime(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
verify_vrl_runtime(runtime_root)

import importlib
import os
import time

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import gd_lab  # noqa: F401  (registers the tasks)
from gd_lab.core.paths import LOG_ROOT
from gd_lab.deploy.export_vrl import export_policy_vrl
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(checkpoint_path)

    configure_vrl_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class = _resolve(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if not args_cli.no_export and isinstance(runner.alg.policy, DreamwaqVrlActorCritic):
        export_dir = os.path.join(log_dir, "exported")
        policy_term_dims = [t.dim for t in DREAMWAQ_SPEC.policy.terms]
        jit_path, onnx_path = export_policy_vrl(
            runner.alg.policy, export_dir, policy_term_dims, terrain_latent_dim=agent_cfg.policy.terrain_latent_dim
        )
        print(f"[INFO] Exported policy: {jit_path}, {onnx_path}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
