"""Train the handstand skill S1 (docs/design/SKILL_HANDSTAND_WALK_DESIGN.md) from the blind DWB-38000.

Same flow as scripts/train.py (arm 4 overrides via TRAIN_ARM=4), plus:
  - registers Gd-HandstandS1-Rbq10-Dreamwaq-v0,
  - --warm_start <DWB-38000 .pt>: copies the blind weights with zero columns for the mode command,
  - HANDSTAND_MIN_STD (default 0.5) floors the start action std (DWB's is small; the skill needs exploration).
"""
import argparse
import faulthandler
import os
import signal
import sys
from pathlib import Path

faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
# training framework (gd_lab) = the gd_lab_vrl checkout; this skill package lives outside it (local only)
REPO = Path(os.environ.get("GD_LAB_VRL_ROOT", Path(__file__).resolve().parents[3])).resolve()
sys.path.insert(0, str(REPO / "scripts"))   # cli_args.py

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

from gd_lab.core.experiments import training_arm_overrides  # noqa: E402

parser = argparse.ArgumentParser(description="Train the handstand skill (S1).")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--warm_start", type=str, required=True, help="DWB-38000 checkpoint")
parser.add_argument("--init_from", type=str, default=None,
                    help="handstand checkpoint with the same layout (weights only, new optimizer); replaces the DWB widening")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
train_arm = os.environ.get("TRAIN_ARM", "4")
# arm 4 = 100 Hz policy + nominal gains; its velocity-command pulse keys do not exist in this task
_arm = [o for o in training_arm_overrides(train_arm) if "pulse_prob" not in o and "experiment_name" not in o]
sys.argv = [sys.argv[0]] + _arm + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import gd_lab  # noqa: E402,F401
from gd_lab.core import registry  # noqa: E402
from gd_lab.core.paths import LOG_ROOT  # noqa: E402
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking  # noqa: E402
from gd_lab.rl.runner import DreamwaqRunner  # noqa: E402
from skill_handstand.warm_start import warm_start_from_dwb, widen_from_handstand  # noqa: E402

# v1 = assist wrench curriculum, v2 = start-pose resets, v3 = v2 + lift-off poses + front-load reward
# s2 = stepping in place (walk flag appended; widened from an S1 checkpoint)
_VARIANT = os.environ.get("HANDSTAND_VARIANT", "s2")
_TASK, _ENV, _AGENT = {
    "v1": ("HandstandS1", "HandstandS1EnvCfg", "HandstandRunnerCfg"),
    "v2": ("HandstandS1V2", "HandstandS1V2EnvCfg", "HandstandRunnerCfg"),
    "v3": ("HandstandS1V3", "HandstandS1V3EnvCfg", "HandstandRunnerCfg"),
    "s2": ("HandstandS2", "HandstandS2EnvCfg", "HandstandS2RunnerCfg"),
    "s2b": ("HandstandS2B", "HandstandS2BEnvCfg", "HandstandS2RunnerCfg"),
    "s2c": ("HandstandS2C", "HandstandS2CEnvCfg", "HandstandS2RunnerCfg"),
}[_VARIANT]
TASK = registry.register_task(task=_TASK, robot="Rbq10", method="Dreamwaq", env_cfg="skill_handstand.tasks:" + _ENV,
                              agent_cfg="skill_handstand.agent:" + _AGENT)
CRITIC_EXTRA = 5 + (1 if _VARIANT.startswith("s2") else 0)   # skill_command 3 + skill_curriculum 2 (+ skill_walk)


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    log_dir = os.path.join(os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name)),
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""))
    print(f"[INFO] Logging run in: {log_dir}", flush=True)
    env_cfg.log_dir = log_dir
    env = gym.make(TASK, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = DreamwaqRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.init_from:
        ck = torch.load(args_cli.init_from, map_location=agent_cfg.device, weights_only=False)
        old_step = ck["model_state_dict"]["actor_obs_normalizer._mean"].shape[1] // runner.alg.policy.history_length
        if old_step == runner.alg.policy.one_step_obs_dim:
            runner.alg.policy.load_state_dict(ck["model_state_dict"], strict=True)
            report = dict(source=args_cli.init_from, source_iteration=int(ck.get("iter", -1)), kind="handstand_continuation")
        else:
            report = widen_from_handstand(runner.alg.policy, args_cli.init_from, old_step=old_step,
                                          critic_extra=runner.alg.policy.one_step_obs_dim - old_step)
    else:
        report = warm_start_from_dwb(runner.alg.policy, args_cli.warm_start,
                                     min_std=float(os.environ.get("HANDSTAND_MIN_STD", "0.5")), critic_extra=CRITIC_EXTRA)
    start_level = int(os.environ.get("HANDSTAND_START_LEVEL", "0"))
    if start_level:
        env.unwrapped.command_manager.get_term("skill_mode").level = start_level
        report["start_level"] = start_level
    runner.gast_warm_start = dict(report, task=TASK)   # stored in every checkpoint (infos.gd_lab.gast_warm_start)
    print(f"[HANDSTAND] warm start {report}", flush=True)
    runner.add_git_repo_to_log(__file__)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
