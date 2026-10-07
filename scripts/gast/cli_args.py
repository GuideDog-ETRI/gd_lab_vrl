"""Shared CLI arguments for the RSL-RL entry scripts."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for the RSL-RL agent.")
    arg_group.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name for logs.")
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix for the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module."
    )
    arg_group.add_argument("--log_project_name", type=str, default=None, help="wandb/neptune project name.")


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    if getattr(args_cli, "seed", None) is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    # A store_true flag can only turn resume ON; a config-level resume=True must
    # not be silently reset by the flag's False default.
    if args_cli.resume:
        agent_cfg.resume = True
    if getattr(args_cli, "device", None) is not None:
        agent_cfg.device = args_cli.device
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg
