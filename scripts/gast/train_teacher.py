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
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Train an RSL-RL agent on a gd_lab task.")
parser.add_argument("--task", type=str, default="Gd-Gast-Rbq10-Dreamwaq-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
env_count_args = parser.add_mutually_exclusive_group()
env_count_args.add_argument("--num_envs", type=int, default=None, help="Environments per process.")
env_count_args.add_argument("--total_envs", type=int, default=None, help="Total environments across distributed processes.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU / multi-node training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
parser.add_argument("--warm_start_bivt", type=str, default=None,
                    help="BIVT-Ray teacher checkpoint: copy its DreamWaQ backbone (actor/critic/normalizers/CENet) into "
                         "a fresh GAST terrain encoder (gd_lab.gast.warm_start). Starts a new run at iteration 0.")
parser.add_argument("--warm_start_gast", type=str, default=None,
                    help="UPDATE134B: GAST v2.1 teacher checkpoint copied whole into the Ray+IR teacher (first Conv "
                         "widened with zero IR columns). Starts a new run at iteration 0.")
parser.add_argument("--warm_start_dwb", type=str, default=None,
                    help="UPDATE135: blind DreamWaQ checkpoint as the backbone of a fresh Ray+IR GAST teacher "
                         "(actor latent columns and latent head zero). Starts a new run at iteration 0.")
parser.add_argument("--resume_rayir149", type=str, default=None, help="Strict native UPDATE149B continuation")
parser.add_argument("--warm_start_rayir", type=str, default=None, help="UPDATE149 native Ray+IR weights; new PPO, restored terrain levels, iteration zero.")
parser.add_argument("--warm_start_sha256", type=str, default=None,
                    help="Expected sha256 of --warm_start_bivt; the 3-GPU launcher requires it for train runs.")
parser.add_argument("--policy_lr", type=float, default=None, help="Fixed PPO lr of the backbone group (with --terrain_lr).")
parser.add_argument("--terrain_lr", type=float, default=None, help="Fixed PPO lr of the GAST terrain encoder/decoder group.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if (args_cli.policy_lr is None) != (args_cli.terrain_lr is None):
    parser.error("--policy_lr and --terrain_lr go together")
if sum(x is not None for x in (args_cli.warm_start_bivt, args_cli.warm_start_gast, args_cli.warm_start_dwb, args_cli.warm_start_rayir, args_cli.resume_rayir149)) > 1:
    parser.error("choose one of --warm_start_bivt / --warm_start_gast / --warm_start_dwb / --warm_start_rayir")
if args_cli.warm_start_rayir is not None:
    if args_cli.policy_lr is None or args_cli.warm_start_sha256 is None or not os.path.isfile(args_cli.warm_start_rayir):
        parser.error("--warm_start_rayir needs an existing checkpoint, sha256 and both learning rates")
    if getattr(args_cli, "resume", False):
        parser.error("--warm_start_rayir cannot be combined with resume")
if args_cli.warm_start_dwb is not None:
    if args_cli.policy_lr is None or args_cli.warm_start_sha256 is None:
        parser.error("--warm_start_dwb needs --policy_lr, --terrain_lr and --warm_start_sha256")
    if not os.path.isfile(args_cli.warm_start_dwb):
        parser.error(f"--warm_start_dwb not found: {args_cli.warm_start_dwb}")
if args_cli.warm_start_gast is not None:
    if args_cli.policy_lr is None:
        parser.error("--warm_start_gast needs --policy_lr and --terrain_lr")
    if not os.path.isfile(args_cli.warm_start_gast):
        parser.error(f"--warm_start_gast not found: {args_cli.warm_start_gast}")
if args_cli.warm_start_bivt is not None:
    if args_cli.policy_lr is None:
        parser.error("--warm_start_bivt needs --policy_lr and --terrain_lr")
    if not os.path.isfile(args_cli.warm_start_bivt):
        parser.error(f"--warm_start_bivt not found: {args_cli.warm_start_bivt}")
    if getattr(args_cli, "resume", False):
        parser.error("--warm_start_bivt starts a new GAST run; resume a warm-started run with --resume (and the same lrs)")
if args_cli.total_envs is not None and (not args_cli.distributed or args_cli.total_envs < int(os.environ.get("WORLD_SIZE", "1"))):
    parser.error("--total_envs requires --distributed and at least one environment per process")

# The actor terrain target is masked by rendered camera visibility.
if args_cli.video:
    parser.error('GAST teacher is rendering-free')
args_cli.enable_cameras = False

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

runtime_root = prepare_vrl_runtime(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
verify_vrl_runtime(runtime_root)

import importlib
import json
import time
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
import gd_lab.gast.tasks
from gd_lab.core.paths import LOG_ROOT
from gd_lab.deploy.metadata import capture_context
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.mdp.terrain_families import family_column_masks
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
    if args_cli.total_envs is not None:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        base, remainder = divmod(args_cli.total_envs, world_size)
        env_cfg.scene.num_envs = base + (rank < remainder)
        print(f"[INFO] Distributed environments: rank {rank}/{world_size} has {env_cfg.scene.num_envs} of {args_cli.total_envs}")
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    resume149 = None
    if args_cli.resume_rayir149:
        if agent_cfg.resume or args_cli.policy_lr is None or not args_cli.warm_start_sha256:
            raise ValueError("UPDATE149B requires explicit SHA/grouped LR and no generic resume")
        from gd_lab.gast.resume149 import validate_resume
        resume149 = validate_resume(args_cli.resume_rayir149, args_cli.warm_start_sha256, args_cli.task)
        print('[UPDATE149B source] ' + json.dumps(resume149), flush=True)
        # Approved mature-source protocol: do not restart force/speed ramps.
        os.environ['GD_LAB_V2_FORCE_RAMP_STEPS'] = '0'
        os.environ['GD_LAB_V21_SPEED_RAMP_STEPS'] = '0'
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.distributed:
        if args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training requires a GPU device.")
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        # Global rank: with several nodes the local rank repeats, and equal seeds duplicate rollouts across nodes.
        seed = agent_cfg.seed + int(os.environ.get("RANK", app_launcher.local_rank))
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    log_dir = os.environ.get("GAST_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging run in: {log_dir}")
    env_cfg.log_dir = log_dir

    if env_cfg.uses_cameras:
        raise ValueError('Teacher must have no camera sensors')
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
    columns = {name: indices.cpu().tolist() for name, indices in family_column_masks(env.unwrapped).items()}
    runner.configure_online_top5(
        columns, spacing=agent_cfg.top5_min_spacing,
        min_platform_gap_mean_level=agent_cfg.top5_min_platform_gap_mean_level,
    )
    # A capture failure must not cost a training run; export refuses later
    # rather than guessing.
    try:
        runner.deploy_context = capture_context(env, runner.alg.policy)
    except (ValueError, AttributeError, KeyError, TypeError) as exc:
        print(f"[WARN] Deployment metadata unavailable; checkpoints will not carry it: {exc}")
    if not resume149:
        runner.add_git_repo_to_log(__file__)
    warm_start = None
    if args_cli.policy_lr is not None:
        from gd_lab.gast.warm_start import (build_grouped_optimizer, warm_start_from_bivt, warm_start_from_dwb,
                                            warm_start_from_gast, warm_start_from_rayir)
        if any(x is not None for x in (args_cli.warm_start_bivt, args_cli.warm_start_gast, args_cli.warm_start_dwb, args_cli.warm_start_rayir)):
            if args_cli.warm_start_rayir is not None:
                warm_start = warm_start_from_rayir(runner.alg, args_cli.warm_start_rayir, args_cli.policy_lr,
                    args_cli.terrain_lr, expected_sha256=args_cli.warm_start_sha256, expected_task=args_cli.task)
            elif args_cli.warm_start_dwb is not None:
                warm_start = warm_start_from_dwb(runner.alg, args_cli.warm_start_dwb, args_cli.policy_lr,
                                                 args_cli.terrain_lr, expected_sha256=args_cli.warm_start_sha256)
            elif args_cli.warm_start_gast is not None:
                warm_start = warm_start_from_gast(runner.alg, args_cli.warm_start_gast, args_cli.policy_lr,
                                                  args_cli.terrain_lr, expected_sha256=args_cli.warm_start_sha256)
            else:
                warm_start = warm_start_from_bivt(runner.alg, args_cli.warm_start_bivt, args_cli.policy_lr,
                                                  args_cli.terrain_lr, expected_sha256=args_cli.warm_start_sha256)
            means = warm_start.get("terrain_level_means_by_family") or {}
            family_columns = getattr(runner, "_top5_family_columns", None)
            if args_cli.warm_start_rayir is not None and (not family_columns or set(family_columns) - set(means)):
                raise ValueError("UPDATE149 requires saved levels for every terrain family")
            if means and family_columns:
                from gd_lab.rl.terrain_resume import restore_family_start_levels
                scene = env.unwrapped.scene
                warm_start["terrain_levels_restored"] = bool(
                    restore_family_start_levels(scene.terrain, scene.env_origins, family_columns, means))
            if args_cli.warm_start_rayir is not None and not warm_start.get("terrain_levels_restored"):
                raise ValueError("UPDATE149 terrain level restoration failed; refusing to start")
            if args_cli.warm_start_rayir is not None:
                env.reset()  # align robot pose/history with the restored family origins
                print("[UPDATE149] Native weights copied, new PPO, restored family levels, fresh iteration 0", flush=True)
        else:  # resuming a warm-started run: the saved optimizer state has the two groups
            if agent_cfg.algorithm.schedule != "fixed":
                raise ValueError("per-group learning rates need agent.algorithm.schedule=fixed")
            runner.alg.optimizer = build_grouped_optimizer(
                runner.alg.policy, args_cli.policy_lr, args_cli.terrain_lr, runner.alg.optimizer)
            runner.alg.learning_rate = args_cli.policy_lr
    if resume149:
        runner.strict_resume149 = resume149
        runner.load(args_cli.resume_rayir149)
        from gd_lab.gast.resume149 import verify_loaded
        resume149['loaded_state_check'] = verify_loaded(runner.alg, args_cli.resume_rayir149)
        if runner.current_learning_iteration != resume149['start_iteration']:
            raise ValueError('UPDATE149B next iteration mismatch')
        raw = env.unwrapped
        raw._aug149_diagnostics = True
        raw._aug149_rank_after_step = env_cfg.mapper_aug_ramp_steps
        term = raw._gast_terrain_term
        term.step_offset = resume149['absolute_step_offset']
        from gd_lab.gast.resume149 import reset_restored_env
        reset_restored_env(env)  # noise clock restored; mapper clock stays zero
        runner._top5_records.clear()
        runner.aug149_start_iteration = runner.current_learning_iteration
        runner.aug149_manifest = {**resume149, 'force_scale':1., 'speed_ceiling':1.2,
            'noise_strength':min(1., .1+term.step_offset/200000.), 'clean_fraction':.25,
            'mapper_ramp_steps':env_cfg.mapper_aug_ramp_steps, 'mapper_enabled':env_cfg.mapper_aug_enabled}
        print('[UPDATE149B restored] ' + json.dumps(runner.aug149_manifest), flush=True)
    if args_cli.warm_start_rayir is not None and agent_cfg.resume:
        raise ValueError("UPDATE149 warm start cannot use agent.resume=true")
    if agent_cfg.resume:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
        terrain_term = getattr(env.unwrapped, '_gast_terrain_term', None)
        if terrain_term is not None:
            terrain_term.step_offset = runner.current_learning_iteration * agent_cfg.num_steps_per_env
            print(f'[INFO] GAST noise ramp resumes at step offset {terrain_term.step_offset}')
    if args_cli.policy_lr is not None:
        # Recorded AFTER any load: the rates actually in the optimizer, and on a resume the original
        # warm-start provenance carried by the checkpoint (never replaced by a placeholder).
        from gd_lab.gast.warm_start import check_resumed_rates, resumed_provenance
        actual = check_resumed_rates(runner.alg.optimizer, args_cli.policy_lr, args_cli.terrain_lr)
        record = warm_start if warm_start is not None else resumed_provenance(args_cli.resume_rayir149 if resume149 else resume_path)
        manager = env.unwrapped.reward_manager
        gap_terms = [n for n in ("platform_gap_foot_drop", "platform_gap_crossing", "platform_gap_monitor",
                                 "platform_gap_intrusion", "platform_gap_clean", "gap_foothold_margin",
                                 "stair_foothold_margin", "gap_slot_probe", "stair_handle_disturbance",
                                 "stair_push_fall", "stair_push_slip") if n in manager.active_terms]
        runner.gast_warm_start = {
            **record, "task": args_cli.task,
            "train_arm_env": os.environ.get("TRAIN_ARM"),
            "reward_weights": {n: float(manager.get_term_cfg(n).weight) for n in gap_terms},
            "learning_rates_requested": {"policy": args_cli.policy_lr, "terrain": args_cli.terrain_lr},
            "learning_rates": actual,
            "schedule": agent_cfg.algorithm.schedule,
            "mapper_augmentation": {name: getattr(env_cfg, name, None) for name in (
                "mapper_aug_enabled", "mapper_aug_ramp_steps", "mapper_aug_clean_fraction", "mapper_aug_seed")}}
        runner.gast_warm_start.pop("terrain_level_means_by_family", None)
        print(f"[INFO] GAST warm start: {runner.gast_warm_start}", flush=True)

    if not runner.disable_logs:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        if getattr(runner, "gast_warm_start", None):
            dump_yaml(os.path.join(log_dir, "params", "gast_warm_start.yaml"), runner.gast_warm_start)

    started = time.perf_counter()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=not bool(resume149))
    completion = {'completed_updates': agent_cfg.max_iterations, 'wall_seconds': time.perf_counter()-started,
                  'checkpoint': os.path.join(log_dir, f'model_{runner.current_learning_iteration}.pt'),
                  'num_envs_per_rank0': env.unwrapped.num_envs,
                  'total_envs': getattr(runner, '_global_env_count', env.unwrapped.num_envs),
                  'horizon': agent_cfg.num_steps_per_env}
    augmentation = getattr(getattr(env.unwrapped, "_gast_terrain_term", None), "mapper_aug", None)
    if augmentation is not None:
        completion["mapper_aug_last_capture"] = {k: (v.item() if torch.is_tensor(v) else v)
            for k, v in augmentation.last_diagnostics.items()}
        print(f"[UPDATE149] final augmentation diagnostics: {completion['mapper_aug_last_capture']}", flush=True)
    if not runner.disable_logs and not getattr(runner, 'stopped_early', False):
        with open(os.path.join(log_dir, 'completion.json'), 'w') as handle:
            json.dump(completion, handle, indent=2)
        print('[GAST_COMPLETE] '+json.dumps(completion), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
