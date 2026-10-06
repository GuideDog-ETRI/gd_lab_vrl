"""Blind-start VRL trainer (stage 2-A / 2-B).

Copy of scripts/train_cvtt.py that adds the gd_lab.teachers.bivt tasks, skips the
cameras on the camera-free task, and can warm-start from a blind DreamWaQ
checkpoint (--blind_init). The original trainer is left unchanged.

2-A: --task Gd-VrlBlindStart-Rbq10-Dreamwaq-v0 --blind_init <blind model_*.pt>
2-B: --task Gd-VrlBlindStart-Rbq10-Dreamwaq-Vision-v0 --resume --load_run <2-A run> --checkpoint <model>
"""

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from gd_lab.core.experiments import training_arm_overrides
from gd_lab.teachers.bivt.gap_guard import (
    ALL_GAP_TASK_IDS,
    CAMERA_FREE_TASK_IDS,
    GAP_TASK_IDS,
    V2_TASK_IDS,
    inspect_checkpoint_v2,
    verify_restored_iteration_v2,
    OBSERVATION_VERSION,
    RAYCAST_TASK_IDS,
    apply_force_ppo_lr,
    arm_of,
    cenet_lr_report,
    inspect_checkpoint,
    sha256_file,
    validate_cli,
    validate_gate_record,
    verify_restored_iteration,
)
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Train an RSL-RL agent on a gd_lab task.")
parser.add_argument("--task", type=str, default="Gd-VrlBlindStart-Rbq10-Dreamwaq-v0", help="Name of the task.")
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
parser.add_argument("--blind_init", type=str, default=None, help="Blind DreamWaQ model_*.pt to warm-start the policy from.")
parser.add_argument('--resume_checkpoint', type=str, help='Absolute checkpoint path, including Top5 checkpoints.')
parser.add_argument('--resume_sha256', type=str, default=None,
                    help='v2 gap/stair task only: sha256 of the Clean-arm --resume_checkpoint (required there).')
parser.add_argument('--target_iterations', type=int, help='Total completed PPO updates, not additional updates.')
parser.add_argument('--force_ppo_lr', type=float, default=None,
                    help='Gap fine-tuning only: fixed PPO LR applied after the checkpoint is loaded (required there).')
parser.add_argument('--baseline_gate', type=str, default=None,
                    help='Gap fine-tuning training only: structured gate-C record (see gd_lab.teachers.bivt.gap_guard).')
parser.add_argument('--baseline_gate_waiver', type=str, default=None,
                    help='Gap fine-tuning training only: explicit gate-C waiver; the reason is recorded in the manifest.')
parser.add_argument('--rollout_only_steps', type=int, default=None,
                    help='Gap fine-tuning smoke: run the loaded policy for N env steps and exit; no PPO update.')
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.resume_checkpoint:
    if not Path(args_cli.resume_checkpoint).is_file():
        parser.error('Resume checkpoint does not exist')
    args_cli.resume = True
if args_cli.total_envs is not None and (not args_cli.distributed or args_cli.total_envs < int(os.environ.get("WORLD_SIZE", "1"))):
    parser.error("--total_envs requires --distributed and at least one environment per process")

# Only the camera-free tasks (2-A, 2-R, gap fine-tuning arms) run without rendered cameras.
args_cli.enable_cameras = args_cli.task not in CAMERA_FREE_TASK_IDS
try:  # fail closed before the simulator starts
    validate_cli(
        args_cli.task, resume_checkpoint=args_cli.resume_checkpoint, force_ppo_lr=args_cli.force_ppo_lr,
        blind_init=args_cli.blind_init, distributed=args_cli.distributed,
        rollout_only_steps=args_cli.rollout_only_steps, target_iterations=args_cli.target_iterations,
        baseline_gate=args_cli.baseline_gate, baseline_gate_waiver=args_cli.baseline_gate_waiver,
        resume_sha256=args_cli.resume_sha256,
    )
except ValueError as exc:
    parser.error(str(exc))
if args_cli.blind_init is not None and args_cli.resume:
    parser.error("--blind_init starts a new run; it cannot be combined with --resume")

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
from datetime import datetime

import gd_lab  # noqa: F401  (registers the tasks)
import gd_lab.teachers.bivt  # noqa: F401  (registers the blind-start tasks)
import gymnasium as gym
import torch
from gd_lab.core.paths import LOG_ROOT
from gd_lab.deploy.metadata import capture_context
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.mdp.platform_gap_attempts import EpisodeArchive
from gd_lab.mdp.terrain_families import family_column_masks
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
from gd_lab.teachers.bivt.blind_init import load_blind_checkpoint
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _resolve(path: str) -> type[OnPolicyRunner]:
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


def _finalize_gap_run(env, runner, agent_cfg, env_cfg, checkpoint) -> dict:
    """Gap arms only: prove the 17206 state was restored, pin the PPO LR, record the effective setup."""
    if args_cli.task in V2_TASK_IDS:
        verify_restored_iteration_v2(runner.current_learning_iteration, checkpoint)
    else:
        verify_restored_iteration(runner.current_learning_iteration, checkpoint)  # load() resumes at iter + 1
    ppo_lr = apply_force_ppo_lr(runner.alg, args_cli.force_ppo_lr)  # after load: load restores the saved LR
    cenet = cenet_lr_report(runner.alg.policy)
    if not cenet["as_expected"]:
        print(f"[WARN] CENet LR differs from the expected value: {cenet}", flush=True)
    context = getattr(runner, "observation_context", None) or {}
    if context.get("version") != OBSERVATION_VERSION:
        raise RuntimeError(f"observation contract {context.get('version')!r} != {OBSERVATION_VERSION!r}")
    gate = None
    if not args_cli.rollout_only_steps:  # defense in depth: training never starts without a validated gate record
        if args_cli.baseline_gate_waiver is not None:
            gate = {"waived": True, "reason": args_cli.baseline_gate_waiver.strip()}
            print(f"[WARN] Gate C waived for this run: {gate['reason']}", flush=True)
        elif args_cli.task in V2_TASK_IDS:
            raise RuntimeError("v2 training needs a gate waiver")
        else:
            validate_gate_record(args_cli.baseline_gate)
            gate = {"path": str(args_cli.baseline_gate), "sha256": sha256_file(args_cli.baseline_gate)}
    manager = env.unwrapped.reward_manager
    terms = [name for name in ("platform_gap_monitor", "platform_gap_intrusion", "platform_gap_clean",
                               "gap_foothold_margin", "stair_foothold_margin", "gap_slot_probe",
                               "stair_handle_disturbance", "stair_push_fall", "stair_push_slip")
             if name in manager.active_terms]
    manifest = {
        "task": args_cli.task, "arm": arm_of(args_cli.task), "checkpoint": checkpoint,
        "camera_profile": env_cfg.camera_profile,
        "resumed_iteration": int(runner.current_learning_iteration), "ppo_lr": ppo_lr, "cenet_lr": cenet,
        "baseline_gate": gate,
        "reward_weights": {name: float(manager.get_term_cfg(name).weight) for name in terms},
        "train_arm_env": os.environ.get("TRAIN_ARM"), "hydra_overrides": list(hydra_args),
        "seed": int(agent_cfg.seed), "num_envs": int(env_cfg.scene.num_envs),
        "distributed": bool(args_cli.distributed), "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "total_envs": args_cli.total_envs,
        "source_commit": os.environ.get("GD_LAB_SOURCE_COMMIT"), "patch_sha256": os.environ.get("GD_LAB_PATCH_SHA256"),
    }
    print(f"[INFO] Gap fine-tuning manifest: {manifest}", flush=True)
    return manifest


def _run_rollout_only(env, runner, steps: int) -> None:
    """Server smoke (UNVERIFIED on this PC): exercise the manager graph; no PPO update, no optimizer step.

    The tracker is wiped on every env reset, so the totals come from an EpisodeArchive that snapshots
    each env's first episode right before its reset.
    """
    unwrapped = env.unwrapped
    monitor = unwrapped.reward_manager.get_term_cfg("platform_gap_monitor").func
    archive = EpisodeArchive(monitor.tracker)
    archive.install(monitor)
    policy = runner.get_inference_policy(device=unwrapped.device)
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    log_keys, resets = set(), 0
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, dones, extras = env.step(policy(obs))
            resets += int(dones.sum())
            log_keys.update(k for k in extras.get("log", {}) if "platform_gap_diagnostics" in k)
    stats, records = archive.finish()
    print(f"[SMOKE] steps={steps} total_resets={resets} truncated_envs={archive.truncated} "
          f"diag_keys={sorted(log_keys)} records={len(records)} stats={stats}", flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if (args_cli.enable_cameras or args_cli.task in RAYCAST_TASK_IDS) and not args_cli.rollout_only_steps:
        from gd_lab.core.camera_contract import require_training_camera_profile
        require_training_camera_profile(env_cfg.camera_profile)  # legacy calibration only with explicit opt-in
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

    if getattr(env_cfg, "uses_cameras", True) != args_cli.enable_cameras:
        raise ValueError(f"{args_cli.task}: camera flag does not match the env config")
    if args_cli.enable_cameras:
        configure_vrl_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ensure_prev_prev_action_tracking(env.unwrapped)

    if agent_cfg.resume:
        resume_path = args_cli.resume_checkpoint or get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

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
    if args_cli.task in RAYCAST_TASK_IDS:
        from gd_lab.core.camera_transport import CameraTransportConfig
        runner.observation_context = dict(version='bivt_ray_occlusion_v2',
            occlusion='terrain + leg capsules + trunk OBB + foot spheres',
            boundary='four-neighbour depth spread <= tolerance',
            transport=CameraTransportConfig().manifest(env.unwrapped.step_dt),
            proxy_caveat='Foot collision spheres are conservative, not exact rendered visual meshes')
    # A capture failure must not cost a training run; export refuses later
    # rather than guessing.
    try:
        runner.deploy_context = capture_context(env, runner.alg.policy)
    except (ValueError, AttributeError, KeyError, TypeError) as exc:
        print(f"[WARN] Deployment metadata unavailable; checkpoints will not carry it: {exc}")
    runner.add_git_repo_to_log(__file__)
    gap_checkpoint = None
    if agent_cfg.resume:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        if args_cli.task in V2_TASK_IDS:
            gap_checkpoint = inspect_checkpoint_v2(resume_path, args_cli.resume_sha256, args_cli.target_iterations)
        elif args_cli.task in GAP_TASK_IDS:
            gap_checkpoint = inspect_checkpoint(resume_path)  # raises if any required state is missing
        runner.load(resume_path)
    if args_cli.blind_init is not None:
        blind_infos = load_blind_checkpoint(runner.alg.policy, args_cli.blind_init)
        if "learning_rate" in blind_infos:
            runner.alg.learning_rate = blind_infos["learning_rate"]
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = runner.alg.learning_rate
        print(f"[INFO] Warm-started from blind checkpoint: {args_cli.blind_init}", flush=True)

    gap_manifest = None
    if args_cli.task in ALL_GAP_TASK_IDS:
        gap_manifest = _finalize_gap_run(env, runner, agent_cfg, env_cfg, gap_checkpoint)
        runner.gap_finetune_manifest = gap_manifest

    if args_cli.target_iterations is not None:
        agent_cfg.max_iterations = args_cli.target_iterations - runner.current_learning_iteration
        if agent_cfg.max_iterations <= 0:
            raise ValueError('Checkpoint has already reached the requested target')
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if gap_manifest is not None:
        dump_yaml(os.path.join(log_dir, "params", "gap_finetune_manifest.yaml"), gap_manifest)
    if args_cli.rollout_only_steps:
        _run_rollout_only(env, runner, args_cli.rollout_only_steps)
        env.close()
        return

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
