"""Closed-loop validation for vision-RL stage 3: run the frozen stage-1 actor
with the terrain_encoder's privileged latent REPLACED by the stage-3 camera
student's own output, entirely inside Isaac Sim.

Uses the same external terrain-latent substitution as the deployment actor.

Uses VisionRoughEnvCfg_PLAY (fixed difficulty, no curriculum drift, no domain
randomization noise) rather than the training task, so different student/
teacher checkpoint combinations can be compared under identical conditions.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Closed-loop test: frozen teacher actor + stage-3 camera student.")
parser.add_argument(
    "--task", type=str, default="Gd-Vrl-Rbq10-Dreamwaq-VisionPlay-v0", help="Camera-enabled play task."
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to simulate.")
parser.add_argument("--student_checkpoint", type=str, required=True, help="Path to a perception_*.pt (stage-3) file.")
parser.add_argument("--gru_hidden_dim", type=int, default=64, help="Must match the student's training-time value.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many policy steps (0: unlimited).")
parser.add_argument(
    "--diag_every",
    type=int,
    default=25,
    help="Print camera/latent diagnostics every N camera updates (0 to disable).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # this script always renders the belly cameras

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
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_timing import camera_refresh_mask
from gd_lab.core.paths import LOG_ROOT
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic
from gd_lab.rl.perception import CameraPerceptionEncoder
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
    print(f"[INFO] Teacher checkpoint: {checkpoint_path}")

    configure_vrl_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class = _resolve(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path, load_optimizer=False)
    teacher = runner.alg.policy
    if not isinstance(teacher, DreamwaqVrlActorCritic):
        raise TypeError(f"Expected a DreamwaqVrlActorCritic teacher, got {type(teacher)}")
    teacher.eval()
    device = env.unwrapped.device

    student = CameraPerceptionEncoder(
        num_cameras=4,
        gru_hidden_dim=args_cli.gru_hidden_dim,
        latent_dim=agent_cfg.policy.terrain_latent_dim,
    ).to(device)
    ckpt = torch.load(args_cli.student_checkpoint, map_location=device)
    camera_contract = load_camera_contract(env_cfg.camera_profile)
    recorded = ckpt.get("camera_contract")
    if recorded is not None and recorded != camera_contract.manifest():
        raise ValueError("Student camera contract does not match the environment calibration")
    if recorded is None:
        print("[WARN] Legacy student checkpoint has no camera calibration metadata.")
    student.load_state_dict(ckpt["model"])
    student.eval()
    print(f"[INFO] Student checkpoint: {args_cli.student_checkpoint} (iteration {ckpt.get('iteration')})")

    dt = env.unwrapped.step_dt
    camera_steps = camera_contract.period_steps
    cam_period = camera_steps * dt
    print(f"[INFO] Camera updates every {camera_steps} env steps (update_period={cam_period}s, step_dt={dt}s)")

    hidden = student.init_hidden(env.unwrapped.num_envs, device)
    terrain_latent = torch.zeros(env.unwrapped.num_envs, agent_cfg.policy.terrain_latent_dim, device=device)

    obs = env.get_observations()
    step_i = 0
    camera_update_i = 0
    last_camera_step = torch.full((env.unwrapped.num_envs,), -1, device=device, dtype=torch.long)
    while simulation_app.is_running() and (args_cli.max_steps == 0 or step_i < args_cli.max_steps):
        start = time.time()
        with torch.inference_mode():
            common_step = env.unwrapped.common_step_counter
            refresh = camera_refresh_mask(last_camera_step, common_step, camera_steps)
            if refresh.any():
                snapshot = getattr(env.unwrapped, "_vrl_camera_snapshot", None)
                if snapshot is None:
                    raise RuntimeError("VRL camera snapshot was not produced by the terrain observation")
                frames = snapshot[0]
                new_latent, new_hidden = student(frames[refresh], hidden[refresh])
                terrain_latent[refresh], hidden[refresh] = new_latent, new_hidden
                last_camera_step[refresh] = common_step

                if args_cli.diag_every > 0 and camera_update_i % args_cli.diag_every == 0:
                    # Proves the vision path is actually live: real, moving camera
                    # numbers feeding a latent that (if distillation worked) tracks
                    # what the teacher's own privileged height_scan encoder would
                    # have said for the exact same instant -- same comparison
                    # train_perception.py's loss makes, just for eyeballing here.
                    # The canonical snapshot maps no-hit/inf pixels to 1.0
                    # (see its docstring), so plain isfinite() would trivially
                    # read 1.0 here -- "< 0.999" recovers the same "did this pixel
                    # actually hit something" signal instead.
                    depth = frames[:, :, 0]
                    ir = frames[:, :, 1]
                    teacher_latent = teacher.terrain_latent(obs)
                    latent_mse = torch.nn.functional.mse_loss(terrain_latent, teacher_latent).item()
                    print(
                        f"[VISION] step={step_i} depth_mean={depth.mean().item():.3f} "
                        f"depth_real_return_frac={(depth < 0.999).float().mean().item():.3f} "
                        f"ir_mean={ir.mean().item():.3f} "
                        f"student_latent_norm={terrain_latent.norm(dim=-1).mean().item():.3f} "
                        f"vs_teacher_latent_mse={latent_mse:.5f}",
                        flush=True,
                    )
                camera_update_i += 1

            actions = teacher.act_with_terrain_latent(obs, terrain_latent)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Student closed-loop actions are non-finite")
            obs, _, dones, _ = env.step(actions)
            if dones.any():
                keep = (~dones.reshape(-1).bool()).unsqueeze(-1).to(hidden.dtype)
                hidden = hidden * keep
                terrain_latent = terrain_latent * keep
                last_camera_step[dones.reshape(-1).bool()] = -1
        step_i += 1
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
