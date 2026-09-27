"""Closed-loop validation for vision-RL stage 3: run the frozen stage-1 actor
with the terrain_encoder's privileged latent REPLACED by the stage-3 camera
student's own output, entirely inside Isaac Sim.

Uses the same external terrain-latent substitution as the deployment actor.

Uses VisionRoughEnvCfg_PLAY (fixed difficulty, no curriculum drift, no domain
randomization noise) rather than the training task, so different student/
teacher checkpoint combinations can be compared under identical conditions.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from gd_lab.core.camera_transport import CameraTransport, CameraTransportConfig, delivery_mask
from gd_lab.core.experiments import training_arm_overrides
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
parser.add_argument("--camera_interval_ms", type=float, nargs=2, default=(80.0, 80.0), metavar=("MIN", "MAX"))
parser.add_argument("--camera_delay_ms", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"))
parser.add_argument("--camera_drop_prob", type=float, default=0.0)
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
try:
    transport_config = CameraTransportConfig(tuple(args_cli.camera_interval_ms),
                                            tuple(args_cli.camera_delay_ms), args_cli.camera_drop_prob)
except ValueError as exc:
    parser.error(str(exc))

try:
    arm_overrides = training_arm_overrides(os.environ.get("TRAIN_ARM"), play=True)
except ValueError as exc:
    parser.error(str(exc))
arm_overrides = [item.replace("agent.experiment_name=blind_", "agent.experiment_name=vision_") for item in arm_overrides]
sys.argv = [sys.argv[0]] + arm_overrides + hydra_args

runtime_root = prepare_vrl_runtime(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
verify_vrl_runtime(runtime_root)

import importlib
import time

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import gd_lab  # noqa: F401  (registers the tasks)
from gd_lab.core.camera_contract import camera_contract_for_policy
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
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Teacher checkpoint: {checkpoint_path}")

    configure_vrl_cameras(env_cfg)
    env_cfg.sim.render_interval = env_cfg.decimation
    transport = CameraTransport(transport_config, env_cfg.decimation * env_cfg.sim.dt, agent_cfg.seed)
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
    camera_contract = camera_contract_for_policy(env_cfg.camera_profile, env.unwrapped.step_dt)
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
    print(f"[INFO] Nominal camera contract: {camera_steps} env steps ({cam_period}s), step_dt={dt}s")
    print(f"[INFO] Evaluation camera transport: {transport_config.manifest(dt)}")

    hidden = student.init_hidden(env.unwrapped.num_envs, device)
    terrain_latent = torch.zeros(env.unwrapped.num_envs, agent_cfg.policy.terrain_latent_dim, device=device)

    obs = env.get_observations()
    step_i = 0
    camera_update_i = 0
    episode_ids = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
    last_camera_step = torch.full_like(episode_ids, -1)
    last_receive_step = torch.full_like(episode_ids, -1)
    capture_step = env.unwrapped.common_step_counter
    while simulation_app.is_running() and (args_cli.max_steps == 0 or step_i < args_cli.max_steps):
        start = time.time()
        with torch.inference_mode():
            common_step = env.unwrapped.common_step_counter
            if common_step == capture_step:
                snapshot = getattr(env.unwrapped, "_vrl_camera_snapshot", None)
                stamps = getattr(env.unwrapped, "_vrl_camera_snapshot_steps", None)
                if snapshot is None or stamps is None or not (stamps == common_step).all():
                    raise RuntimeError("Scheduled camera capture did not produce fresh images")
                transport.capture(common_step, (snapshot[0].clone(), teacher.terrain_latent(obs).clone(), episode_ids.clone()))
                capture_step = transport.next_capture(common_step)
            env.unwrapped._vrl_camera_capture_step = capture_step
            for packet in transport.receive(common_step):
                frames, capture_target, captured_episodes = packet.payload
                refresh = delivery_mask(packet.capture_step, captured_episodes, episode_ids, last_camera_step)
                if not refresh.any():
                    continue
                new_latent, new_hidden = student(frames[refresh], hidden[refresh])
                terrain_latent[refresh], hidden[refresh] = new_latent, new_hidden
                last_camera_step[refresh] = packet.capture_step
                last_receive_step[refresh] = common_step

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
                    latent_mse = torch.nn.functional.mse_loss(new_latent, capture_target[refresh]).item()
                    print(
                        f"[VISION] step={step_i} depth_mean={depth.mean().item():.3f} "
                        f"depth_real_return_frac={(depth < 0.999).float().mean().item():.3f} "
                        f"ir_mean={ir.mean().item():.3f} "
                        f"student_latent_norm={terrain_latent.norm(dim=-1).mean().item():.3f} "
                        f"capture_teacher_latent_mse={latent_mse:.5f} "
                        f"delay_ms={(common_step - packet.capture_step) * dt * 1000:.1f}",
                        flush=True,
                    )
                camera_update_i += 1

            # Match deployment's 250 ms latent expiry / existing zero fallback.
            expired = (last_receive_step < 0) | ((common_step - last_receive_step) * dt >= 0.25)
            terrain_latent[expired] = 0
            actions = teacher.act_with_terrain_latent(obs, terrain_latent)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Student closed-loop actions are non-finite")
            obs, _, dones, _ = env.step(actions)
            if dones.any():
                episode_ids += dones.reshape(-1).long()
                keep = (~dones.reshape(-1).bool()).unsqueeze(-1).to(hidden.dtype)
                hidden = hidden * keep
                terrain_latent = terrain_latent * keep
                last_camera_step[dones.reshape(-1).bool()] = -1
                last_receive_step[dones.reshape(-1).bool()] = -1
        step_i += 1
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
