"""Vision-RL stage 3: distill the frozen terrain-encoder teacher into the
4-camera CNN-GRU student (``gd_lab.rl.perception.CameraPerceptionEncoder``).

Rolls out the trained stage-1/2 policy (frozen, in inference mode -- it
decides the robot's motion, exactly like deployment will) and, every time the
belly cameras refresh, regresses the student's latent onto the teacher
terrain_encoder's privileged latent for that same instant. This mirrors
APT-RL's teacher-student split (Fig. 2iii): the actor and terrain_encoder are
never touched here, only the student's own weights are trained.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Distill the vision-RL terrain-encoder teacher into a camera student.")
parser.add_argument(
    "--task", type=str, default="Gd-Vrl-Rbq10-Dreamwaq-Vision-v0", help="Camera-enabled env task (not the teacher's own training task -- see VisionRoughEnvCfg)."
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
parser.add_argument(
    "--num_envs", type=int, default=256, help="Number of environments (camera rendering is expensive)."
)
parser.add_argument("--iterations", type=int, default=2000, help="Camera ticks; one optimizer update per BPTT window.")
parser.add_argument("--lr", type=float, default=1.0e-3, help="Student optimizer learning rate.")
parser.add_argument("--save_interval", type=int, default=200, help="Iterations between student checkpoints.")
parser.add_argument("--gru_hidden_dim", type=int, default=64, help="Student GRU hidden size.")
parser.add_argument(
    "--bptt_steps",
    type=int,
    default=8,
    help="Camera ticks per truncated-BPTT window before one backward() call. "
    "1 reproduces the old per-tick-detached behavior; >1 lets the GRU's hidden "
    "state actually receive gradient across time, which is what lets it learn to "
    "integrate a hazard (a gap/step edge) across several ticks instead of only "
    "ever being trained on a single, possibly sensor-noise-corrupted frame.",
)
parser.add_argument(
    "--hazard_loss_coef",
    type=float,
    default=0.5,
    help="Weight on the auxiliary hazard-proximity loss (see CameraPerceptionEncoder.hazard_head). "
    "Ground truth comes from the privileged height_scan, not the camera frames, so this term "
    "keeps pushing the shared CNN+GRU trunk toward hazard-relevant features even on ticks where "
    "the (possibly noise-hotspot-corrupted) camera frames alone wouldn't give a clean signal.",
)
parser.add_argument("--perception_run_name", type=str, default=None, help="Perception log subfolder name.")
parser.add_argument(
    "--no_camera_noise",
    action="store_true",
    default=False,
    help="Disable depth/IR sensor-noise domain randomization (see CameraNoiseCfg) -- "
    "off by default means noise IS applied; only disable for an apples-to-apples "
    "comparison against an older noiseless run.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if min(args_cli.iterations, args_cli.bptt_steps, args_cli.save_interval, args_cli.num_envs) <= 0:
    parser.error("iterations, bptt_steps, save_interval and num_envs must be positive")
if args_cli.lr <= 0 or args_cli.hazard_loss_coef < 0:
    parser.error("lr must be positive and hazard_loss_coef nonnegative")
args_cli.enable_cameras = True  # this script always renders the belly cameras

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
import os
import time

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from torch.utils.tensorboard import SummaryWriter

import gd_lab  # noqa: F401  (registers the tasks)
from gd_lab.core.paths import LOG_ROOT
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic
from gd_lab.rl.perception import CameraPerceptionEncoder, height_discontinuity_metres
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
from gd_lab.tasks.vrl_rough import CameraNoiseCfg, belly_camera_frames, noisy_camera_frames


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
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
        raise TypeError(f"Stage-3 distillation requires a DreamwaqVrlActorCritic teacher, got {type(teacher)}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    device = env.unwrapped.device

    dt = env.unwrapped.step_dt
    cam_period = env.unwrapped.scene["front_depth_camera0"].cfg.update_period
    camera_steps = max(1, round(cam_period / dt))
    print(f"[INFO] Camera updates every {camera_steps} env steps (update_period={cam_period}s, step_dt={dt}s)")

    student = CameraPerceptionEncoder(
        num_cameras=4,
        gru_hidden_dim=args_cli.gru_hidden_dim,
        latent_dim=agent_cfg.policy.terrain_latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)

    camera_noise_cfg = None if args_cli.no_camera_noise else CameraNoiseCfg()
    print(f"[INFO] Camera sensor-noise domain randomization: {'OFF' if camera_noise_cfg is None else camera_noise_cfg}")

    run_name = args_cli.perception_run_name or time.strftime("%Y-%m-%d_%H-%M-%S") + "_perception"
    log_dir = os.path.join(log_root_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Perception log dir: {log_dir}")

    obs = env.get_observations()
    hidden = student.init_hidden(env.unwrapped.num_envs, device)

    it = 0
    while it < args_cli.iterations:
        window_len = min(args_cli.bptt_steps, args_cli.iterations - it)
        optimizer.zero_grad()
        window_loss = torch.zeros((), device=device)
        window_mse_sum, window_hazard_sum = 0.0, 0.0

        # Truncated BPTT window: hidden is NOT detached between these
        # window_len ticks, so one backward() at the end of the window
        # propagates gradient through the GRU across all of them -- unlike
        # detaching every tick (bptt_steps=1), this actually trains the
        # recurrence to integrate/remember across time, not just "given
        # whatever hidden happens to be, fit this one frame."
        for _ in range(window_len):
            for _ in range(camera_steps):
                with torch.no_grad():
                    actions = teacher.act_inference(obs)
                obs, _, dones, _ = env.step(actions)
                if dones.any():
                    keep = (~dones.reshape(-1).bool()).unsqueeze(-1).to(hidden.dtype)
                    hidden = hidden * keep  # stays in-graph on purpose; see module docstring

            if camera_noise_cfg is not None:
                frames = noisy_camera_frames(env.unwrapped.scene, camera_noise_cfg)
            else:
                frames = belly_camera_frames(env.unwrapped.scene)
            student_latent, hidden = student(frames, hidden)
            hazard_pred = student.hazard_head(hidden).squeeze(-1)

            with torch.no_grad():
                teacher_latent = teacher.terrain_latent(obs)
                raw_height_scan = obs[teacher.obs_groups["critic"][0]][..., teacher._height_scan_slice]
                hazard_label = height_discontinuity_metres(
                    raw_height_scan, env_cfg.observations.critic.height_scan.scale,
                    teacher.terrain_encoder.grid_shape,
                )

            latent_loss = F.mse_loss(student_latent, teacher_latent)
            hazard_loss = F.mse_loss(hazard_pred, hazard_label)
            window_loss = window_loss + latent_loss + args_cli.hazard_loss_coef * hazard_loss
            window_mse_sum += latent_loss.item()
            window_hazard_sum += hazard_loss.item()

            it += 1

        (window_loss / window_len).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        hidden = hidden.detach()  # window boundary: gradient stops here, not mid-window

        mse_val = window_mse_sum / window_len
        hazard_val = window_hazard_sum / window_len
        writer.add_scalar("perception/mse", mse_val, it)
        writer.add_scalar("perception/hazard_mse", hazard_val, it)
        if (it // window_len) % 10 == 0:
            print(f"[INFO] iter {it}/{args_cli.iterations} mse={mse_val:.5f} hazard_mse={hazard_val:.5f}")

        if it % args_cli.save_interval < window_len or it >= args_cli.iterations:
            ckpt_path = os.path.join(log_dir, f"perception_{it}.pt")
            torch.save({"model": student.state_dict(), "iteration": it}, ckpt_path)
            print(f"[INFO] Saved {ckpt_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
