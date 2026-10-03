"""Distill a frozen teacher into a four-camera CNN-GRU student.

Images and teacher targets are queued together with timing jitter, delay and
packet loss. Targets use capture time; only student weights receive gradients.
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from gd_lab.core.camera_transport import CameraTransport, CameraTransportConfig, delivery_mask
from gd_lab.core.experiments import training_arm_overrides
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Distill the vision-RL terrain-encoder teacher into a camera student.")
parser.add_argument(
    "--task", type=str, default="Gd-Vrl-Rbq10-Dreamwaq-Vision-v0", help="Camera-enabled env task (not the teacher's own training task -- see VisionRoughEnvCfg)."
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--num_envs", type=int, default=256, help="Number of environments (camera rendering is expensive)."
)
parser.add_argument("--iterations", type=int, default=2000, help="Capture attempts including drops; optimize per nonempty BPTT window.")
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
parser.add_argument("--student_arch", choices=("cnn_gru", "grid_attention_v1"), default="cnn_gru")
parser.add_argument("--teacher_checkpoint", help="Explicit frozen checkpoint; overrides run lookup.")
parser.add_argument("--student_resume", help="Restore student and optimizer; iterations is total target. Environment/history reset.")
parser.add_argument("--student_warmup", type=int, default=1000)
parser.add_argument("--student_ramp", type=int, default=4000)
parser.add_argument("--camera_interval_ms", type=float, nargs=2, default=(70.0, 100.0), metavar=("MIN", "MAX"))
parser.add_argument("--camera_delay_ms", type=float, nargs=2, default=(0.0, 50.0), metavar=("MIN", "MAX"))
parser.add_argument("--camera_drop_prob", type=float, default=0.05, help="Probability of dropping a complete four-camera set.")
parser.add_argument(
    "--no_camera_noise",
    action="store_true",
    default=False,
    help="Disable all student sensor noise, including platform-gap depth ghosts -- "
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
if args_cli.student_warmup < 0 or args_cli.student_ramp < 1:
    parser.error("student_warmup must be nonnegative and student_ramp positive")
try:
    transport_config = CameraTransportConfig(tuple(args_cli.camera_interval_ms),
                                            tuple(args_cli.camera_delay_ms), args_cli.camera_drop_prob)
except ValueError as exc:
    parser.error(str(exc))
args_cli.enable_cameras = True  # this script always renders the belly cameras

try:
    arm_overrides = training_arm_overrides(os.environ.get("TRAIN_ARM"))
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
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from torch.utils.tensorboard import SummaryWriter

import gd_lab  # noqa: F401  (registers the tasks)
if args_cli.task == 'Gd-VrlRayStudent-Rbq10-Dreamwaq-Vision-v0':
    import gd_lab.teachers.bivt.student_task  # noqa: F401
from gd_lab.core.camera_contract import camera_contract_for_policy
from gd_lab.core.paths import LOG_ROOT
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.mdp.camera_noise import augment_student_camera_frames
from gd_lab.mdp.platform_gap_noise import PlatformGapDepthGhost, PlatformGapDepthGhostCfg
from gd_lab.mdp.terrain_families import terrain_family_gate
from gd_lab.teachers.cvtt.actor_critic import DreamwaqVrlActorCritic
from gd_lab.students.alignment import validate_teacher_camera_capture
from gd_lab.students.gavd.distillation import AttentionDistillation
from gd_lab.students.rvld.model import CameraPerceptionEncoder, height_discontinuity_metres
from gd_lab.students.rvld.distillation import TerrainAlignmentHead
from gd_lab.students.gavd.model import GridAttentionStudent, spatial_loss
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
from gd_lab.teachers.cvtt.student_env import CameraNoiseCfg


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


@torch.no_grad()
def capture_teacher_packet(env, obs, teacher, episode_ids, noise_cfg, gap_ghost, camera_contract):
    """Freeze calibrated frames and all teacher targets at one exact capture step."""
    terrain = validate_teacher_camera_capture(env, obs, camera_contract)
    snapshot = env._vrl_camera_snapshot
    step = env.common_step_counter
    gap_envs = terrain_family_gate(env, ("platform_gap",)) == 0
    frames = augment_student_camera_frames(
        snapshot[0].clone(), snapshot, env.scene.env_origins, gap_envs, noise_cfg, gap_ghost
    )
    latent = teacher.terrain_latent(obs).clone()
    base_actor = teacher._actor_input(obs, inference=True)[:, :-32].clone()
    teacher_action = teacher.act_inference(obs).clone()
    cells = teacher._height_scan_slice.stop - teacher._height_scan_slice.start
    if cells != 187 or terrain.shape[-1] != 2 * cells:
        raise RuntimeError(f"Teacher terrain must be [masked height(187), Ray visibility(187)], got {tuple(terrain.shape)}")
    visibility = terrain[..., cells:].bool()
    hazard = height_discontinuity_metres(terrain[..., :cells], 5.0, teacher.terrain_encoder.grid_shape, visibility)
    grid = visibility.reshape(-1, *teacher.terrain_encoder.grid_shape)
    supervised = ((grid[:, 1:, :] & grid[:, :-1, :]).flatten(1).any(1)
                  | (grid[:, :, 1:] & grid[:, :, :-1]).flatten(1).any(1))
    # Packet order is stable across RVLD and GAVD; CameraTransport preserves capture_step.
    return (frames.clone(), latent, terrain.clone(), hazard, supervised,
            visibility.float().mean(-1), episode_ids.clone(), base_actor, teacher_action)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    checkpoint_path = (os.path.abspath(args_cli.teacher_checkpoint) if args_cli.teacher_checkpoint else
                       get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint))
    print(f"[INFO] Teacher checkpoint: {checkpoint_path}")
    teacher_env_yaml = next((parent / "params" / "env.yaml" for parent in Path(checkpoint_path).parents
                             if (parent / "params" / "env.yaml").is_file()), None)
    if teacher_env_yaml is None:
        raise RuntimeError("Teacher package must include params/env.yaml to verify its camera profile")
    teacher_profiles = re.findall(r"(?m)^camera_profile:\s*([A-Za-z0-9_-]+)\s*$", teacher_env_yaml.read_text())
    if len(teacher_profiles) != 1 or teacher_profiles[0] != env_cfg.camera_profile:
        raise RuntimeError(f"Teacher/student camera profile mismatch: teacher={teacher_profiles}, student={env_cfg.camera_profile}")
    print(f"[INFO] Verified teacher/student camera profile={teacher_profiles[0]}", flush=True)
    with open(checkpoint_path, "rb") as teacher_file:
        teacher_sha256 = hashlib.sha256(teacher_file.read()).hexdigest()

    configure_vrl_cameras(env_cfg)
    # A capture may fall on any policy tick. Render at policy boundaries so
    # scheduled camera reads are current, including asynchronous reset rows.
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
        raise TypeError(f"Stage-3 distillation requires a DreamwaqVrlActorCritic teacher, got {type(teacher)}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    device = env.unwrapped.device

    dt = env.unwrapped.step_dt
    camera_contract = camera_contract_for_policy(env_cfg.camera_profile, dt)
    camera_steps = camera_contract.period_steps
    cam_period = camera_steps * dt
    print(f"[INFO] Nominal camera contract: {camera_steps} env steps ({cam_period}s), step_dt={dt}s")
    print(f"[INFO] Camera transport: {transport_config.manifest(dt)}; interval_steps={transport.interval}, delay_steps={transport.delay}")

    student = CameraPerceptionEncoder(
        num_cameras=4,
        gru_hidden_dim=args_cli.gru_hidden_dim,
        latent_dim=agent_cfg.policy.terrain_latent_dim,
    ).to(device)
    attention = None
    alignment_head = None
    if args_cli.student_arch == "grid_attention_v1":
        student = GridAttentionStudent(env_cfg.camera_profile).to(device)
        attention = AttentionDistillation(teacher, student, env.unwrapped.num_envs, device,
                                          args_cli.student_warmup, args_cli.student_ramp)
    else:
        # RVLD keeps its deployable CNN-GRU graph; this map decoder is auxiliary-only.
        alignment_head = TerrainAlignmentHead(args_cli.gru_hidden_dim).to(device)
    trainable = list(student.parameters()) + ([] if alignment_head is None else list(alignment_head.parameters()))
    optimizer = torch.optim.Adam(trainable, lr=args_cli.lr)
    start_iteration = 0
    if args_cli.student_resume:
        saved = torch.load(args_cli.student_resume, map_location=device, weights_only=False)
        if saved['teacher_sha256'] != teacher_sha256 or saved['student_arch'] != args_cli.student_arch:
            raise ValueError('Resume teacher hash or student architecture differs')
        if saved.get('student_alignment_contract') != 2:
            raise ValueError('Resume checkpoint lacks the strict capture-alignment supervision contract; start a new aligned run')
        student.load_state_dict(saved['model'], strict=True)
        if alignment_head is not None:
            if 'alignment_head' not in saved:
                raise ValueError('RVLD resume checkpoint is missing its auxiliary terrain alignment head')
            alignment_head.load_state_dict(saved['alignment_head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_iteration = int(saved['iteration'])
        if not 0 <= start_iteration < args_cli.iterations:
            raise ValueError('Resume iteration must be below total target')
        print(f'[INFO] Student/optimizer resumed at {start_iteration}; physics, curriculum, hidden state and transport reset', flush=True)

    camera_noise_cfg = None if args_cli.no_camera_noise else CameraNoiseCfg()
    gap_ghost = PlatformGapDepthGhost(PlatformGapDepthGhostCfg())
    print(f"[INFO] Camera sensor-noise domain randomization: {'OFF' if camera_noise_cfg is None else camera_noise_cfg}")
    print("[WARN] IR channel is RGB luminance (ir_proxy), not a physical infrared sensor simulation.")

    run_name = args_cli.perception_run_name or time.strftime("%Y-%m-%d_%H-%M-%S") + "_perception"
    log_dir = os.path.join(log_root_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[INFO] Perception log dir: {log_dir}")

    obs = env.get_observations()
    hidden = student.init_hidden(env.unwrapped.num_envs, device)
    episode_ids = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
    last_delivered = torch.full_like(episode_ids, -1)
    rvld_latent = torch.zeros(env.unwrapped.num_envs, agent_cfg.policy.terrain_latent_dim, device=device)
    rvld_ready = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.bool)
    rvld_stamp = torch.full((env.unwrapped.num_envs,), -100.0, device=device)
    learning_started = time.perf_counter()

    it = start_iteration
    while it < args_cli.iterations:
        window_len = min(args_cli.bptt_steps, args_cli.iterations - it)
        optimizer.zero_grad()
        window_loss = torch.zeros((), device=device)
        window_mse_sum, window_hazard_sum = 0.0, 0.0
        window_action_sum, window_geometry_sum = 0.0, 0.0
        window_height_sum, window_visibility_sum = 0.0, 0.0
        window_visible_sum, window_supervised_sum = 0.0, 0.0
        window_updates = 0
        window_delay_sum = 0.0

        # Keep hidden states in-graph across delivered frames in this BPTT window.
        for _ in range(window_len):
            capture_step = transport.next_capture(env.unwrapped.common_step_counter)
            env.unwrapped._vrl_camera_capture_step = capture_step
            # Drain the final capture's transit time before saving the last student.
            stop_step = capture_step + (transport.delay[1] if it + 1 == args_cli.iterations else 0)
            while env.unwrapped.common_step_counter < stop_step:
                with torch.no_grad():
                    if attention:
                        actions = attention.actions(obs, it, env.unwrapped.common_step_counter * dt)
                    else:
                        teacher_actions = teacher.act_inference(obs)
                        age = env.unwrapped.common_step_counter * dt - rvld_stamp
                        fresh = rvld_ready & (age >= 0) & (age < 0.3)
                        student_actions = teacher.act_with_terrain_latent(obs, rvld_latent * fresh[:, None])
                        probability = min(1.0, max(0.0, (it - args_cli.student_warmup) / max(1, args_cli.student_ramp)))
                        use_student = (torch.rand_like(rvld_ready, dtype=torch.float) < probability) & fresh
                        writer.add_scalar("distillation/student_rollout_fraction", use_student.float().mean().item(),
                                          env.unwrapped.common_step_counter)
                        actions = torch.where(use_student[:, None], student_actions, teacher_actions)
                obs, _, dones, _ = env.step(actions)
                if dones.any():
                    done_rows = dones.reshape(-1).bool()
                    keep = (~done_rows).unsqueeze(-1).to(hidden.dtype)
                    hidden = hidden * keep  # stays in-graph on purpose; see module docstring
                    episode_ids += done_rows.long()
                    gap_ghost.reset(done_rows.nonzero(as_tuple=False).flatten())
                    if attention:
                        attention.reset(done_rows)
                    else:
                        rvld_latent[done_rows] = 0
                        rvld_ready[done_rows] = False
                        rvld_stamp[done_rows] = -100.0

                step = env.unwrapped.common_step_counter
                if step == capture_step:
                    payload = capture_teacher_packet(env.unwrapped, obs, teacher, episode_ids,
                                                     camera_noise_cfg, gap_ghost, camera_contract)
                    if attention:
                        payload = (*payload, attention.capture(obs, payload[8], payload[7], payload[2]))
                    transport.capture(step, payload)
                for packet in transport.receive(step):
                    (frames, teacher_latent, teacher_terrain, hazard_label, hazard_supervised,
                     visible, captured_episodes, base_actor, teacher_action) = packet.payload[:9]
                    valid = delivery_mask(packet.capture_step, captured_episodes, episode_ids, last_delivered)
                    if not valid.any():
                        continue
                    rows = valid.nonzero(as_tuple=False).flatten()
                    extra_loss = torch.zeros((), device=device)
                    if attention:
                        student_latent, new_hidden, extra_loss = attention.update(
                            frames[rows], hidden[rows], rows, packet.payload[9],
                            (step-packet.capture_step)*dt, packet.capture_step*dt)
                        action_value = attention.metrics["action_mse"]
                        geometry_value = attention.metrics["spatial_loss"]
                        height_value = attention.metrics["height_visible_loss"]
                        visibility_value = attention.metrics["visibility_bce"]
                    else:
                        student_latent, new_hidden = student(frames[rows], hidden[rows])
                        predicted_action = teacher.actor(torch.cat((base_actor[rows], student_latent), -1))
                        action_loss = F.mse_loss(predicted_action, teacher_action[rows])
                        spatial_prediction = alignment_head(new_hidden)
                        geometry_loss, height_loss, visibility_loss, _ = spatial_loss(
                            spatial_prediction, teacher_terrain[rows], return_components=True)
                        extra_loss = action_loss + 0.5 * geometry_loss
                        action_value, geometry_value = action_loss.item(), geometry_loss.item()
                        height_value, visibility_value = height_loss.item(), visibility_loss.item()
                        rvld_latent[rows] = student_latent.detach()
                        rvld_ready[rows] = True
                        rvld_stamp[rows] = packet.capture_step * dt
                    hidden = hidden.index_copy(0, rows, new_hidden)
                    last_delivered[rows] = packet.capture_step
                    hazard_pred = student.hazard_head(new_hidden).squeeze(-1)
                    supervised = hazard_supervised[rows]
                    latent_target = teacher_latent[rows]
                    latent_loss = F.mse_loss(student_latent, latent_target)
                    hazard_loss = (F.mse_loss(hazard_pred[supervised], hazard_label[rows][supervised])
                                   if supervised.any() else hazard_pred.sum() * 0.0)
                    window_loss = window_loss + latent_loss + args_cli.hazard_loss_coef * hazard_loss + extra_loss
                    window_mse_sum += latent_loss.item()
                    window_hazard_sum += hazard_loss.item()
                    window_action_sum += action_value
                    window_geometry_sum += geometry_value
                    window_height_sum += height_value
                    window_visibility_sum += visibility_value
                    window_visible_sum += visible[rows].mean().item()
                    window_supervised_sum += supervised.float().mean().item()
                    window_delay_sum += (step - packet.capture_step) * dt * 1000
                    window_updates += 1

            it += 1
        if window_updates:
            (window_loss / window_updates).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
        hidden = hidden.detach()  # stop gradients at the BPTT window boundary
        count = max(window_updates, 1)
        if attention:
            for key, value in attention.metrics.items():
                writer.add_scalar(f"attention/{key}", value, it)
        mse_val = window_mse_sum / count
        hazard_val = window_hazard_sum / count
        action_val = window_action_sum / count
        geometry_val = window_geometry_sum / count
        height_val = window_height_sum / count
        visibility_val = window_visibility_sum / count
        if window_updates:
            writer.add_scalar("perception/mse", mse_val, it)
            writer.add_scalar("perception/hazard_mse", hazard_val, it)
            writer.add_scalar("perception/action_mse", action_val, it)
            writer.add_scalar("perception/spatial_loss", geometry_val, it)
            writer.add_scalar("perception/height_visible_loss", height_val, it)
            writer.add_scalar("perception/visibility_bce", visibility_val, it)
            writer.add_scalar("perception/visible_fraction", window_visible_sum / count, it)
            writer.add_scalar("perception/hazard_supervised_fraction", window_supervised_sum / count, it)
            if attention:
                for key, value in attention.metrics.items():
                    writer.add_scalar(f"attention/{key}", value, it)
        writer.add_scalar("transport/updates", window_updates, it)
        writer.add_scalar("transport/delay_ms", window_delay_sum / count, it)
        writer.add_scalar("transport/dropped", transport.dropped, it)
        if (it // window_len) % 10 == 0 or it >= args_cli.iterations:
            print(f"[INFO] iter {it}/{args_cli.iterations} mse={mse_val:.5f} hazard_mse={hazard_val:.5f} "
                  f"action_mse={action_val:.5f} spatial={geometry_val:.5f} "
                  f"height_visible={height_val:.5f} visibility_bce={visibility_val:.5f} "
                  f"updates={window_updates} delay_ms={window_delay_sum / count:.1f} dropped={transport.dropped} "
                  f"visible={window_visible_sum / count:.4f} "
                  f"hazard_supervised={window_supervised_sum / count:.4f}")

        if it % args_cli.save_interval < window_len or it >= args_cli.iterations:
            ckpt_path = os.path.join(log_dir, f"perception_{it}.pt")
            torch.save({
                "model": student.state_dict(), "iteration": it,
                "student_arch": args_cli.student_arch,
                "alignment_head": alignment_head.state_dict() if alignment_head is not None else None,
                "optimizer": optimizer.state_dict(),
                "student_config": {"camera_profile": env_cfg.camera_profile} if attention else {},
                "age_input": {"hidden_slot": 63, "units": "seconds_clipped_0_1"} if attention else None,
                "distillation_args": vars(args_cli),
                "camera_contract": camera_contract.manifest(),
                "student_alignment_contract": 2,
                "teacher_geometry_supervision": "teacher_visible_valid_finite_cells_only",
                "teacher_visibility_supervision": "teacher_ray_mask_all_cells",
                "teacher_behavior_target": "same_capture_step_teacher_action",
                "teacher_checkpoint": checkpoint_path,
                "teacher_sha256": teacher_sha256,
                "camera_transport": transport_config.manifest(dt),
                "camera_transport_seed": agent_cfg.seed,
            }, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print(f"[INFO] Saved {ckpt_path}")

    print(f"[INFO] Distillation complete: captures={transport.captured} dropped={transport.dropped} "
          f"delivered_packets={transport.delivered} learning_wall_seconds={time.perf_counter() - learning_started:.2f}")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
