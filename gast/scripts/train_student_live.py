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
from gd_lab.gast.student_top5 import StudentTop5
from gd_lab.gast.live_config import StudentLiveConfig
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

def find_teacher_env_yaml(checkpoint_path):
    checkpoint = Path(checkpoint_path).resolve()
    for parent in checkpoint.parents:
        for package_dir in ("params", "teacher_params"):
            candidate = parent / package_dir / "env.yaml"
            if candidate.is_file():
                return candidate
    return None

parser = argparse.ArgumentParser(description="Distill the vision-RL terrain-encoder teacher into a camera student.")
parser.add_argument(
    "--task", type=str, default="Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0", help="GAST student task; current supervision requires a BIVT-Ray teacher"
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point name.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--num_envs", type=int, default=256, help="Number of environments (camera rendering is expensive)."
)
parser.add_argument("--iterations", type=int, default=2000, help="Capture attempts including drops; optimize per nonempty BPTT window.")
parser.add_argument("--lr", type=float, default=1.0e-3, help="Student optimizer learning rate.")
parser.add_argument("--save_interval", type=int, default=1000, help="Capture iterations between full student checkpoints.")
parser.add_argument("--gru_hidden_dim", type=int, default=64, help="Student GRU hidden size.")
parser.add_argument(
    "--bptt_steps",
    type=int,
    default=16,
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
parser.add_argument("--student_arch", choices=("gast_spatiotemporal_v1",), default="gast_spatiotemporal_v1")
parser.add_argument("--teacher_checkpoint", help="Explicit frozen checkpoint; overrides run lookup.")
parser.add_argument("--student_resume", help="Restore student and optimizer; iterations is total target. Environment/history reset.")
parser.add_argument("--top5_start_iteration", type=int, default=5000)
parser.add_argument("--live_config", default=None,
                    help="JSON settings file polled during distillation (default: <run_dir>/student_live.json).")
parser.add_argument("--top5_keep", type=int, default=5)
parser.add_argument("--top5_smoothing_windows", type=int, default=8)
parser.add_argument("--top5_min_visible_fraction", type=float, default=0.95)
parser.add_argument("--top5_min_hazard_supervised_fraction", type=float, default=0.95)
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
if (args_cli.top5_start_iteration < 0 or args_cli.top5_keep < 1 or args_cli.top5_smoothing_windows < 1
        or not 0.0 <= args_cli.top5_min_visible_fraction <= 1.0
        or not 0.0 <= args_cli.top5_min_hazard_supervised_fraction <= 1.0):
    parser.error("invalid Top-5 checkpoint criteria")
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
import gd_lab.gast.tasks
import gd_lab.gast.cvtt_student_task
import gd_lab.gast.bivt_student_task
from gd_lab.gast.student import GastStudent
from gd_lab.gast.distillation import GastDistillation
if args_cli.task == 'Gd-VrlRayStudent-Rbq10-Dreamwaq-Vision-v0':
    import gd_lab.teachers.bivt.student_task  # noqa: F401
from gd_lab.core.camera_contract import camera_contract_for_policy
from gd_lab.core.paths import LOG_ROOT
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.mdp.camera_noise import augment_student_camera_frames
from gd_lab.mdp.platform_gap_noise import PlatformGapDepthGhost, PlatformGapDepthGhostCfg
from gd_lab.mdp.terrain_families import terrain_family_gate
from gd_lab.teachers.cvtt.actor_critic import DreamwaqVrlActorCritic
from gd_lab.students.gavd.distillation import AttentionDistillation
from gd_lab.students.rvld.model import CameraPerceptionEncoder, height_discontinuity_metres
from gd_lab.students.gavd.model import GridAttentionStudent
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
from gd_lab.teachers.cvtt.student_env import CameraNoiseCfg


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


@torch.no_grad()
def capture_teacher_packet(env, obs, teacher, episode_ids, noise_cfg, gap_ghost):
    """Freeze images AND supervision at capture time, before simulated transit."""
    snapshot = getattr(env.unwrapped, "_vrl_camera_snapshot", None)
    stamps = getattr(env.unwrapped, "_vrl_camera_snapshot_steps", None)
    step = env.unwrapped.common_step_counter
    if snapshot is None or stamps is None or not (stamps == step).all():
        raise RuntimeError("Scheduled camera capture did not produce fresh images for every environment")
    ray_steps = getattr(env.unwrapped, "_vrl_teacher_ray_capture_steps", None)
    ray_snapshot = getattr(env.unwrapped, "_vrl_teacher_ray_snapshot", None)
    if ray_steps is None or not (ray_steps == step).all():
        raise RuntimeError("Teacher Ray map is not from the current student camera capture step")
    gap_envs = terrain_family_gate(env.unwrapped, ("platform_gap",)) == 0
    frames = augment_student_camera_frames(
        snapshot[0].clone(), snapshot, env.unwrapped.scene.env_origins, gap_envs, noise_cfg, gap_ghost
    )
    latent = teacher.terrain_latent(obs).clone()
    terrain = obs["terrain"]
    if terrain.shape[-1] != 374:
        raise RuntimeError(f"BIVT-Ray teacher terrain must be [masked height(187), visibility(187)], got {terrain.shape[-1]}")
    if ray_snapshot is None or not torch.allclose(terrain, ray_snapshot, atol=0, rtol=0):
        raise RuntimeError("Teacher behavior/latent input is not the synchronized Ray target")
    teacher_action = teacher.act_inference(obs).clone()
    cells = teacher._height_scan_slice.stop - teacher._height_scan_slice.start
    visibility = terrain[..., cells:].bool()
    hazard = height_discontinuity_metres(terrain[..., :cells], 5.0, teacher.terrain_encoder.grid_shape, visibility)
    grid = visibility.reshape(-1, *teacher.terrain_encoder.grid_shape)
    supervised = ((grid[:, 1:, :] & grid[:, :-1, :]).flatten(1).any(1)
                  | (grid[:, :, 1:] & grid[:, :, :-1]).flatten(1).any(1))
    return (frames.clone(), latent, hazard, supervised, visibility.float().mean(-1),
            episode_ids.clone(), teacher_action)


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
    teacher_env_yaml = find_teacher_env_yaml(checkpoint_path)
    if teacher_env_yaml is None:
        raise RuntimeError("Teacher package must include params/env.yaml so its camera profile can be verified")
    teacher_cfg_text = teacher_env_yaml.read_text()
    teacher_profiles = re.findall(r"(?m)^camera_profile:\s*([A-Za-z0-9_-]+)\s*$", teacher_cfg_text)
    if len(teacher_profiles) != 1 or teacher_profiles[0] != env_cfg.camera_profile:
        raise RuntimeError(f"Teacher/student camera profile mismatch or missing metadata: teacher={teacher_profiles}, student={env_cfg.camera_profile}")
    print(f"[INFO] Verified teacher/student camera profile={teacher_profiles[0]} from {teacher_env_yaml}", flush=True)
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
    print(f'[INFO] Frozen teacher class={type(teacher).__name__} sha256={teacher_sha256}', flush=True)
    device = env.unwrapped.device

    dt = env.unwrapped.step_dt
    camera_contract = camera_contract_for_policy(env_cfg.camera_profile, dt)
    camera_steps = camera_contract.period_steps
    cam_period = camera_steps * dt
    print(f"[INFO] Nominal camera contract: {camera_steps} env steps ({cam_period}s), step_dt={dt}s")
    print(f"[INFO] Camera transport: {transport_config.manifest(dt)}; interval_steps={transport.interval}, delay_steps={transport.delay}")

    student = GastStudent(env_cfg.camera_profile).to(device)
    attention = GastDistillation(teacher, student, env.unwrapped, camera_contract,
                                args_cli.student_warmup, args_cli.student_ramp)
    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)
    start_iteration = 0
    if args_cli.student_resume:
        saved = torch.load(args_cli.student_resume, map_location=device, weights_only=False)
        if saved['teacher_sha256'] != teacher_sha256 or saved['student_arch'] != args_cli.student_arch:
            raise ValueError('Resume teacher hash or student architecture differs')
        if saved.get('gast_distillation_contract') != 2:
            raise ValueError('Student checkpoint uses a different GAST supervision contract; do not silently mix objectives')
        if saved.get('camera_contract') != camera_contract.manifest():
            raise ValueError('Resume camera calibration/timing contract differs from this run')
        student.load_state_dict(saved['model'], strict=True)
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
    top5_manager = StudentTop5(os.path.join(log_dir, "top5"),
        start_iteration=args_cli.top5_start_iteration, keep=args_cli.top5_keep,
        smoothing_windows=args_cli.top5_smoothing_windows,
        min_visible_fraction=args_cli.top5_min_visible_fraction,
        min_hazard_supervised_fraction=args_cli.top5_min_hazard_supervised_fraction)
    writer = SummaryWriter(log_dir=log_dir)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    live_config_path = args_cli.live_config or os.path.join(log_dir, "student_live.json")
    live_config = StudentLiveConfig(live_config_path, {
        "learning_rate": args_cli.lr,
        "student_warmup": args_cli.student_warmup,
        "student_ramp": args_cli.student_ramp,
        "checkpoint_interval": args_cli.save_interval,
        "bptt_steps": args_cli.bptt_steps,
    })
    print(f"[INFO] Perception log dir: {log_dir}")
    print(f"[INFO] Live settings file: {live_config_path}", flush=True)

    obs = env.get_observations()
    hidden = student.init_hidden(env.unwrapped.num_envs, device)
    episode_ids = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
    last_delivered = torch.full_like(episode_ids, -1)
    learning_started = time.perf_counter()

    def poll_live_settings(iteration):
        settings, changed, warning = live_config.poll()
        if warning:
            print(f"[WARN] {warning}", flush=True)
        if not changed:
            return
        if "learning_rate" in changed:
            args_cli.lr = float(settings["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = args_cli.lr
        if "student_warmup" in changed:
            args_cli.student_warmup = int(settings["student_warmup"])
            attention.warmup = args_cli.student_warmup
        if "student_ramp" in changed:
            args_cli.student_ramp = int(settings["student_ramp"])
            attention.ramp = args_cli.student_ramp
        if "checkpoint_interval" in changed:
            args_cli.save_interval = int(settings["checkpoint_interval"])
        if "bptt_steps" in changed:
            args_cli.bptt_steps = int(settings["bptt_steps"])
        print(f"[LIVE_CONFIG] iteration={iteration} applied={changed} "
              f"(BPTT length takes effect at the next window)", flush=True)


    def make_student_checkpoint(iteration, top5_selection=None):
        payload = {"model": student.state_dict(), "iteration": iteration,
            "student_arch": args_cli.student_arch, "optimizer": optimizer.state_dict(),
            "student_config": {"camera_profile": env_cfg.camera_profile} if attention else {},
            "age_input": {"explicit_seconds": True, "stale_seconds": .3},
            "gast_contract": {"external_gap_signal": False, "pose": "xy_yaw_wxyz", "hidden_dim": student.gru_hidden_dim},
            "distillation_args": vars(args_cli), "camera_contract": camera_contract.manifest(),
            "camera_noise_enabled": camera_noise_cfg is not None,
            "teacher_checkpoint": checkpoint_path, "teacher_sha256": teacher_sha256,
            "camera_transport": transport_config.manifest(dt), "camera_transport_seed": agent_cfg.seed,
            "gast_distillation_contract": 2,
            "student_geometry_supervision": "teacher_ray_visible_valid_cells_only",
            "student_visibility_supervision": "teacher_ray_mask_all_cells",
            "teacher_behavior_target": "same_capture_step_teacher_action"}
        if top5_selection is not None: payload["student_top5_selection"] = top5_selection
        return payload

    def write_student_checkpoint(path, iteration, top5_selection=None):
        temporary_path = path + ".tmp"
        torch.save(make_student_checkpoint(iteration, top5_selection), temporary_path)
        os.replace(temporary_path, path)

    it = start_iteration
    last_saved_iteration = start_iteration
    consecutive_skips = skipped_windows = nonfinite_rows = 0
    while it < args_cli.iterations:
        poll_live_settings(it)
        window_len = min(args_cli.bptt_steps, args_cli.iterations - it)
        optimizer.zero_grad()
        window_loss = torch.zeros((), device=device)
        window_mse_sum, window_hazard_sum = 0.0, 0.0
        window_extra_sum, window_total_loss_sum = 0.0, 0.0
        window_visible_sum, window_supervised_sum = 0.0, 0.0
        window_updates = 0
        window_delay_sum = 0.0

        # Keep hidden states in-graph across delivered frames in this BPTT window.
        for _ in range(window_len):
            poll_live_settings(it)
            capture_step = transport.next_capture(env.unwrapped.common_step_counter)
            env.unwrapped._vrl_camera_capture_step = capture_step
            # Drain the final capture's transit time before saving the last student.
            stop_step = capture_step + (transport.delay[1] if it + 1 == args_cli.iterations else 0)
            while env.unwrapped.common_step_counter < stop_step:
                with torch.no_grad():
                    actions = attention.actions(obs, it) if attention else teacher.act_inference(obs)
                obs, _, dones, _ = env.step(actions)
                if dones.any():
                    done_rows = dones.reshape(-1).bool()
                    keep = (~done_rows).unsqueeze(-1).to(hidden.dtype)
                    # where(), not multiply (NaN*0 is NaN); stays in-graph on purpose.
                    hidden = torch.where(keep > 0, hidden, torch.zeros_like(hidden))
                    episode_ids += done_rows.long()
                    gap_ghost.reset(done_rows.nonzero(as_tuple=False).flatten())
                    if attention:
                        attention.reset(done_rows)

                step = env.unwrapped.common_step_counter
                if step == capture_step:
                    payload = capture_teacher_packet(env, obs, teacher, episode_ids, camera_noise_cfg, gap_ghost)
                    if attention:
                        payload = (*payload, attention.capture(obs, payload[6]))
                    transport.capture(step, payload)
                for packet in transport.receive(step):
                    frames, teacher_latent, hazard_label, hazard_supervised, visible, captured_episodes, teacher_action = packet.payload[:7]
                    valid = delivery_mask(packet.capture_step, captured_episodes, episode_ids, last_delivered)
                    # One exploded env must not kill the run: drop its non-finite rows.
                    finite = (torch.isfinite(frames.flatten(1)).all(1) & torch.isfinite(teacher_latent).all(1)
                              & torch.isfinite(teacher_action).all(1) & torch.isfinite(hazard_label.flatten(1)).all(1)
                              & torch.isfinite(visible.flatten(1)).all(1))
                    if attention:
                        for value in packet.payload[7]:
                            if value.is_floating_point():
                                finite &= torch.isfinite(value.flatten(1)).all(1)
                    bad_rows = valid & ~finite
                    if bad_rows.any():
                        # Same rule as main-tree distill_student.py: a non-finite packet
                        # means that env's state is unusable; drop its memory now.
                        nonfinite_rows += int(bad_rows.sum())
                        hidden = torch.where(bad_rows[:, None], torch.zeros_like(hidden), hidden)
                        if attention:
                            attention.reset(bad_rows)
                        print(f"[GAST_STUDENT_NONFINITE] capture_step={packet.capture_step} "
                              f"envs={(~finite).nonzero().flatten()[:16].tolist()}", flush=True)
                    valid &= finite
                    if not valid.any():
                        continue
                    rows = valid.nonzero(as_tuple=False).flatten()
                    extra_loss = torch.zeros((), device=device)
                    if attention:
                        student_latent, new_hidden, extra_loss = attention.update(
                            frames[rows], hidden[rows], rows, packet.payload[7], (step-packet.capture_step)*dt)
                    else:
                        student_latent, new_hidden = student(frames[rows], hidden[rows])
                    hidden = hidden.index_copy(0, rows, new_hidden)
                    last_delivered[rows] = packet.capture_step
                    hazard_pred = student.hazard_head(student.hazard_input(new_hidden)).squeeze(-1)
                    supervised = hazard_supervised[rows]
                    latent_loss = F.mse_loss(student_latent, teacher_latent[rows]*attention.target_gate)
                    hazard_loss = (F.mse_loss(hazard_pred[supervised], hazard_label[rows][supervised])
                                   if supervised.any() else hazard_pred.sum() * 0.0)
                    window_loss = window_loss + latent_loss + args_cli.hazard_loss_coef * hazard_loss + extra_loss
                    latent_value, hazard_value, extra_value = latent_loss.item(), hazard_loss.item(), extra_loss.item()
                    window_mse_sum += latent_value
                    window_hazard_sum += hazard_value
                    window_extra_sum += extra_value
                    window_total_loss_sum += latent_value + args_cli.hazard_loss_coef * hazard_value + extra_value
                    window_visible_sum += visible[rows].mean().item()
                    window_supervised_sum += supervised.float().mean().item()
                    window_delay_sum += (step - packet.capture_step) * dt * 1000
                    window_updates += 1

            it += 1
        if window_updates:
            (window_loss / window_updates).backward()
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if torch.isfinite(norm):
                optimizer.step()
                consecutive_skips = 0
            else:
                optimizer.zero_grad(set_to_none=True)
                consecutive_skips += 1
                skipped_windows += 1
                bad_hidden = ~torch.isfinite(hidden).all(-1)
                hidden = torch.where(bad_hidden[:, None], torch.zeros_like(hidden), hidden)
                print(f"[GAST_STUDENT_NONFINITE] window skipped at iteration {it}, "
                      f"consecutive={consecutive_skips}, reset_hidden={int(bad_hidden.sum())}", flush=True)
                if consecutive_skips > 5:
                    raise RuntimeError("Student gradients non-finite for 6 consecutive windows")
        hidden = hidden.detach()  # stop gradients at the BPTT window boundary
        count = max(window_updates, 1)
        if attention:
            for key, value in attention.metrics.items():
                writer.add_scalar(f"attention/{key}", value, it)
        mse_val = window_mse_sum / count
        hazard_val = window_hazard_sum / count
        extra_val = window_extra_sum / count
        total_loss_val = window_total_loss_sum / count
        visible_val = window_visible_sum / count
        supervised_val = window_supervised_sum / count
        top5_result = None
        if window_updates:
            top5_result = top5_manager.consider(it, total_loss_val,
                {"latent_mse": mse_val, "hazard_mse": hazard_val, "extra_loss": extra_val,
                 "visible_fraction": visible_val, "hazard_supervised_fraction": supervised_val,
                 "updates": window_updates},
                save_fn=lambda path, record: write_student_checkpoint(str(path), it, record))
        if window_updates:
            writer.add_scalar("perception/mse", mse_val, it)
            writer.add_scalar("perception/hazard_mse", hazard_val, it)
            writer.add_scalar("perception/extra_loss", extra_val, it)
            writer.add_scalar("perception/total_loss", total_loss_val, it)
            writer.add_scalar("perception/skipped_windows", skipped_windows, it)
            writer.add_scalar("perception/nonfinite_rows", nonfinite_rows, it)
            writer.add_scalar("perception/visible_fraction", visible_val, it)
            writer.add_scalar("perception/hazard_supervised_fraction", supervised_val, it)
        writer.add_scalar("transport/updates", window_updates, it)
        writer.add_scalar("transport/delay_ms", window_delay_sum / count, it)
        writer.add_scalar("transport/dropped", transport.dropped, it)
        if (it // window_len) % 2 == 0 or it >= args_cli.iterations:
            print(f"[INFO] iter {it}/{args_cli.iterations} mse={mse_val:.5f} hazard_mse={hazard_val:.5f} "
                  f"total_loss={total_loss_val:.5f} updates={window_updates} "
                  f"delay_ms={window_delay_sum / count:.1f} dropped={transport.dropped} "
                  f"visible={visible_val:.4f} hazard_supervised={supervised_val:.4f} "
                  f"learning_wall_seconds={time.perf_counter()-learning_started:.2f}")
        if top5_result and top5_result.get("evaluated"):
            print(f"[TOP5] iter={it} score={top5_result.get('score', float('nan')):.6f} "
                  f"windows={top5_result.get('windows', 0)} rank={top5_result.get('rank', '-')} "
                  f"saved={top5_result.get('saved', False)} reason={top5_result.get('reason', '-')}")

        if it - last_saved_iteration >= args_cli.save_interval or it >= args_cli.iterations:
            ckpt_path = os.path.join(log_dir, f"perception_{it}.pt")
            write_student_checkpoint(ckpt_path, it)
            last_saved_iteration = it
            print(f"[INFO] Saved {ckpt_path}")

    print(f"[INFO] Distillation complete: captures={transport.captured} dropped={transport.dropped} "
          f"delivered_packets={transport.delivered} learning_wall_seconds={time.perf_counter() - learning_started:.2f}")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
