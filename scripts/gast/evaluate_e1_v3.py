"""Distill a frozen teacher into a four-camera CNN-GRU student.

Images and teacher targets are queued together with timing jitter, delay and
packet loss. Targets use capture time; only student weights receive gradients.
"""

import argparse
import hashlib
import math
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
from gd_lab.gast.student_utils import finite_batch_rows
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
parser.add_argument("--student_arch", choices=("gast_belief_v3",), default="gast_belief_v3")
parser.add_argument("--teacher_checkpoint", help="Explicit frozen checkpoint; overrides run lookup.")
parser.add_argument("--student_resume", help="Restore student and optimizer; iterations is total target. Environment/history reset.")
# Start the student Top-5 only after enough captures: a board seeded early stays on top while later,
# better students are scored against it (same failure as the teacher board stuck at 883 on 2026-10-07).
parser.add_argument("--top5_start_iteration", type=int, default=5000)
parser.add_argument("--live_config", default=None,
                    help="JSON settings file polled during distillation (default: <run_dir>/student_live.json).")
parser.add_argument("--top5_keep", type=int, default=5)
parser.add_argument("--top5_smoothing_windows", type=int, default=8)
parser.add_argument("--top5_min_visible_sample_fraction", "--top5_min_visible_fraction",
                    dest="top5_min_visible_sample_fraction", type=float, default=0.95)
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
parser.add_argument("--v21_env", action="store_true", default=False,
                    help="Distill a v2.1 teacher in its own env: --v2_env plus the v2.1 commands (up to 1.2 m/s) "
                         "and v2.1 stair disturbance (gd_lab.mdp.gap_stair_v21.add_student_v21_env).")
parser.add_argument("--v2_env", action="store_true", default=False,
                    help="Distill a v2 teacher in its own env: 26 cm gaps and the stair hip-handle disturbance "
                         "(full force from the first step).")
parser.add_argument("--train_parts", choices=('all', 'head', 'head+gate'), default='all')
parser.add_argument("--group_balanced_latent", action='store_true')
parser.add_argument("--actor_jacobian_latent_weight", type=float, default=0.,
                    help='Reserved for UPDATE10A; nonzero is currently rejected.')
parser.add_argument("--gap_loss_weight", type=float, default=1.0,
                    help="Loss weight for samples whose 11x17 body grid contains a known gap cell (1 = off).")
parser.add_argument("--gap_ghost_probability", type=float, default=0.25,
                    help="Platform-gap depth-ghost probability; 0 disables only ghosts "
                         "while retaining generic camera noise.")
parser.add_argument("--hidden_gap_latent_weight", type=float, default=1.0,
                    help="Absolute latent row-weight floor for a known teacher-hidden "
                         "ahead corridor gap; max(existing gap weight, this value).")
parser.add_argument("--ahead_gap_latent_weight", type=float, default=1.0,
                    help="Absolute latent row-weight floor for a known ahead-corridor gap, visible or hidden; "
                         "max(existing row weight, this value).")
parser.add_argument("--gap_action_weight", type=float, default=1.0,
                    help="Absolute near-gap action row-weight floor; "
                         "max(existing gap weight, this value).")
parser.add_argument("--gap_terrain_columns", type=int, default=0,
                    help="Platform-gap terrain columns next to one column per other family (0 = task default). "
                         "E.g. 20 gives 20/30 gap columns instead of 5/15.")
parser.add_argument("--gap_top5_min_rows", type=int, default=64,
                    help="Minimum near-gap rows in a BPTT window for it to count toward the gap Top-5.")
parser.add_argument("--seen_supervision_seconds", type=float, default=0.0,
                    help="Also supervise height/gap on cells a camera saw within this many seconds (warped by body "
                         "motion), not only the currently teacher-visible cells. 0 = original supervision.")
parser.add_argument("--gap_observed_supervision", action="store_true",
                    help="Also supervise the semantic gap channel on gap cells the cameras look through at deck height "
                         "(free-space test on the privileged mesh), not only where the pit floor is teacher-visible.")
parser.add_argument("--resume_lr", type=float, default=None,
                    help="With --student_resume: set this learning rate instead of the checkpoint optimizer's.")
parser.add_argument("--e1_condition", choices=("a_student_gate","b_student_gate1","c_teacher_full","d_teacher_early","e_teacher_late","f_teacher_student_gate"), default=None, help="Frozen closed-loop E1; compact per-approach metrics instead of RolloutLog")
parser.add_argument("--replay_out", default=None,
                    help="Analysis replay (tools/rl_replay): load --student_resume frozen, let the student drive, "
                         "record --replay_seconds of every step to this JSON and exit. Nothing is trained or saved.")
parser.add_argument("--replay_seconds", type=float, default=4.0)
parser.add_argument("--replay_warmup_seconds", type=float, default=1.5, help="walk this long before recording")
parser.add_argument("--replay_vx", type=float, default=None, help="fix the forward command (m/s)")
parser.add_argument("--replay_live", default=None, help="also stream each step (NDJSON) for the live viewer")
parser.add_argument("--replay_cloud_envs", type=int, default=2)
parser.add_argument("--replay_terrain_history", action="store_true", help="opt in to full GAST history in replay")
parser.add_argument("--replay_supervision_out", default=None,
                    help="with --replay_out: also write P2 gap-supervision coverage (JSON) counted at every capture")
parser.add_argument("--replay_decoder_out", default=None,
                    help="with --replay_out: also score the decoded gap channel against clean labels (JSON)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.e1_condition != 'a_student_gate' or os.environ.get('E1_GPU_APPROVED') != '1':
    parser.error('UPDATE62 requires explicit GPU approval and a_student_gate')
if args_cli.e1_condition and (not args_cli.replay_out or args_cli.replay_warmup_seconds != 0 or args_cli.replay_vx is None or args_cli.replay_vx <= 0):
    parser.error("E1 requires --replay_out, --replay_warmup_seconds 0 and positive --replay_vx")
if args_cli.actor_jacobian_latent_weight != 0:
    parser.error('actor Jacobian weighting awaits UPDATE10A; only 0 is supported')
if args_cli.replay_out:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "rl_replay"))
    from runtime import validate_window, reserve_recording
    try:
        if not os.environ.get("TRAIN_ARM"):
            raise ValueError("replay requires explicit TRAIN_ARM")
        validate_window(args_cli.num_envs, args_cli.replay_seconds, args_cli.replay_warmup_seconds, args_cli.replay_vx)
        if not 0 <= args_cli.replay_cloud_envs <= args_cli.num_envs:
            raise ValueError("invalid replay_cloud_envs")
        if args_cli.e1_condition:
            output = Path(args_cli.replay_out).resolve()
            if not output.is_relative_to(Path(__file__).resolve().parents[2]):
                raise ValueError("E1 output must be inside the authorized runtime copy")
        replay_lock = reserve_recording(args_cli.replay_out, args_cli.replay_live,
            lock_path=Path(__file__).resolve().parents[2] / "outputs/e1.lock" if args_cli.e1_condition else None)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not args_cli.student_resume:
        parser.error("--replay_out needs --student_resume (the student to replay)")
    # Student always drives, never learns, never saves; the run stops when the recording is full.
    args_cli.student_warmup, args_cli.student_ramp = 0, 1
    args_cli.iterations = args_cli.save_interval = args_cli.top5_start_iteration = 10**9
if min(args_cli.iterations, args_cli.bptt_steps, args_cli.save_interval, args_cli.num_envs) <= 0:
    parser.error("iterations, bptt_steps, save_interval and num_envs must be positive")
if (args_cli.top5_start_iteration < 0 or args_cli.top5_keep < 1 or args_cli.top5_smoothing_windows < 1
        or not 0.0 <= args_cli.top5_min_visible_sample_fraction <= 1.0
        or not 0.0 <= args_cli.top5_min_hazard_supervised_fraction <= 1.0):
    parser.error("invalid Top-5 checkpoint criteria")
if args_cli.lr <= 0 or args_cli.hazard_loss_coef < 0:
    parser.error("lr must be positive and hazard_loss_coef nonnegative")
if not (0 <= args_cli.seen_supervision_seconds < 60):  # also rejects NaN
    parser.error("seen_supervision_seconds must be finite and in [0, 60)")
if args_cli.gap_loss_weight < 1 or args_cli.gap_terrain_columns < 0:
    parser.error("gap_loss_weight must be >= 1 and gap_terrain_columns >= 0")
for option in (
    "gap_loss_weight", "hidden_gap_latent_weight", "gap_action_weight", "ahead_gap_latent_weight"
):
    value = getattr(args_cli, option)
    if not math.isfinite(value) or value < 1:
        parser.error(f"{option} must be finite and >= 1")
if (
    not math.isfinite(args_cli.gap_ghost_probability)
    or not 0 <= args_cli.gap_ghost_probability <= 1
):
    parser.error("gap_ghost_probability must be finite and in [0,1]")
if args_cli.resume_lr is not None and (args_cli.resume_lr <= 0 or not args_cli.student_resume):
    parser.error("resume_lr must be positive and requires --student_resume")
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
import gd_lab.gast.gast_teacher_student_task
from gd_lab.gast.student import GastStudent
from gd_lab.core.camera_contract import check_checkpoint_camera_contract
from gd_lab.gast.distillation import GastDistillation
from gd_lab.gast.geometry import weighted_mean
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
    from gd_lab.core.camera_contract import require_training_camera_profile
    require_training_camera_profile(env_cfg.camera_profile)  # legacy calibration only with explicit opt-in
    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    if args_cli.replay_out and args_cli.perception_run_name:
        if Path(log_root_path, args_cli.perception_run_name).exists():
            raise ValueError("replay refuses an existing training run directory")
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

    from gd_lab.mdp.gap_stair_v2 import add_student_v2_env, set_gap_terrain_columns
    if args_cli.v21_env:
        from gd_lab.mdp.gap_stair_v21 import add_student_v21_env
        os.environ["GD_LAB_V2_FORCE_RAMP_STEPS"] = "0"  # the teacher already handles full-strength pushes
        add_student_v21_env(env_cfg)
        print("[INFO] v2.1 distillation env: 26 cm gaps, stair disturbance at full force, v2.1 commands", flush=True)
    elif args_cli.v2_env:
        os.environ["GD_LAB_V2_FORCE_RAMP_STEPS"] = "0"  # the teacher already handles full-strength pushes
        add_student_v2_env(env_cfg)
        print("[INFO] v2 distillation env: 26 cm gaps + stair hip-handle disturbance at full force", flush=True)
    if args_cli.gap_terrain_columns:
        cols = set_gap_terrain_columns(env_cfg, args_cli.gap_terrain_columns)
        print(f"[INFO] Gap-focused terrain: platform_gap {args_cli.gap_terrain_columns}/{cols} columns", flush=True)
    if args_cli.e1_condition:
        from e1_runner import configure
        configure(env_cfg)
    configure_vrl_cameras(env_cfg)
    # A capture may fall on any policy tick. Render at policy boundaries so
    # scheduled camera reads are current, including asynchronous reset rows.
    env_cfg.sim.render_interval = env_cfg.decimation
    transport = CameraTransport(transport_config, env_cfg.decimation * env_cfg.sim.dt, agent_cfg.seed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        ensure_prev_prev_action_tracking(env.unwrapped)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if args_cli.e1_condition:
            from e1_runner import load_teacher
            teacher = load_teacher(env, agent_cfg.to_dict(), checkpoint_path, agent_cfg.device)
        else:
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
        from gd_lab.gast.teacher import GastActorCritic
        if isinstance(teacher, GastActorCritic):
            # GAST teacher: full-grid memory input from the clean gast_history group (see gast_teacher_student_task).
            if "gast_history" not in env.unwrapped.observation_manager.active_terms:
                raise RuntimeError("A GAST teacher needs the GastTeacherGastStudent task (gast_history group)")
            from gd_lab.gast.gast_teacher_student_task import GastTeacherView
            teacher = GastTeacherView(teacher)
            print("[INFO] GAST teacher: latent and actions from the clean full-grid 8-step memory", flush=True)
        device = env.unwrapped.device

        dt = env.unwrapped.step_dt
        camera_contract = camera_contract_for_policy(env_cfg.camera_profile, dt)
        camera_steps = camera_contract.period_steps
        cam_period = camera_steps * dt
        print(f"[INFO] Nominal camera contract: {camera_steps} env steps ({cam_period}s), step_dt={dt}s")
        print(f"[INFO] Camera transport: {transport_config.manifest(dt)}; interval_steps={transport.interval}, delay_steps={transport.delay}")

        from e1_v3_adapter import contract_value, load_checkpoint
        from e1_v3_runner import run
        student, saved = load_checkpoint(args_cli.student_resume, device=device)
        manager = env.unwrapped.observation_manager
        if list(manager.active_terms['policy']) != saved['manifest']['policy_terms']:
            raise ValueError('policy term ordering differs from training')
        dims = contract_value(manager.group_obs_term_dim['policy'], 'policy_term_dims')
        if dims != contract_value(saved['manifest']['policy_term_dims'], 'policy_term_dims'):
            raise ValueError('policy history dimensions differ from training')
        from gd_lab.core.camera_contract import check_checkpoint_camera_contract
        check_checkpoint_camera_contract(saved['manifest']['camera_contract'], camera_contract, purpose='resume')
        run(env, teacher, student, None, transport, args_cli, agent_cfg, env_cfg,
            checkpoint_path, int(saved['updates']))
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
