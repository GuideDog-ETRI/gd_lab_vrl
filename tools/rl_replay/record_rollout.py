"""Record a policy rollout for offline reward analysis (tools/rl_replay/viewer.html plays it back).

Per control step (100 Hz) and env: base pose and velocity, every body position (for drawing the robot),
joint positions/velocities, the command, the policy's mean action and std, the critic value V(s), every
reward term (weighted, per step, as the RewardManager adds it), foot contacts, the 11x17 height-scan points
and the terrain latent. After the rollout it adds, per env, the discounted return-to-go G_t, its split by
reward term (G_t^k = sum_i gamma^i r^k_{t+i}), the TD error delta_t and the GAE advantage A_t -- the numbers
that describe this recorded trajectory relative to the teacher critic, not an actual PPO update.

  TRAIN_ARM=4 PYTHONPATH=gast/src:src python tools/rl_replay/record_rollout.py --headless \
      --task Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0 --checkpoint <model.pt> --num_envs 8 --seconds 4 \
      --out logs/rl_replay/gast2000.json [--vx 0.8] [--stochastic] [--student_view] [--live <file.ndjson>]

--student_view renders the four belly cameras and stores what they see as world points (the student's view),
next to the ground-truth scan and the policy's own terrain input. --live also streams every step as one JSON
line while the simulator runs, so the viewer can follow it live (G_t and A_t there are provisional).

The JSON is self-contained (meta + per-env arrays, rounded to 4 decimals).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from runtime import fix_velocity, reserve_recording, validate_window, environment_metadata, task_source
bootstrap_parser = argparse.ArgumentParser(add_help=False)
bootstrap_parser.add_argument("--task", required=True)
bootstrap_args, _ = bootstrap_parser.parse_known_args()
sys.path.insert(0, str(task_source(ROOT, bootstrap_args.task)))
from gd_lab.core.experiments import training_arm_overrides

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="the task the policy was trained on")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seconds", type=float, default=4.0, help="recorded window (3-4 s shows a gap or a stair run)")
parser.add_argument("--warmup_seconds", type=float, default=1.0, help="walk this long before recording")
parser.add_argument("--vx", type=float, default=None, help="fix the forward command (m/s); default: the task's commands")
parser.add_argument("--stochastic", action="store_true", help="sample actions (training behaviour) instead of the mean")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", required=True)
parser.add_argument("--student_view", action="store_true", help="render the belly cameras and record their depth as points")
parser.add_argument("--cloud_envs", type=int, default=2, help="envs that keep camera points (they are large)")
parser.add_argument("--cloud_stride", type=int, default=4, help="keep every n-th depth pixel per axis")
parser.add_argument("--terrain_history", action="store_true", help="opt in to full 8-frame GAST history (large)")
parser.add_argument("--live", default=None, help="also stream each step as an NDJSON line to this file")
parser.add_argument("--terrain_input", choices=("train", "lidar_v25or"), default="train",
                    help="train: the task's own terrain observation; lidar_v25or: GAST teachers see the deployed LiDAR "
                         "map instead (front LiDAR 10 Hz -> 100 ms / 0-150 ms / 5%% drop transport -> L0S + DUAL2 + "
                         "DDEEP.7, as E1 v25or), IR channel = 0 for Ray+IR teachers. Rewards are unchanged.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.student_view or "Raycast" in args_cli.task or "Vision" in args_cli.task:
    args_cli.enable_cameras = True  # BIVT-Ray's camera-visible terrain needs the cameras too
try:
    if not os.environ.get("TRAIN_ARM"):
        raise ValueError("TRAIN_ARM must be explicitly set")
    validate_window(args_cli.num_envs, args_cli.seconds, args_cli.warmup_seconds, args_cli.vx)
    if args_cli.terrain_input != "train" and not args_cli.task.startswith("Gd-Gast"):
        raise ValueError("--terrain_input lidar_v25or needs a GAST teacher task")
    if args_cli.cloud_stride < 1 or not 0 <= args_cli.cloud_envs <= args_cli.num_envs:
        raise ValueError("invalid cloud_envs/cloud_stride")
    arm_overrides = training_arm_overrides(os.environ["TRAIN_ARM"], play=False)
    replay_lock = reserve_recording(args_cli.out, args_cli.live)
except (ValueError, OSError) as exc:
    parser.error(str(exc))
sys.argv = [sys.argv[0]] + arm_overrides + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import gd_lab  # noqa: F401  (registers the tasks)
import gd_lab.teachers.bivt  # noqa: F401
if args_cli.task.startswith("Gd-Gast"):
    import gd_lab.gast.tasks  # noqa: F401
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_log import RolloutLog  # noqa: E402


class LidarV25orInput:
    """E1 v25or terrain input for a GAST teacher: the policy's terrain obs and 8-frame history come from the LiDAR map."""

    REQUIRED_ENV = {"U130_DUAL": "2", "U130_DDEEP": "0.7", "U127_SENSOR_COUNT": "1"}

    def __init__(self, base, policy, seed):
        # lidar126 / structure130 read these at import: fix them first and refuse a conflicting value.
        for key, value in self.REQUIRED_ENV.items():
            if os.environ.get(key, value) != value:
                raise ValueError(f"--terrain_input lidar_v25or needs {key}={value}, got {os.environ[key]}")
            os.environ[key] = value
        if "structure130" in sys.modules or "lidar126" in sys.modules:
            raise RuntimeError("lidar126/structure130 imported before the v25or settings were fixed")
        import lidar126, structure130
        from lidar126 import Lidar, Maps
        assert structure130.S["dual_or"] and abs(structure130.S["d_deep"] - .7) < 1e-9 and lidar126.SENSOR_COUNT == 1
        from gd_lab.core.camera_transport import CameraTransport, CameraTransportConfig, delivery_mask
        from gd_lab.gast.observations import TerrainHistory
        self.base, self.n, self.device, self.delivery_mask = base, base.num_envs, base.device, delivery_mask
        channels = getattr(policy.terrain_encoder, "channels", None)
        if channels not in (2, 3):
            raise ValueError(f"unknown GAST terrain encoder channels: {channels}")
        self.rayir = channels == 3
        if self.rayir:
            from rayir_e1 import RayIrHistory
            self.history = RayIrHistory(base)
        else:
            self.history = TerrainHistory(base)
        self.sensor, self.maps = Lidar(base), Maps(self.n, "L0S")
        config = CameraTransportConfig((100.0, 100.0), (0.0, 150.0), 0.05)
        self.transport = CameraTransport(config, base.step_dt, seed)
        self.episodes = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.last = torch.full_like(self.episodes, -1)
        self.next_capture = self.transport.next_capture(base.common_step_counter)
        base._vrl_camera_capture_step = self.next_capture
        here = Path(__file__).resolve().parent
        digest = lambda p: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
        self.manifest = {
            "mapper": "L0S (h0d 5 cm world map + structure130 GAP/pending overlay)", "sensor_count": lidar126.SENSOR_COUNT,
            "structure130": {k: structure130.S[k] for k in ("dual_or", "d_deep", "step_max", "gap_depth", "gap_max_age")},
            "transport": config.manifest(base.step_dt), "transport_seed": seed, "lidar_noise_seed": 42,
            "ir_channel": "zero" if self.rayir else None, "terrain_channels": channels,
            "latent_gate": "map: terrain obs replaced by the current LiDAR map, so a map with no valid cell gives latent 0 "
                           "(E1 v25or kept the task terrain obs for the gate)",
            "source_sha256": {f: digest(here / f) for f in ("lidar126.py", "structure130.py", "heightmap116.py",
                                                            "heightmap119z.py", "rayir_e1.py")}}

    def view(self, obs):
        from gd_lab.gast.observations import pose_xyyaw
        dt = self.base.step_dt
        poses = pose_xyyaw(self.base).cpu().numpy()
        scanner_z = self.base.scene["height_scanner"].data.pos_w[:, 2].cpu().numpy()
        terrain = torch.as_tensor(self.maps.sample(poses, scanner_z, int(self.base.common_step_counter) * dt),
                                  device=self.device, dtype=torch.float32)
        frames = torch.cat((terrain, torch.zeros_like(terrain[:, :187])), -1) if self.rayir else terrain
        view = obs.copy()
        view["gast_history"] = self.history.update(self.base, frames)
        if "terrain" not in view.keys() or view["terrain"].shape[-1] != 374:
            raise ValueError("GAST teacher obs has no 374-wide 'terrain' group to replace")
        view["terrain"] = terrain
        return view

    def after_step(self, done):
        dt = self.base.step_dt
        reset = done.reshape(-1).bool()
        self.episodes += reset.long()
        ids = reset.nonzero().flatten()
        self.history.reset(ids)
        self.maps.reset(ids.tolist())
        step = int(self.base.common_step_counter)
        if step == self.next_capture:
            packet = self.sensor.capture(self.episodes.cpu().numpy(), step)
            self.transport.capture(step, (packet, self.episodes.clone()))
            self.next_capture = self.transport.next_capture(step)
            self.base._vrl_camera_capture_step = self.next_capture
        for delivery in self.transport.receive(step):
            packet, captured = delivery.payload
            mask = self.delivery_mask(delivery.capture_step, captured, self.episodes, self.last)
            ids = mask.nonzero().flatten()
            if len(ids):
                self.maps.integrate(ids.tolist(), packet, delivery.capture_step * dt, step * dt)
                self.last[ids] = delivery.capture_step


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


def action_std(policy, mean):
    """The Gaussian policy's std without sampling (rsl_rl keeps either ``std`` or ``log_std``)."""
    if getattr(policy, "std", None) is not None:
        return policy.std.detach().expand_as(mean)
    if getattr(policy, "log_std", None) is not None:
        return policy.log_std.detach().exp().expand_as(mean)
    return torch.zeros_like(mean)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.enable_cameras:
        from gd_lab.core.camera_contract import DEFAULT_CAMERA_PROFILE
        from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
        if not hasattr(env_cfg, "camera_profile"):
            env_cfg.camera_profile = DEFAULT_CAMERA_PROFILE
        configure_vrl_cameras(env_cfg)
        env_cfg.sim.render_interval = env_cfg.decimation
    env = gym.make(args_cli.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    fix_velocity(base.command_manager.get_term("base_velocity"), args_cli.vx)
    runner = _resolve(agent_cfg.class_name)(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.alg.policy
    policy.eval()
    dt, n = base.step_dt, base.num_envs

    obs, _ = env.reset()
    lidar = LidarV25orInput(base, policy, args_cli.seed) if args_cli.terrain_input == "lidar_v25or" else None
    see = (lambda o: lidar.view(o)) if lidar is not None else (lambda o: o)
    with torch.inference_mode():
        for _ in range(int(round(args_cli.warmup_seconds / dt))):
            obs, _, done, _ = env.step(policy.act_inference(see(obs)))
            if lidar is not None:
                lidar.after_step(done)

    log = RolloutLog(base, task=args_cli.task, checkpoint=args_cli.checkpoint,
                     gamma=float(agent_cfg.algorithm.gamma), lam=float(agent_cfg.algorithm.lam),
                     steps=int(round(args_cli.seconds / dt)), stochastic=args_cli.stochastic, vx=args_cli.vx,
                     live=args_cli.live, student_view=args_cli.student_view and args_cli.enable_cameras,
                     camera_profile=getattr(env_cfg, "camera_profile", None),
                     cloud_envs=args_cli.cloud_envs, cloud_stride=args_cli.cloud_stride,
                     extra_meta={"driver": "teacher", "terrain_input": args_cli.terrain_input,
                                 "lidar_v25or": lidar.manifest if lidar is not None else None,
                                 **environment_metadata(base)},
                     terrain_source="gast_history" if args_cli.task.startswith("Gd-Gast") else "terrain",
                     capture_current_camera=args_cli.student_view, terrain_history=args_cli.terrain_history)
    with torch.inference_mode():
        while not log.full():
            seen = see(obs)  # what the actor's terrain encoder gets (the LiDAR map with --terrain_input lidar_v25or)
            mean = policy.act_inference(seen)
            action = policy.act(seen) if args_cli.stochastic else mean
            latent = policy.terrain_latent(seen) if hasattr(policy, "terrain_latent") else torch.zeros(n, 0, device=mean.device)
            log.before_step(seen, mean=mean, std=action_std(policy, mean), action=action,
                            value=policy.evaluate(obs).squeeze(-1), latent=latent,
                            command=base.command_manager.get_command("base_velocity"))
            obs, reward, done, _ = env.step(action)
            log.after_step(reward, done)
            if lidar is not None:
                lidar.after_step(done)
        log.finish(policy.evaluate(obs).squeeze(-1), args_cli.out)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
