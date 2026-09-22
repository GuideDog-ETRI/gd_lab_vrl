"""Ten policy steps: force off-phase resets and check real camera capture timing.

No PPO or student optimizer is run. Uses the same calibrated camera pipeline
as training and checks render/pose timestamps rather than only tensor shapes.
"""

import argparse

from isaaclab.app import AppLauncher

from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
runtime_root = prepare_vrl_runtime(args)
app = AppLauncher(args).app
verify_vrl_runtime(runtime_root)

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import gd_lab  # noqa: F401
import gd_lab.mdp.camera_observations as camera_obs
from gd_lab.core.camera_contract import CAMERA_NAMES
from gd_lab.core.camera_geometry import mounted_camera_world_poses
from gd_lab.core.camera_timing import camera_refresh_mask
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras


def main():
    task = "Gd-Vrl-Rbq10-Dreamwaq-VisionPlay-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=8)
    cfg.seed = 42
    configure_vrl_cameras(cfg)
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    ensure_prev_prev_action_tracking(base)
    original_render = base.sim.render
    original_snapshot = camera_obs.canonical_camera_snapshot
    try:
        env.reset()
        last_render = base._sim_step_counter
        captures = []
        last_debug = None

        def track_render(*a, **kw):
            nonlocal last_render
            result = original_render(*a, **kw)
            last_render = base._sim_step_counter
            return result

        def checked_snapshot(scene, contract):
            nonlocal last_debug
            assert last_render == base._sim_step_counter, "snapshot read between renders"
            snapshot = original_snapshot(scene, contract)
            for name in CAMERA_NAMES:
                sensor = scene[name]
                assert torch.allclose(sensor._timestamp, sensor._timestamp_last_update), "stale camera pose"
            assert torch.isfinite(snapshot[0]).all()
            robot = scene["robot"]
            trunk = robot.body_names.index("trunk")
            pos, quat = mounted_camera_world_poses(robot.data.body_pos_w[:, trunk],
                                                 robot.data.body_quat_w[:, trunk], contract)
            assert torch.allclose(snapshot[2], pos) and torch.allclose(snapshot[3], quat)
            last_debug = {
                "snapshot": tuple(x.cpu().clone() for x in snapshot),
                "points": scene["height_scanner"].data.ray_hits_w.cpu().clone(),
                "scan_pos": scene["height_scanner"].data.pos_w.cpu().clone(),
                "robot_pos": robot.data.body_pos_w[:, trunk].cpu().clone(),
                "robot_quat": robot.data.body_quat_w[:, trunk].cpu().clone(),
                "origins": scene.env_origins.cpu().clone(),
            }
            captures.append(base.common_step_counter)
            return snapshot

        base.sim.render = track_render
        camera_obs.canonical_camera_snapshot = checked_snapshot
        expected = base._vrl_camera_snapshot_steps.clone()
        start = base.common_step_counter
        for index in range(1, 11):
            # Exercise all three non-render phases and distinct environment rows.
            forced = {1: 0, 6: 1, 7: 2}.get(index)
            if forced is not None:
                base.episode_length_buf[forced] = base.max_episode_length - 1
            obs, _, terminated, truncated, _ = env.step(torch.zeros(8, 12, device=base.device))
            done = terminated | truncated
            if forced is not None:
                assert done[forced], "requested reset did not occur"
            expected[done] = -1
            step = base.common_step_counter
            expected[camera_refresh_mask(expected, step, 4)] = step
            assert torch.equal(base._vrl_camera_snapshot_steps, expected)
            assert torch.isfinite(obs["terrain"]).all()
            print(f"[SYNC] step={step - start} reset={done.nonzero().flatten().tolist()} "
                  f"snapshot_steps={expected.tolist()} "
                  f"visible_fraction={obs['terrain'][:, 187:].mean().item():.4f}")
        assert all(start + step in captures for step in (1, 4, 6, 7, 8))
        print("[PASS] Live camera render/pose/target synchronization across partial resets (10 steps)")
        torch.save(last_debug, runtime_root / "sensor_snapshot.pt")
        assert obs["terrain"][:, 187:].any(), "all terrain labels are hidden in the smoke scene"
    finally:
        base.sim.render = original_render
        camera_obs.canonical_camera_snapshot = original_snapshot
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
