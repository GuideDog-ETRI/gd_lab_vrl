"""Measure how well rendering-free raycast visibility matches the rendered mask.

Runs the camera-rendering VRL teacher task with a trained VRL policy walking,
and at every camera capture compares, per env and scan cell, the rendered
CameraVisibleTerrain mask with raycast_visible_mask computed from the exact
capture poses. Terrain levels are spread uniformly so all difficulties appear.

  TRAIN_ARM=4 python scripts/compare_visibility.py --headless --num_envs 64 \
      --load_run <run> --checkpoint model_999.pt --steps 2000 --out logs/visibility_compare.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from gd_lab.core.experiments import training_arm_overrides
from vrl_runtime import prepare_vrl_runtime, verify_vrl_runtime

parser = argparse.ArgumentParser(description="Compare raycast and rendered camera visibility.")
parser.add_argument("--task", type=str, default="Gd-Vrl-Rbq10-Dreamwaq-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000, help="Policy steps to simulate.")
parser.add_argument("--out", type=str, required=True, help="JSON file for the summary.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
arm_overrides = [item.replace("agent.experiment_name=blind_", "agent.experiment_name=vision_")
                 for item in training_arm_overrides(os.environ.get("TRAIN_ARM"))]
sys.argv = [sys.argv[0]] + arm_overrides + hydra_args

runtime_root = prepare_vrl_runtime(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
verify_vrl_runtime(runtime_root)

import importlib

import gd_lab  # noqa: F401  (registers the tasks)
import gd_lab.teachers.bivt  # noqa: F401
import gymnasium as gym
import torch
import time
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.paths import LOG_ROOT
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking
from gd_lab.tasks.vrl_cameras import configure_vrl_cameras
from gd_lab.teachers.bivt.raycast_terrain import raycast_visible_mask
from gd_lab.teachers.bivt.raycast_visibility import contract_intrinsic, warp_mesh_cast
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _resolve(path: str):
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    log_root = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    configure_vrl_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    ensure_prev_prev_action_tracking(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = _resolve(agent_cfg.class_name)(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    terrain = base.scene.terrain
    rows = terrain.terrain_origins.shape[0]
    terrain.terrain_levels[:] = torch.randint(rows, terrain.terrain_levels.shape, device=base.device)
    base.scene.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    obs, _ = env.reset()

    contract = load_camera_contract(base.cfg.camera_profile)
    intrinsic = contract_intrinsic(contract, base.device)
    scanner = base.scene["height_scanner"]
    cast = warp_mesh_cast(scanner.meshes[scanner.cfg.mesh_prim_paths[0]])
    totals = dict(render=0, ray=0, both=0, cells=0, captures=0)
    baseline = dict(render=0, ray=0, both=0, cells=0, captures=0)
    groups = {}
    from gd_lab.mdp.terrain_families import family_column_masks
    column_family = {int(c): name for name, cols in family_column_masks(base).items() for c in cols.tolist()}
    def accumulate(counter, r, q):
        for key, value in (("render", r.sum()), ("ray", q.sum()), ("both", (r&q).sum()),
                           ("cells", r.numel()), ("captures", r.shape[0])):
            counter[key] = counter.get(key, 0) + int(value)
    def metrics(counter):
        tp=counter['both']; fp=counter['ray']-tp; fn=counter['render']-tp
        return dict(**counter, false_visible_cells=fp, missed_visible_cells=fn,
                    precision=tp/max(counter['ray'],1), recall=tp/max(counter['render'],1),
                    iou=tp/max(tp+fp+fn,1), false_positive_rate=fp/max(counter['cells']-counter['render'],1))
    started = time.perf_counter()
    per_row = torch.zeros(11, 3, device=base.device)  # render, ray, both by scan row (y)
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
            captured = base._vrl_camera_snapshot_steps == base.common_step_counter
            if not captured.any():
                continue
            rendered = obs["terrain"][:, 187:] > 0.5
            ray = raycast_visible_mask(base, contract, intrinsic, cast)
            old = raycast_visible_mask(base, contract, intrinsic, cast, enhanced=False)
            r, q = rendered[captured], ray[captured]
            accumulate(baseline, r, old[captured])
            for column in terrain.terrain_types[captured].unique().tolist():
                subset = captured & (terrain.terrain_types == column)
                name = column_family.get(column, str(column))
                counters = groups.setdefault(name, {'baseline': {}, 'enhanced': {}})
                accumulate(counters['baseline'], rendered[subset], old[subset])
                accumulate(counters['enhanced'], rendered[subset], ray[subset])
            for key, value in (("render", r.sum()), ("ray", q.sum()), ("both", (r & q).sum()),
                               ("cells", r.numel()), ("captures", captured.sum())):
                totals[key] += int(value)
            per_row += torch.stack([m.reshape(-1, 11, 17).sum((0, 2)).float() for m in (r, q, r & q)], -1)
    union = totals["render"] + totals["ray"] - totals["both"]
    summary = {
        "baseline": metrics(baseline), "enhanced": metrics(totals),
        "per_family": {name: {version: metrics(counter) for version,counter in versions.items()} for name,versions in groups.items()},
        "wall_seconds": time.perf_counter()-started,
        "comparison_time": "same capture pose, before transport latency or blackout",
        "checkpoint": checkpoint, "captures": totals["captures"],
        "render_visible_fraction": totals["render"] / max(totals["cells"], 1),
        "raycast_visible_fraction": totals["ray"] / max(totals["cells"], 1),
        "iou": totals["both"] / max(union, 1),
        "precision_raycast_vs_render": totals["both"] / max(totals["ray"], 1),
        "recall_raycast_vs_render": totals["both"] / max(totals["render"], 1),
        "cell_agreement": 1 - (union - totals["both"]) / max(totals["cells"], 1),
        "per_scan_row_render_ray_both": per_row.tolist(),
    }
    Path(args_cli.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scan_row_render_ray_both"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
