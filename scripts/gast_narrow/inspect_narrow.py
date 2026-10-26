#!/usr/bin/env python3
"""Roll a GAST-Narrow checkpoint on a few envs and dump the narrow state + LiDAR maps (npz) for inspection.

  apptainer exec ... python scripts/gast_narrow/inspect_narrow.py --checkpoint <model.pt> --out <dir> [--steps 600]
Per narrow env and step: local xy, yaw, kind, width, lateral, clearance min, d_left/right, command, success.
LiDAR frames of the first env of every narrow family every 10 steps.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gast"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=62)
parser.add_argument("--level", type=int, default=None, help="force every env to this terrain level")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
from vrl_runtime import prepare_vrl_runtime  # noqa: E402
prepare_vrl_runtime(args)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import gd_lab  # noqa: E402,F401
from gd_lab.core import registry  # noqa: E402
from gd_lab.managers.action_history import ensure_prev_prev_action_tracking  # noqa: E402

TASK = "Gd-GastNarrowV1-Rbq10-Dreamwaq-v0"
registry.register_task(task="GastNarrowV1", robot="Rbq10", method="Dreamwaq",
                       env_cfg="gd_lab.gast_narrow.tasks:GastNarrowV1TeacherCfg",
                       agent_cfg="gd_lab.gast_narrow.model:GastNarrowRunnerCfg")
from gd_lab.gast_narrow import narrow_top5  # noqa: E402
from gd_lab.gast_narrow.state import narrow_state  # noqa: E402
from gd_lab.mdp.terrain_families import family_column_masks  # noqa: E402

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
import yaml  # noqa: E402
from gd_lab.core.experiments import training_arm_overrides  # noqa: E402
for item in training_arm_overrides("4"):  # the training timing/gains (train_teacher applies them via Hydra)
    key, value = item.split("=", 1)
    if not key.startswith("env."):
        continue
    *path, leaf = key[4:].split(".")
    target = env_cfg
    for part in path:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    value = yaml.safe_load(value)
    if isinstance(target, dict):
        target[leaf] = value
    else:
        setattr(target, leaf, value)
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = 7
env_cfg.events.push_robot = None
env = gym.make(TASK, cfg=env_cfg)
ensure_prev_prev_action_tracking(env.unwrapped)
raw = env.unwrapped
if args.level is not None:
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = args.level
    raw.scene.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
from gd_lab.rl.runner import DreamwaqRunner  # noqa: E402
runner = DreamwaqRunner(venv, agent_cfg.to_dict(), log_dir=None, device=raw.device)
runner.alg.policy.load_state_dict(torch.load(args.checkpoint, map_location=raw.device,
                                             weights_only=False)["model_state_dict"], strict=True)
policy = runner.get_inference_policy(device=raw.device)
obs = venv.get_observations() if hasattr(venv, "get_observations") else venv.reset()[0]
obs, _ = venv.reset()

families = {name: cols.tolist() for name, cols in family_column_masks(raw).items() if name.startswith("narrow_")}
cmd = raw.command_manager.get_term("base_velocity")
monitor = raw.reward_manager.get_term_cfg("narrow_monitor").func
lidar_term = raw.observation_manager._group_obs_class_term_cfgs["narrow_lidar"][0].func
rows = []
maps, map_meta = [], []
watch = {}
types = raw.scene.terrain.terrain_types
for name, cols in families.items():
    ids = torch.isin(types, torch.tensor(cols, device=raw.device)).nonzero().flatten()
    if len(ids):
        watch[name] = int(ids[0])
for step in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = venv.step(actions)
    s = narrow_state(raw)
    m = s.narrow.nonzero().flatten()
    rec = torch.stack((torch.full_like(s.yaw[m], step), m.float(), s.local[m, 0], s.local[m, 1], s.yaw[m],
                       s.kind[m].float(), s.width[m], s.lateral[m], s.clearance[m].amin(-1), s.d_left[m],
                       s.d_right[m], cmd.vel_command_b[m, 0], cmd.vel_command_b[m, 2],
                       monitor.success[m].float(), dones[m].float(),
                       raw.scene.terrain.terrain_levels[m].float()), -1)
    rows.append(rec.cpu().numpy())
    if step % 10 == 0:
        for name, i in watch.items():
            maps.append(lidar_term.frames[i].cpu().numpy())
            geo = raw.scene.terrain.cfg.terrain_generator.narrow_tile_metadata[
                int(raw.scene.terrain.terrain_levels[i])][int(types[i])]
            map_meta.append((step, name, i, float(s.local[i, 0]), float(s.local[i, 1]), float(s.yaw[i]),
                             repr(geo["rects"]), geo["width"]))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "narrow_inspect.npz", rows=np.concatenate(rows), maps=np.stack(maps),
                    map_meta=np.array(map_meta, dtype=object), columns=np.array(
                        "step env x y yaw kind width lateral clearance d_left d_right vx wz success done level".split()))
print(f"[INSPECT] wrote {out/'narrow_inspect.npz'} rows={sum(len(r) for r in rows)} maps={len(maps)}", flush=True)
env.close()
app.close()
