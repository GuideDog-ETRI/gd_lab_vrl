"""Small Isaac terrain restore smoke test; no policy actions or learning steps are run."""

from __future__ import annotations

import os

import pytest

if os.environ.get("GD_LAB_ISAAC_TESTS") != "1":
    pytest.skip("set GD_LAB_ISAAC_TESTS=1 to run simulator tests", allow_module_level=True)

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import gd_lab  # noqa: E402,F401
from gd_lab.mdp.terrain_families import family_column_masks  # noqa: E402
from gd_lab.rl.terrain_resume import capture_family_level_means, restore_family_start_levels  # noqa: E402


def test_small_env_save_and_restore_family_levels(tmp_path):
    task_id = "Gd-Blind-Rbq10-Dreamwaq-v0"
    env_cfg = parse_env_cfg(task_id, device="cuda:0")
    terrain_cfg = env_cfg.scene.terrain.terrain_generator
    env_cfg.scene.num_envs = terrain_cfg.num_cols
    env = gym.make(task_id, cfg=env_cfg)
    try:
        scene = env.unwrapped.scene
        terrain = scene.terrain
        families = family_column_masks(env.unwrapped)
        assert len(families) >= 3

        generator = torch.Generator(device=terrain.terrain_levels.device).manual_seed(91)
        terrain.terrain_levels[:] = torch.randint(
            terrain.terrain_origins.shape[0],
            terrain.terrain_levels.shape,
            generator=generator,
            device=terrain.terrain_levels.device,
        )
        scene.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        saved_means = capture_family_level_means(terrain, families)
        state_path = tmp_path / "terrain_state.pt"
        torch.save({"terrain_level_means_by_family": saved_means}, state_path)

        terrain.terrain_levels.zero_()
        scene.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        restored = restore_family_start_levels(
            terrain, scene.env_origins, families, state["terrain_level_means_by_family"]
        )

        for family, columns in families.items():
            target = int(saved_means[family] // 1)
            mask = torch.isin(
                terrain.terrain_types,
                torch.as_tensor(columns, device=terrain.terrain_types.device),
            )
            if not bool(mask.any()):
                continue
            assert restored[family] == target
            assert torch.all(terrain.terrain_levels[mask] == target)
        expected_origins = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        assert torch.equal(scene.env_origins, expected_origins)
    finally:
        env.close()
