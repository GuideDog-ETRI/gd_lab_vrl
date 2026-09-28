from types import SimpleNamespace

import torch

from gd_lab.rl.terrain_resume import capture_family_level_means, restore_family_start_levels


def _terrain():
    terrain_types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    levels = torch.tensor([2, 4, 5, 9, 0], dtype=torch.long)
    terrain_origins = torch.arange(4 * 3 * 3, dtype=torch.float).reshape(4, 3, 3)
    env_origins = torch.zeros((5, 3), dtype=torch.float)
    terrain = SimpleNamespace(
        terrain_types=terrain_types,
        terrain_levels=levels,
        terrain_origins=terrain_origins,
        env_origins=env_origins,
    )
    return terrain, env_origins


def test_capture_family_level_means():
    terrain, _ = _terrain()
    means = capture_family_level_means(terrain, {"stairs": [0], "gap": [1], "rough": [2]})
    assert means == {"stairs": 3.0, "gap": 7.0, "rough": 0.0}


def test_restore_uses_floor_and_updates_levels_and_origins():
    terrain, env_origins = _terrain()
    columns = {"stairs": [0], "gap": [1], "rough": [2]}
    means = {"stairs": 3.9, "gap": 2.0, "rough": -0.2}

    restored = restore_family_start_levels(terrain, env_origins, columns, means)

    assert restored == {"stairs": 3, "gap": 2, "rough": 0}
    assert terrain.terrain_levels.tolist() == [3, 3, 2, 2, 0]
    expected = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    assert torch.equal(env_origins, expected)


def test_restore_clamps_to_available_maximum():
    terrain, env_origins = _terrain()
    restored = restore_family_start_levels(terrain, env_origins, {"gap": [1]}, {"gap": 99.0})
    assert restored == {"gap": 3}
    assert terrain.terrain_levels.tolist()[2:4] == [3, 3]
