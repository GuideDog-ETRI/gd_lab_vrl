"""The episode collector reads the gap monitor's clean bonus before the reset (no simulator)."""

from types import SimpleNamespace

import torch

from gd_lab.rl.online_rollout import install_episode_collector


class _Scene(SimpleNamespace):
    def __getitem__(self, key):
        return self.robot


def _fake_env(monitor_paid):
    n = 2 if monitor_paid is None else len(monitor_paid)
    terms = {
        "platform_gap_crossing": SimpleNamespace(func=SimpleNamespace(
            achieved=torch.tensor([[True, False]] * n), bypassed=torch.zeros(n, dtype=torch.bool))),
    }
    if monitor_paid is not None:
        terms["platform_gap_monitor"] = SimpleNamespace(func=SimpleNamespace(
            tracker=SimpleNamespace(paid=torch.tensor(monitor_paid))))
    scene = _Scene(
        terrain=SimpleNamespace(terrain_types=torch.zeros(n, dtype=torch.long), terrain_levels=torch.full((n,), 9),
                                cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_rows=15))),
        env_origins=torch.zeros(n, 3),
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.ones(n, 3))),
    )
    raw = SimpleNamespace(
        scene=scene,
        termination_manager=SimpleNamespace(get_term=lambda name: torch.zeros(n, dtype=torch.bool)),
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(cmd_dist_integral=torch.ones(n))),
        episode_length_buf=torch.full((n,), 100),
        step_dt=0.01,
        reward_manager=SimpleNamespace(_episode_sums={}, active_terms=list(terms),
                                       get_term_cfg=lambda name: terms[name]),
        _reset_idx=lambda ids: None,
    )
    return SimpleNamespace(unwrapped=raw)


def test_collector_records_clean_bonus_from_monitor():
    env = _fake_env([[True, False], [False, False]])
    records = install_episode_collector(env, {"platform_gap": [0]})
    env.unwrapped._reset_idx(torch.tensor([0, 1]))
    assert [row["gap_success"] for row in records] == [1.0, 1.0]
    assert [row["gap_clean_success"] for row in records] == [1.0, 0.0]


def test_collector_without_monitor_keeps_the_old_record():
    env = _fake_env(None)
    records = install_episode_collector(env, {"platform_gap": [0]})
    env.unwrapped._reset_idx(torch.tensor([0, 1]))
    assert all("gap_clean_success" not in row for row in records)
