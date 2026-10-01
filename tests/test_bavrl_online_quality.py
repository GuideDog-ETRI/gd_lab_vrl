import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_adapter():
    # File loading avoids simulator/package initialization in this CPU-only test.
    name = 'gd_lab.rl.online_top5'
    spec = importlib.util.spec_from_file_location(name, ROOT/'src/gd_lab/rl/online_top5.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    spec = importlib.util.spec_from_file_location('bavrl_quality', ROOT/'src/gd_lab/residuals/bavrl/online_quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OnlineQuality


def test_empty_then_eligible_checkpoint(tmp_path):
    quality = load_adapter()(tmp_path, ROOT/'configs/online_top5.json')
    saved = []
    def save(path):
        saved.append(path)
        Path(path).write_text('checkpoint')
    row = quality.update(4401, [], save)
    assert row['failed_gates'] == ['no_completed_gap_episode']
    assert not saved
    episodes = [dict(family=family, level=9, gap_success=1., base_contact=0.,
                     progress=.8, tracking=.8, energy=.8, **{'return': .8})
                for family in ['platform_gap', 'pyramid_stairs']]
    row = quality.update(4402, episodes, save)
    assert row['saved'] and not row['failed_gates']
    assert (tmp_path/'best_top5/4402_top1.pt').is_file()
    assert json.loads((tmp_path/'best_top5/leaderboard.json').read_text())['entries'][0]['iteration'] == 4402
    assert len((tmp_path/'quality_samples.jsonl').read_text().splitlines()) == 2


def test_low_difficulty_does_not_relax_gate(tmp_path):
    quality = load_adapter()(tmp_path, ROOT/'configs/online_top5.json')
    episodes = [dict(family='platform_gap', level=1, gap_success=1., base_contact=0.,
                     progress=1., tracking=1., energy=1., **{'return': 1.})]
    row = quality.update(4401, episodes, lambda _: None)
    assert not row['saved']
    assert 'platform_gap_mean_level_pass' in row['failed_gates']
    assert 'stairs_termination_pass' in row['failed_gates']
