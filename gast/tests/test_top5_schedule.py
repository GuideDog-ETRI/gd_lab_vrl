import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('online_top5', Path(__file__).parents[1]/'src/gd_lab/rl/online_top5.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

class Top5Tests(unittest.TestCase):
    def test_schedule_admission_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = [dict(family=family, level=9, gap_success=1., base_contact=0., progress=.9,
                             tracking=.9, energy=.9, **{'return': .9})
                        for family in ('platform_gap', 'pyramid_stairs', 'flat')]
            def save(path): Path(path).write_bytes(b'test checkpoint')
            board = root/'best_top5'
            before = t.rank_and_save_top5(0, episodes, board, 4999, 1, save)
            self.assertFalse(before['saved']); self.assertFalse(board.exists())
            for iteration in range(5000,5006):
                result = t.rank_and_save_top5(0, episodes, board, iteration, 1, save)
                self.assertIsNone(result['error'])
            entries = json.loads((board/'leaderboard.json').read_text())['entries']
            self.assertEqual([r['iteration'] for r in entries], list(range(5000,5005)))
            episodes[0]['base_contact'] = .10
            result = t.rank_and_save_top5(0, episodes, board, 5006, 1, save)
            self.assertFalse(result['saved'])
            criteria_file = root/'criteria.json'
            criteria_file.write_text(json.dumps({'top5_start_iteration': 5000, 'top5_min_spacing': 1}))
            reader = t.Top5CriteriaReloader(criteria_file)
            self.assertEqual(reader.refresh()[0]['top5_start_iteration'],5000)
            criteria_file.write_text(json.dumps({'max_base_contact_rate': .08}))
            self.assertEqual(reader.refresh()[0]['max_base_contact_rate'],.08)
            criteria_file.write_text('{broken')
            self.assertEqual(reader.refresh()[0]['max_base_contact_rate'],.08)

if __name__ == '__main__': unittest.main()
