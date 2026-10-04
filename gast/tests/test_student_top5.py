import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location("student_top5",Path(__file__).parents[1]/"src/gd_lab/gast/student_top5.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
StudentTop5=module.StudentTop5
def metrics(v,visible=1.,supervised=1.,sample_visible=None):
    result={"latent_mse":v,"hazard_mse":0.,"extra_loss":0.,"visible_fraction":visible,
            "hazard_supervised_fraction":supervised,"updates":1}
    if sample_visible is not None:
        result["visible_sample_fraction"]=sample_visible
    return result
class StudentTop5Tests(unittest.TestCase):
    def test_sample_gate_uses_coverage_not_visible_cell_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            board=StudentTop5(tmp, min_visible_fraction=None,
                min_visible_sample_fraction=.95, smoothing_windows=1)
            save=lambda path,record: path.write_bytes(b"ckpt")
            rejected=board.consider(5000,1,metrics(1,.12,sample_visible=.94),save)
            accepted=board.consider(5064,1,metrics(1,.12,sample_visible=.99),save)
            self.assertEqual(rejected["reason"],"low_visibility_sample_coverage")
            self.assertTrue(accepted["saved"])
            self.assertEqual(accepted["visible_fraction"],.12)
            self.assertEqual(accepted["visible_sample_fraction"],.99)
    def test_start_quality_and_finite_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            board=StudentTop5(tmp); saved=[]
            def save(path,record): path.write_bytes(b"ckpt"); saved.append(record)
            self.assertEqual(board.consider(4999,1,metrics(1),save)["reason"],"before_start")
            self.assertEqual(board.consider(5000,1,metrics(1,.8),save)["reason"],"low_visibility")
            self.assertEqual(board.consider(5064,1,metrics(1,supervised=.8),save)["reason"],"low_hazard_supervision")
            self.assertEqual(board.consider(5128,float("nan"),metrics(1),save)["reason"],"invalid_metrics")
            self.assertFalse(saved)
    def test_rolling_topk_replacement_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            board=StudentTop5(tmp,keep=2,smoothing_windows=2)
            def save(path,record): path.write_text(str(record["score"]))
            a=board.consider(5000,10,metrics(10),save)
            b=board.consider(5064,2,metrics(2),save)
            c=board.consider(5128,6,metrics(6),save)
            d=board.consider(5192,1,metrics(1),save)
            self.assertFalse(a["saved"]); self.assertEqual(a["reason"],"smoothing_warmup")
            self.assertTrue(b["saved"]); self.assertAlmostEqual(b["score"],6)
            self.assertAlmostEqual(c["score"],4); self.assertAlmostEqual(d["score"],3.5)
            self.assertEqual([e["score"] for e in board.entries],[3.5,4.])
            self.assertFalse((Path(tmp)/"student_top5_iter_05064.pt").exists())
            loaded=StudentTop5(tmp,keep=2,smoothing_windows=2)
            self.assertEqual([e["score"] for e in loaded.entries],[3.5,4.])
            self.assertTrue((Path(tmp)/"criteria.json").is_file())
            self.assertTrue((Path(tmp)/"leaderboard.json").is_file())
if __name__=="__main__": unittest.main()
