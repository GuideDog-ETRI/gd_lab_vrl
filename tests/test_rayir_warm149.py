import hashlib,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from gd_lab.gast.warm_start import warm_start_from_rayir

class Policy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.actor=torch.nn.Linear(2,2);self.terrain_encoder=torch.nn.Linear(2,2);self.cenet=torch.nn.Linear(2,2);self.cenet.optimizer=torch.optim.Adam(self.cenet.parameters())

class TestWarm149(unittest.TestCase):
 def test_exact_copy_reset_ppo_restore_cenet_and_reject_mismatch(self):
  source=Policy();source.cenet(torch.ones(1,2)).sum().backward();source.cenet.optimizer.step()
  task='Gd-GastRayIrV21-Rbq10-Dreamwaq-v0'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'native.pt';torch.save(dict(model_state_dict=source.state_dict(),iter=2769,infos={'gd_lab':dict(gast_warm_start={'task':task},terrain_level_means_by_family={'platform_gap':9.38},cenet_optimizer_state_dict=source.cenet.optimizer.state_dict())}),p)
   sha=hashlib.sha256(p.read_bytes()).hexdigest();target=Policy();alg=SimpleNamespace(policy=target,optimizer=torch.optim.Adam(target.parameters()),schedule='fixed')
   report=warm_start_from_rayir(alg,str(p),1e-4,1e-3,expected_sha256=sha,expected_task=task)
   for k,v in source.state_dict().items():self.assertTrue(torch.equal(v,target.state_dict()[k]))
   self.assertEqual(len(alg.optimizer.state),0);self.assertEqual(len(target.cenet.optimizer.state),2)
   self.assertEqual(report['source_iteration'],2769);self.assertEqual(report['terrain_level_means_by_family']['platform_gap'],9.38)
   with self.assertRaises(ValueError):warm_start_from_rayir(alg,str(p),1e-4,1e-3,expected_sha256='bad',expected_task=task)
   with self.assertRaises(ValueError):warm_start_from_rayir(alg,str(p),1e-4,1e-3,expected_sha256=sha,expected_task='wrong')
if __name__=='__main__':unittest.main()
