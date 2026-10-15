import unittest,types,sys
from unittest.mock import patch
import torch
from e1_b2_metrics import summarize,FlatPitchGeometry

class Tests(unittest.TestCase):
    def test_run_relative_and_legacy_not_certified(self):
        r={'pitch':{'pre_0p7_0p4':{'mean_absolute_deg':4.,'mean_delta_deg':3.}},
           'flat_pitch_deg':1.,'reached':True,'clean':False,'foot_drop':True,'crossing_swings':[]}
        d={'approaches':[r,r],'condition':'a_student_gate','student_iteration':40624,
           'update34_flat_pitch':{'mean_deg':2.,'samples':10}}
        s=summarize(d);self.assertEqual(s['delta_pitch_deg'],2.);self.assertTrue(s['pass_delta_ge2'])
        del d['update34_flat_pitch'];del r['pitch']['pre_0p7_0p4']['mean_absolute_deg']
        s=summarize(d);self.assertEqual(s['prep_absolute_deg'],4.);self.assertIsNone(s['pass_delta_ge2'])
        self.assertEqual(s['legacy_run_delta_proxy_deg'],3.)
    def test_empty(self):
        s=summarize({'approaches':[],'condition':'a','student_iteration':1})
        self.assertIsNone(s['delta_pitch_deg']);self.assertEqual(s['pitch_approaches'],0)
    def test_all_edges_and_boundary(self):
        n=3;N=types.SimpleNamespace
        robot=N(data=N(root_pos_w=torch.tensor([[0.,0,0],[1.,0,0],[0.,0,0]])))
        class Scene:
            env_origins=torch.zeros(n,3);terrain=N(terrain_types=torch.zeros(n,dtype=torch.long))
            def __getitem__(self,key):return robot
        clean=torch.cat((torch.zeros(n,187),torch.ones(n,187)),1)
        b=N(scene=Scene(),observation_manager=N(compute_group=lambda name:clean))
        # Both left and right gap boundaries matter. Row1 too close, row0 exactly 1.3m.
        slots=torch.tensor([[[-1.4,-1.3],[1.3,1.4]]]).repeat(n,1,1)
        fake_stair=N(edges=torch.tensor([[float('nan')]]),is_stair=torch.tensor([False]))
        modules={'gd_lab.mdp.gap_stair_v2':N(_StairGeometry=lambda base:fake_stair),
                 'gd_lab.mdp.platform_gap_terms':N(boarding_family_mask=lambda base:torch.ones(n,dtype=torch.bool))}
        with patch.dict(sys.modules,modules):f=FlatPitchGeometry(b,N(for_envs=lambda base:(slots,None)))
        f.sample(torch.arange(n),torch.deg2rad(torch.tensor([1.,100.,3.])))
        self.assertEqual(f.count,2);self.assertAlmostEqual(f.result()['mean_deg'],2.,places=5)
        # Stair perimeter, rather than the forward-gap distance, rejects row1.
        f.family=lambda base:torch.zeros(n,dtype=torch.bool)
        f.stair.edges=torch.tensor([[1.4]]);f.stair.is_stair=torch.tensor([True])
        f.total=0.;f.count=0
        f.sample(torch.arange(n),torch.deg2rad(torch.tensor([1.,100.,3.])))
        self.assertEqual(f.count,2);self.assertAlmostEqual(f.result()['mean_deg'],2.,places=5)

if __name__=='__main__':unittest.main(verbosity=2)
