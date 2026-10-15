import unittest, numpy as np, sys, os
from pathlib import Path
from heightmap116 import ElevationMap, R
from heightmap117 import BatchMap
class BatchTests(unittest.TestCase):
    def test_equivalence_integrate_sample_reset(self):
        n=4;b=BatchMap(n);old=[ElevationMap('h1') for _ in range(n)]
        rng=np.random.default_rng(117)
        depth=rng.uniform(.7,1.3,(n,4,45,80));valid=rng.random(depth.shape)>.1
        ir=np.ones_like(depth)*.8;ir[:,:,:,30:45]=.1
        positions=np.tile([0.,0.,.8],(n,4,1));quats=np.tile([1.,0,0,0],(n,4,1))
        ks=np.tile([[45.,0,40],[0,45,22.5],[0,0,1.]],(n,4,1,1));body=np.tile([0.,0,.8],(n,1));bq=np.tile([1.,0,0,0],(n,1))
        for tick in range(5):
            stamp=tick*.1;body[:,0]+=.025
            b.integrate(list(range(n)),depth,valid,ir,positions,quats,ks,body,bq,stamp)
            for e in range(n):
                old[e].integrate(depth[e],valid[e],ir[e],positions[e],quats[e],ks[e],body[e],bq[e],stamp)
                for key in b.arrays:np.testing.assert_allclose(getattr(b.maps[e],key),getattr(old[e],key),atol=1e-12,err_msg=key)
            poses=np.c_[body[:,:2],np.linspace(-.3,.3,n)];scanner=np.full(n,.8)
            np.testing.assert_allclose(b.sample(poses,scanner,stamp),np.stack([m.sample(poses[e],scanner[e],stamp)[0] for e,m in enumerate(old)]),atol=1e-6)
        b.reset([1,3]);self.assertFalse(b.sample(poses,scanner,.6)[[1,3]].any())
    def test_512_batch_and_sparse_fill(self):
        n=512;b=BatchMap(n);rng=np.random.default_rng(7)
        for m in b.maps:m.origin=np.array([-40,-40])
        b.arrays['z'][:]=rng.uniform(-.01,.01,b.arrays['z'].shape);b.arrays['var'][:]=0;b.arrays['last'][:]=0;b.arrays['finite'][:]=0;b.arrays['ground'][:]=2;b.arrays['noise'][:,35:40,35:40]=3;b.arrays['ground'][:,35:40,35:40]=0
        poses=np.zeros((n,3));scanner=np.full(n,.5)
        actual=b.sample(poses,scanner,.1)
        for e in (0,127,255,511):np.testing.assert_array_equal(actual[e],b.maps[e].sample(poses[e],scanner[e],.1)[0])
    def test_teacher_branch_before_student_construction(self):
        r=Path(__file__).resolve().parents[2];s=(r/'scripts/gast/train_student_live.py').read_text()
        self.assertLess(s.index('from e1_heightmap117 import run'),s.index('student = GastStudent('))
    def test_source_hash_report_repair(self):
        import hashlib,json
        r=Path(__file__).resolve().parents[2];p=json.loads((r/'protocol.json').read_text())
        self.assertEqual(hashlib.sha256((r/'report116.py').read_bytes()).hexdigest(),p['implementation_sha256']['report116.py'])
if __name__=='__main__':unittest.main(verbosity=2)
