import unittest
from dataclasses import replace
import numpy as np
from sensor_independent116 import corrupt, RULES
from heightmap116 import dark_regions

class Sensors(unittest.TestCase):
    def call(self, mode='realistic', rules=RULES, depth=None, ir=None):
        d = np.ones((4, 180, 320), np.float32) if depth is None else depth
        p = np.full(d.shape, .6, np.float32) if ir is None else ir
        return corrupt(d, np.isfinite(d)&(d<5), p, None, None, None, None,
                       np.random.default_rng(42), mode, 'proxy', rules)
    def test_clean_exact_and_alignment(self):
        d=np.linspace(.2,4.9,4*45*80,dtype=np.float32).reshape(4,45,80)
        ir=np.arange(d.size,dtype=np.float32).reshape(d.shape)/d.size
        out=self.call('clean',depth=d,ir=ir)
        np.testing.assert_array_equal(out[0],d)
        np.testing.assert_array_equal(out[2],ir)
    def test_depth_changes_cannot_change_ir(self):
        a=self.call();b=self.call(rules=replace(RULES,depth_sigma_m=.5,depth_dropout=.9))
        np.testing.assert_array_equal(a[2],b[2])
    def test_ir_changes_cannot_change_depth(self):
        a=self.call();b=self.call(rules=replace(RULES,ir_sigma=.3,ir_dropout=.8))
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    def test_depth_missing_bright_ir(self):
        d=np.full((4,45,80),5.,np.float32)
        a=self.call('depth_only',depth=d)
        self.assertFalse(a[1].any());np.testing.assert_allclose(a[2],.6)
    def test_single_modality_isolation(self):
        a=self.call('depth_only');b=self.call('ir_only')
        np.testing.assert_allclose(a[2],.6);np.testing.assert_allclose(b[0],1)
        self.assertTrue(b[1].all());self.assertGreater((a[0]==5).sum(),0)
    def test_drop_independence_and_noise_scales(self):
        d,v,ir,c=self.call()
        n=c['pixels'];self.assertAlmostEqual(c['depth_dropout']/n,.03,delta=.002)
        self.assertAlmostEqual(c['ir_dropout']/n,.03,delta=.002)
        self.assertAlmostEqual(c['both_dropout']/n,.03**2,delta=.0003)
        self.assertGreater(c['depth_only_dropout'],5000);self.assertGreater(c['ir_only_dropout'],5000)
        self.assertAlmostEqual(float(d[v].std()),.01,delta=.0002)
        self.assertAlmostEqual(float(ir[ir>=0].std()),.015,delta=.0003)
    def test_ir_dropout_not_dark_band(self):
        ir=np.full((45,80),.6);ir[:,20:50]=-1
        self.assertFalse(dark_regions(ir).any())
    def test_repeatability(self):
        a=self.call();b=self.call()
        for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):self.call(ir=np.ones((1,3,4)))

if __name__=='__main__':unittest.main(verbosity=2)
