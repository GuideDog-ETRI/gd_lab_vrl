import unittest
import torch
from gd_lab.gast.mapper_aug import MapperAugCfg, MapperAugmentation
from test_mapper_aug149 import fixture


class Adjust149B(unittest.TestCase):
    def test_cap_on_sparse_validity_and_ir_preservation(self):
        h,v,ir,args,cast=fixture(32)
        v[:, ::2]=False;h[~v]=0;ir[~v]=0
        a=MapperAugmentation(32,'cpu',MapperAugCfg(enabled=True))
        a.params[:,0]=.99;a.params[:,1]=0;a.params[:,12:]=0
        a.params[:,6:8]=.5;a.params[:,11]=1
        out=a(h,v,ir,*args,50000,cast)
        fraction=(v & ~out[1]).sum(-1)/v.sum(-1)
        self.assertTrue((fraction<=.15).all());self.assertTrue((fraction>0).any())
        self.assertTrue(torch.equal(out[2][out[1]],ir[out[1]]))
        self.assertTrue((out[2][~out[1]]==0).all())
        self.assertTrue(torch.equal(v | out[1],v))

    def test_metric_widening_total_including_s0_phase(self):
        # Dense 1mm half-plane edge: measure realized expansion, not tensor-grid pooling.
        x=torch.linspace(-.3,.3,601);z=torch.where(x>=0,-.65,0.)
        points=torch.stack((x,torch.zeros_like(x),z),-1)[None]
        def cast(starts,dirs,maxdist):return starts[:,2]-torch.where(starts[:,0]>=0,-.65,0.)
        for phase in (0.,1.):
            a=MapperAugmentation(1,'cpu',MapperAugCfg(enabled=True));a.params[:]=.99
            a.params[:,1]=0;a.params[:,2:4]=1;a.params[:,4]=phase
            q=a.geometry(points,torch.tensor([.55]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor([True]),50000,cast)
            outward=-x[(q[0]<-.3)&(x<0)].min()
            self.assertLessEqual(float(outward),.100001)
            self.assertGreater(float(outward),.08)
            self.assertLessEqual(float(a.last_diagnostics['outward_query_max_m']),.100001)
            self.assertTrue(torch.equal(q,a.geometry(points,torch.tensor([.55]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor([True]),50000,cast)))

    def test_geometry_before_single_noise_clip_and_local_clock(self):
        h,v,ir,args,cast=fixture(2)
        a=MapperAugmentation(2,'cpu',MapperAugCfg(enabled=True));a.params[:]=.2;a.params[:,0]=.99;a.params[:,1]=0
        points,scanner,support,yaw,family=args
        self.assertTrue(torch.equal(a.geometry(*args,0,cast),points[...,2]))
        a.geometry(*args,25000,cast);self.assertEqual(a.last_diagnostics['strength'],.5)
        z=a.geometry(*args,50000,cast)
        noise=torch.full_like(h,.4)
        encoded=((scanner[:,None]-z-.5).clamp(-1,1)+noise).clamp(-1,1)*5
        out=a.unknown(encoded,v,ir,points,yaw)
        self.assertTrue(torch.equal(out[0][out[1]],encoded[out[1]]))
        self.assertTrue(torch.equal(out[2][out[1]],ir[out[1]]))
