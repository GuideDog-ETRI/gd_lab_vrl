import unittest,json,time
from unittest.mock import patch
import numpy as np
from heightmap121 import *
from sensor121 import PatchField,generate,physical_ir,rng,materials

class CPUFixtures(unittest.TestCase):
    def scene(self,kind='fill',offset=0):
        h,w=45,80;k=np.array([[60.,0,40],[0,60,22],[0,0,1.]])
        q=np.array([1.,0,0,0]);pos=np.array([offset,0,1.]);ray=rays(k,q,h,w)
        depth=np.ones((h,w));mask=np.zeros((h,w),bool);mask[15:23,20:60]=True
        ir=np.full((h,w),.65);ir[mask]=.08;valid=np.ones_like(mask)
        if kind=='deep':depth[mask]=1.6
        if kind=='missing':valid[mask]=False;depth[mask]=5
        return depth[None],valid[None],ir[None],pos[None],q[None],k[None],np.array([offset,-1.,.5]),q
    def test_candidate_shapes(self):
        y,x=np.mgrid[:45,:80]
        for mask in [(abs(y-.25*x-10)<2)&(x>10)&(x<65),(abs(y-(20+5*np.sin(x/15)))<2)&(x>10)&(x<65),((x>15)&(x<65)&(y>14)&(y<23))]:
            im=np.where(mask,.1,.6);self.assertGreater(candidates(im).sum(),20)
        for mask in [(x<8)&(y>10)&(y<30),(y==20)&(x>10)&(x<65),(x==30)&(y>10)&(y<35),(y==x)&(x>10)&(x<35)]:
            self.assertFalse(candidates(np.where(mask,.1,.6)).any())
        im=np.full((45,80),.6);im[15:23,20:60]=-1;self.assertFalse(candidates(im).any())
        im[15:23,20:60]=1;self.assertFalse(candidates(im).any())
    def test_degenerate_finite_ir_regions(self):
        # A valid thick candidate becomes singleton/pair/collinear after &finite.
        shapes=[[(18,30)],[(18,30),(18,31)],
                [(18,x) for x in range(25,45)],
                [(y,30) for y in range(16,22)]]
        original_svd=np.linalg.svd
        for mode in ('h3','h3u'):
            for coords in shapes:
                with self.subTest(mode=mode,coords=coords):
                    data=list(self.scene())
                    dark=candidates(data[2][0]);self.assertTrue(dark.any())
                    data[1][0,dark]=False
                    for y,x in coords:data[1][0,y,x]=True
                    self.assertEqual((dark&data[1][0]).sum(),len(coords))
                    seen=[]
                    def checked_svd(a,*args,**kwargs):
                        if kwargs.get('full_matrices') is False:
                            self.assertGreaterEqual(len(a),3)
                            self.assertEqual(np.linalg.matrix_rank(a.T@a/(len(a)-1)),2)
                            seen.append(a.shape)
                        return original_svd(a,*args,**kwargs)
                    with patch('heightmap121.np.linalg.svd',side_effect=checked_svd):
                        H3Map(mode).integrate(*data,0)
                    self.assertTrue(seen)  # Nondegenerate candidates still use PCA.

    def test_deep_and_missing(self):
        deep=H3Map('h3');deep.integrate(*self.scene('deep'),0)
        measured=[v.z for v in deep.cells.values() if np.isfinite(v.z)]
        self.assertTrue(any(z<-.5 for z in measured));self.assertFalse(any(v.latch for v in deep.cells.values()))
        missing=H3Map('h3');missing.integrate(*self.scene('missing'),0)
        self.assertTrue(any(v.blocked for v in missing.cells.values()))
        self.assertFalse(any(np.isfinite(v.z) and v.z<-2 for v in missing.cells.values()))
    def confirm(self,mode='h3'):
        m=H3Map(mode)
        for i,t in enumerate([0,.1,.21]):m.evidence((0,0),t,i,[i*.06,0],0,{(0,i)},positive=True)
        return m
    def test_multiview_latch_ttl_provenance(self):
        for mode in ['h2m','h3','h3u']:
            m=self.confirm(mode);h,v,src,gap=m.sample_points([[.01,.01]],.5,.25)
            self.assertTrue(gap[0]);self.assertEqual(src[0],INFERRED_GAP);self.assertEqual(v[0],mode!='h3u');self.assertEqual(h[0],5 if mode!='h3u' else 0)
            for t in [.3,.6,1.2]:self.assertTrue(m.sample_points([[.01,.01]],.5,t)[3][0])
            self.assertFalse(m.sample_points([[.01,.01]],.5,1.211)[3][0]);self.assertEqual(m.sample_points([[.01,.01]],.5,1.211)[2][0],UNKNOWN)
    def test_indistinguishable_dark_flat_shadow(self):
        a=H3Map('h3');b=H3Map('h3')
        for i,t in enumerate([0,.1,.21]):
            data=self.scene('fill',i*.06);a.integrate(*data,t);b.integrate(*data,t)
        self.assertEqual(a.cells.keys(),b.cells.keys())
        self.assertEqual(sum(v.latch for v in a.cells.values()),sum(v.latch for v in b.cells.values()))
        # Same measurements -> same cost; no fixture can demand opposite labels.
        self.assertGreater(sum(v.latch for v in a.cells.values()),0)
        print('INDISTINGUISHABLE dark paint/shadow vs false-fill; false_gap_cells=',sum(v.latch for v in b.cells.values()))
    def test_duplicate_reverse_delay_pose_reset(self):
        m=H3Map('h3');m.integrate(*self.scene(),.2);score=sum(v.score for v in m.cells.values())
        m.integrate(*self.scene(),.2);m.integrate(*self.scene(),.1);self.assertEqual(score,sum(v.score for v in m.cells.values()))
        n=H3Map('h3');n.integrate(*self.scene(),.2,now=.6);self.assertFalse(n.cells)
        n.integrate(*self.scene(),.7,pose_sigma=.06);self.assertFalse(any(v.latch for v in n.cells.values()))
        m.reset();self.assertFalse(m.cells)
        n=H3Map('h3')
        for i,t in enumerate([0,.1,.21]):n.evidence((0,0),t,i,[0,0],0,{(0,0)},positive=True)
        self.assertFalse(n.cells[(0,0)].latch)
    def test_collision_release(self):
        m=self.confirm();m.evidence((0,0),.3,3,[.2,0],0,{(0,0)},conflict=True)
        self.assertEqual(m.sample_points([[.01,.01]],.5,.3)[2][0],UNKNOWN)
        for i in range(2):m.evidence((0,0),.4+i*.1,4+i,[.2,0],0,{(0,0)},z=0,normal_ir=True)
        self.assertTrue(m.cells[(0,0)].blocked)
        m.evidence((0,0),.6,6,[.2,0],0,{(0,0)},z=0,normal_ir=True)
        self.assertFalse(m.cells[(0,0)].blocked);self.assertEqual(m.sample_points([[.01,.01]],.5,.6)[2][0],OBSERVED_HEIGHT)
    def test_stairs_and_shallow_gap(self):
        for height in [.1,.25,.35,-.1,-.25,-.35]:
            data=list(self.scene('deep'));data[0][0,15:23,20:60]=1-height
            m=H3Map('h3');m.integrate(*data,0)
            self.assertFalse(any(c.latch for c in m.cells.values()))
            self.assertTrue(any(np.isfinite(c.z) and abs(c.z-height)<1e-6 for c in m.cells.values()))
        # Multiple treads, nose and wide tread: actual finite measurements survive.
        for widths in [[8,8,8],[2,12,10],[16,8]]:
            data=list(self.scene());d=data[0][0];start=8
            for i,width in enumerate(widths):d[start:start+width]=1+.15*(i+1);start+=width
            m=H3Map('h3');m.integrate(*data,0);self.assertFalse(any(c.latch for c in m.cells.values()))
        points=np.zeros((12,16,3));y,x=np.mgrid[:12,:16];points[...,0]=x*.02;points[...,1]=y*.02
        valid=np.ones((12,16),bool);valid[:,7:9]=False
        self.assertFalse(tread_certificate(points,valid,support_normals(points,valid)).any())
        # Observed lower tread joins a vertical riser through finite pixels.
        for height in [.1,.25,.35]:
            y,x=np.mgrid[:24,:20];p=np.zeros((24,20,3));p[...,0]=x*.02
            p[...,1]=np.where(y<10,y*.02,np.where(y<15,.18,.18+(y-14)*.02))
            p[...,2]=np.where(y<10,0,np.where(y<15,(y-9)*height/5,height))
            v=np.ones((24,20),bool);support=support_normals(p,v)
            self.assertTrue(tread_certificate(p,v,support)[:9].any())
            v[9:11]=False
            self.assertFalse(tread_certificate(p,v,support_normals(p,v))[:9].any())
    def test_h2m_missing_ir_independence_and_shared_branch(self):
        signatures=[]
        for mode,brightness in [('h2m',.1),('h2m',.8),('h2m',np.nan),('h3',.8)]:
            m=H3Map(mode)
            for i,t in enumerate([0,.1,.21]):
                data=list(self.scene('missing',i*.06));data[2][:]=brightness
                m.integrate(*data,t)
            latched={k for k,c in m.cells.items() if c.latch}
            self.assertTrue(latched);signatures.append(latched)
        for signature in signatures[1:]:self.assertEqual(signatures[0],signature)
        for brightness in [.08,.8,np.nan]:
            m=H3Map('h2m');data=list(self.scene('fill'));data[2][:]=brightness
            m.integrate(*data,0)
            self.assertFalse(any(c.blocked or c.latch for c in m.cells.values()))
    def test_h2m_one_boundary_and_missing_lower_tread(self):
        for mode in ['h2m','h3']:
            m=H3Map(mode)
            for i,t in enumerate([0,.1,.21]):
                data=list(self.scene('missing',i*.06))
                data[1][0,15:23,:60]=False;data[0][0,15:23,:60]=5
                m.integrate(*data,t)
            self.assertFalse(any(c.latch for c in m.cells.values()))
        # Missing lower tread has no measured tread/riser certificate; expose
        # the indistinguishable false-gap cost rather than inventing geometry.
        for height in [.1,.25,.35]:
            m=H3Map('h2m')
            for i,t in enumerate([0,.1,.21]):m.integrate(*self.scene('missing',i*.06),t)
            cost=sum(c.latch for c in m.cells.values())
            print('H2M missing lower tread',height,'m; indistinguishable false_gap_cells=',cost)
            self.assertGreater(cost,0)
    def test_h2m_release_without_ir(self):
        m=self.confirm('h2m')
        for i,t in enumerate([.3,.4,.5]):
            m.evidence((0,0),t,i+3,[.2,0],0,{(0,0)},z=0,normal_ir=True)
        self.assertFalse(m.cells[(0,0)].blocked)

    def test_optical_units_and_sensor_independence(self):
        p=np.array([0,0,1.]);ray=np.array([[.5,0,-.5]])
        self.assertAlmostEqual(plane_intersection(p,ray,np.array([0,0,.2]))[0],1.6)
        self.assertAlmostEqual((p+ray*1.6)[0,2],.2)
        points=np.array([[[0.,0,0.],[1.,0,0.]]]);norm=np.broadcast_to([0,0,1.],points.shape);hit=np.ones((1,2),bool)
        args=(points,norm,hit,np.array([0,0,1.]),hit)
        a=physical_ir(*args,rng(42,0,1,2,0,1));b=physical_ir(*args,rng(42,0,1,2,0,1))
        np.testing.assert_array_equal(a[0],b[0]);self.assertNotEqual(a[1]['ir_clean'][0,0],a[1]['ir_clean'][0,1])
    def test_patch_world_area(self):
        field=PatchField((0,.3,-.5,.5),42)
        self.assertAlmostEqual(field.area_fraction,.25,places=3)
        for p in field.polygons:self.assertTrue(((p[:,0]>=0)&(p[:,0]<=.3)&(abs(p[:,1])<=.5)).all())
        xy=np.array([[.1,.2],[.1,.3]]);np.testing.assert_array_equal(field.sample(xy)[0],field.sample(xy.copy())[0])

if __name__=='__main__':unittest.main(verbosity=2)
