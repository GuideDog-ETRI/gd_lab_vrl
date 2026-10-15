"""Analytic CPU fixtures: metric pinhole rays intersect treads AND risers."""
import json,time,unittest
from pathlib import Path
import numpy as np
from heightmap116 import *
K=np.array([[45.,0,40],[0,45.,22.5],[0,0,1.]])
Q=np.array([1.,0,0,0]);BODY=np.array([0.,0,.8])
P=np.array([.5,0,.8])
def scene(edges=(),levels=(0.,),pos=P,quat=Q):
    ray=rays(K,quat,45,80);depth=np.full((45,80),5.);bounds=[-100,*edges,100]
    for lo,hi,z in zip(bounds[:-1],bounds[1:],levels):
        t=(z-pos[2])/ray[...,2];x=pos[0]+t*ray[...,0]
        ok=(t>.15)&(x>=lo)&(x<hi)&(t<depth);depth[ok]=t[ok]
    for edge,z0,z1 in zip(edges,levels[:-1],levels[1:]):
        t=np.divide(edge-pos[0],ray[...,0],out=np.full((45,80),np.inf),where=np.abs(ray[...,0])>1e-8)
        z=pos[2]+t*ray[...,2];ok=(t>.15)&(z>=min(z0,z1))&(z<=max(z0,z1))&(t<depth);depth[ok]=t[ok]
    return depth[None],(depth<5)[None],ray

def feed(m,d,v,ir,stamp,pos=P,quat=Q,body=BODY):m.integrate(d,v,ir,np.array([pos]),np.array([quat]),K[None],body,Q,stamp)
def grid(m,stamp=.8,pose=(0,0,0)):return m.sample(pose,.8,stamp)
class Fixtures(unittest.TestCase):
    results={}
    def record(self,name,m,expected=None):
        h,g=grid(m);valid=h[187:].astype(bool)
        self.results[name]={'valid_cells':int(valid.sum()),'gap_cells':int(g.sum()),'false_gap_rate':float(g.mean()) if name in ('step','up','down','nose','wide') else None}
        if expected is not None:
            z=.8-.5-h[:187]/5;err=np.abs(z[valid]-expected[valid]);self.results[name]['mae_m']=float(err.mean()) if len(err) else None
        return h,g,valid
    def test_flat(self):
        d,v,_=scene();m=ElevationMap()
        for t in np.arange(0,.81,.1):feed(m,d,v,np.ones_like(d)*.7,t)
        h,g,valid=self.record('flat',m,np.zeros(187));self.assertGreater(valid.sum(),40);self.assertFalse(g.any());np.testing.assert_allclose(h[:187][valid],1.5,atol=1e-5)
    def test_step_and_stairs(self):
        x=np.tile(np.linspace(-.8,.8,17),11)
        for name,edges,levels in [('step',[.35],[0,.2]),('up',[.2,.5,.75],[0,.15,.30,.45]),('down',[.2,.5,.75],[0,-.15,-.30,-.45]),('nose',[.2,.23,.5,.53],[0,.2,.17,.37,.34]),('wide',[.25,.65],[0,-.2,-.4])]:
            d,v,_=scene(edges,levels);ir=np.full_like(d,.65);ir[:,:,45:70]=.04;m=ElevationMap()
            for t in np.arange(0,.81,.1):feed(m,d,v,ir,t)
            expected=np.asarray(levels)[np.searchsorted(edges,x,side='right')]
            h,g,valid=self.record(name,m,expected);self.assertFalse(g.any(),name);self.assertGreater(valid.sum(),15,name)
            self.assertLess(self.results[name]['mae_m'],.055,name)
    def test_gap_invalid_and_falsefill(self):
        d,v,ray=scene([.35,.65],[0,-1.2,0]);plane=.8/-ray[...,2];px=P[0]+plane*ray[...,0];region=(px>.35)&(px<.65)
        ir=np.full_like(d,.7);ir[:,region]=.025
        for name,dd,vv in [('gap',d,v),('no_return',np.where(region[None],5,d),v&~region[None]),('false_fill',np.where(region[None],plane,d),v)]:
            m=ElevationMap()
            for t in np.arange(0,.81,.1):feed(m,dd,vv,ir,t)
            h,g,valid=self.record(name,m)
            if name!='false_fill':self.assertGreater(g.sum(),0,name)
            else:self.assertFalse(g.any()) # conservative finite-return veto; known limitation
        m=ElevationMap();blank=np.full_like(d,5);vv=np.zeros_like(v)
        for t in np.arange(0,.81,.1):feed(m,blank,vv,np.ones_like(d)*.7,t)
        h,g,valid=self.record('invalid_normal_ir',m);self.assertFalse(valid.any());self.assertFalse(g.any())
    def test_movement_rotation_ttl_reset(self):
        d,v,_=scene();m=ElevationMap('h0');feed(m,d,v,np.ones_like(d),0)
        h0,_=grid(m,0)
        for body,angle in [(np.array([.1,0,.8]),0),(np.array([.1,0,.8]),.5)]:
            q=np.array([np.cos(angle/2),0,0,np.sin(angle/2)]);p=np.array([.6,0,.8]);d,v,_=scene(pos=p,quat=q)
            feed(m,d,v,np.ones_like(d),.2+angle,pos=p,quat=q,body=body)
            h,g=m.sample([.1,0,angle],.8,.2+angle);self.assertGreater(h[187:].sum(),30);np.testing.assert_allclose(h[:187][h[187:]>0],1.5,atol=1e-4)
        feed(m,d,v,np.ones_like(d),.1);self.assertEqual(m.last_capture,.7)
        h,_=grid(m,4);self.assertFalse(h[187:].any());m.reset();self.assertFalse(np.isfinite(m.z).any())
    def test_noise_models(self):
        d,v,_=scene([.35,.65],[0,-1.2,0]);args=(d,v,np.ones_like(d)*.6,P[None],Q[None],K[None],0)
        outputs=[]
        for mode in ['proxy','synthetic']:
            a=corrupt(*args,np.random.default_rng(42),True,mode);b=corrupt(*args,np.random.default_rng(42),True,mode)
            for aa,bb in zip(a[:3],b[:3]):np.testing.assert_array_equal(aa,bb)
            self.assertTrue(all(n>0 for n in a[3].values()),a[3]);outputs.append(a)
            self.results['noise_'+mode]=a[3]
        self.assertFalse(np.array_equal(outputs[0][2],outputs[1][2]))
        for index in (0,1):np.testing.assert_array_equal(outputs[0][index],outputs[1][index])
        self.assertEqual(outputs[0][3],outputs[1][3])
    def test_ir_polygon_curved_relative(self):
        yy,xx=np.mgrid[:45,:80]
        masks={'trapezoid':(xx>25-.2*yy)&(xx<40+.25*yy),
               'oblique':(xx>15+.5*yy)&(xx<25+.5*yy),
               'curved':(xx>30+7*np.sin(yy/9))&(xx<40+7*np.sin(yy/9))}
        for name,mask in masks.items():
            ir=np.where(mask,.2,.8);found=dark_regions(ir)
            self.assertGreater((found&mask).sum()/mask.sum(),.9,name)
            for gain,offset in [(0.1,.4),(1.1,0),(.001,.001)]:
                np.testing.assert_array_equal(found,dark_regions(gain*ir+offset))
            d,v,_=scene();d[:,mask]=5;v[:,mask]=False;m=ElevationMap()
            for t in np.arange(0,.81,.1):feed(m,d,v,ir[None],t)
            _,g,_=self.record(name+'_gap',m);self.assertGreater(g.sum(),0,name)
        self.assertFalse(dark_regions(np.ones((45,80))*.01).any())
        self.assertFalse(dark_regions(np.where(xx<40,.1,.8)).any())
    def test_shadow_only_non_gap(self):
        yy,xx=np.mgrid[:45,:80]
        for name,edges,levels in [('dark_floor_patch',[],[0]),('riser_shadow',[.35,.65],[0,-.2,-.4])]:
            d,v,_=scene(edges,levels);mask=(xx>22+3*np.sin(yy/8))&(xx<50+.1*yy)
            ir=np.where(mask,.1,.8)[None];m=ElevationMap()
            self.assertTrue(dark_regions(ir[0]).any())
            for t in np.arange(0,.81,.1):feed(m,d,v,ir,t)
            _,g,valid=self.record(name,m);self.assertFalse(g.any());self.assertGreater(valid.sum(),15)
            self.results[name]['false_gap_rate']=float(g.mean())
    def test_vertical_riser_not_gap(self):
        # A dark, fully finite vertical face with no visible tread must stay unknown.
        from dataclasses import replace
        rr=replace(R,size=80)
        q=np.array([np.cos(np.pi/4),0,np.sin(np.pi/4),0]);pos=np.array([1.,0,.8])
        ray=rays(K,q,45,80);depth=(-.5-pos[0])/ray[...,0]
        d=depth[None];v=(d>.15)&(d<5);ir=np.full_like(d,.8);ir[:,:,25:55]=.1
        points=pos+ray*depth[...,None];self.assertGreater(vertical_returns(points,v[0]).sum(),3000)
        m=ElevationMap(rules=rr)
        for t in np.arange(0,.81,.1):feed(m,d,v,ir,t,pos=pos,quat=q)
        _,g,valid=self.record('vertical_riser',m);self.assertFalse(g.any());self.assertFalse(valid.any())
    def test_h0_unknown(self):
        d,v,_=scene();v[:]=False;m=ElevationMap('h0');feed(m,d,v,np.ones_like(d)*.01,0)
        h,g=grid(m,0);self.assertFalse(h.any());self.assertFalse(g.any())
if __name__=='__main__':
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures);result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).resolve().parents[2]/'outputs/cpu_fixtures.json';out.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,seconds=time.perf_counter()-start,fixtures=Fixtures.results),indent=2));raise SystemExit(not result.wasSuccessful())
