"""CPU-only sampler/validity/isolation regression checks; no Isaac imports."""
import unittest
import torch
from gd_lab.gast.mapper_aug import MapperAugCfg, MapperAugmentation


def fixture(n=8, kind='gap'):
    x,y=torch.meshgrid(torch.linspace(-.8,.8,17),torch.linspace(-.5,.5,11),indexing='xy')
    xy=torch.stack((x.flatten(),y.flatten()),-1).repeat(n,1,1)
    def ground(x):
        if kind=='gap':return torch.where((x>.075)&(x<.225),-.65,0.)
        return torch.where(x>.075,.2 if kind=='up' else -.2,0.)
    z=ground(xy[...,0]);pts=torch.cat((xy,z[...,None]),-1)
    h=(.55-z-.5)*5;v=torch.ones_like(h,dtype=torch.bool);ir=torch.full_like(h,.35)
    def cast(starts,dirs,maxdist):return starts[:,2]-ground(starts[:,0])
    args=(pts,torch.full((n,),.55),torch.zeros(n),torch.zeros(n),torch.full((n,),kind=='gap',dtype=torch.bool))
    return h,v,ir,args,cast


class TestMapperAug(unittest.TestCase):
    def test_disabled_and_ramp_zero(self):
        h,v,ir,args,cast=fixture();state=torch.get_rng_state().clone()
        a=MapperAugmentation(8,'cpu');out=a(h,v,ir,*args,50000,cast)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for x,y in zip(out,(h,v,ir)):self.assertTrue(torch.equal(x,y))
        a=MapperAugmentation(8,'cpu',MapperAugCfg(enabled=True))
        for x,y in zip(a(h,v,ir,*args,0,cast),(h,v,ir)):self.assertTrue(torch.equal(x,y))

    def test_active_corruption_and_clean_episode(self):
        h,v,ir,args,cast=fixture();original=h.clone();pts=args[0].clone()
        a=MapperAugmentation(8,'cpu',MapperAugCfg(enabled=True));a.params[:]=.2
        a.params[:,0]=.9;a.params[0,0]=0.;a.params[:,1]=0.
        a.params[:,6:8]=.5;a.params[:,8:10]=.9;a.params[:,12]=0.
        out=a(h,v,ir,*args,50000,cast)
        self.assertTrue(torch.equal(out[0][0],h[0]));self.assertTrue(torch.equal(out[1][0],v[0]))
        self.assertTrue((out[0]!=h).any());self.assertTrue((~out[1]).any())
        self.assertTrue((out[2][~out[1]]==0).all());self.assertTrue((out[0][~out[1]]==0).all())
        self.assertTrue(torch.equal(h,original));self.assertTrue(torch.equal(args[0],pts))
        self.assertTrue(torch.isfinite(out[0]).all())
        # Pit replacement is exactly support -0.7 before existing observation noise.
        self.assertTrue(torch.isclose(out[0],torch.tensor(3.75)).any())

    def test_stairs_no_pit_floor_and_reset(self):
        for kind in ('up','down'):
            h,v,ir,args,cast=fixture(kind=kind)
            a=MapperAugmentation(8,'cpu',MapperAugCfg(enabled=True));a.params[:]=.2;a.params[:,0]=.9;a.params[:,1]=0
            out=a(h,v,ir,*args,50000,cast)
            self.assertLess(float(out[0].max()),2.)
            keep=a.params[1:].clone();a.reset(torch.tensor([0]));self.assertTrue(torch.equal(a.params[1:],keep))

    def test_reproducible_and_no_global_rng_consumption(self):
        h,v,ir,args,cast=fixture();a=MapperAugmentation(8,'cpu',MapperAugCfg(enabled=True));b=MapperAugmentation(8,'cpu',MapperAugCfg(enabled=True))
        state=torch.get_rng_state().clone()
        for x,y in zip(a(h,v,ir,*args,50000,cast),b(h,v,ir,*args,50000,cast)):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
