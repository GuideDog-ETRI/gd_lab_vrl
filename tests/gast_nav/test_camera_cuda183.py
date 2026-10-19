import pytest
import torch
from dataclasses import replace
from gd_lab.gast_nav.oracle_occgrid176 import *
from gd_lab.gast_nav.producer168 import SensorPrimitives
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='local GPU0 only')


def test_bounded_cuda_first_hit_and_raster_reference():
 from gd_lab.gast_nav.camera_cuda183 import cast183,raster183
 dev='cuda';g=torch.Generator(device=dev).manual_seed(183)
 n,r,p=2,257,4
 c=torch.rand(n,p,3,device=dev,generator=g)*2-1;h=torch.full_like(c,.2)
 rot=torch.eye(3,device=dev)[None,None].expand(n,p,3,3).clone()
 kinds=torch.tensor([[0,0,2,2]]*n,device=dev);active=torch.ones(n,p,device=dev,dtype=torch.bool)
 scene=SensorPrimitives(c,rot,h,kinds,active,torch.ones(n,device=dev,dtype=torch.bool))
 o=torch.rand(n,r,3,device=dev,generator=g)*2-1;d=torch.rand(n,r,3,device=dev,generator=g)*2-1;d=d/d.norm(dim=-1,keepdim=True)
 rays=OracleRays176(o,d,torch.full((n,r),.15,device=dev),torch.full((n,r),5.,device=dev),torch.ones(n,r,device=dev,dtype=torch.bool),torch.full((n,r),4,device=dev,dtype=torch.long))
 ids=torch.zeros(n,device=dev,dtype=torch.long);selfmask=torch.zeros_like(active);selfmask[:,-1]=True
 ref=oracle_first_hits176(scene,rays,scene_capture_id=ids,ray_capture_id=ids,self_mask=selfmask)
 got=cast183(scene,rays,selfmask)
 assert torch.allclose(got.distance_m,ref.distance_m,atol=2e-5,rtol=1e-5)
 assert torch.equal(got.valid,ref.valid&~ref.self_hit)
 cfg=OccConfig176(front_depth183=True);height=torch.full((n,cfg.ny,cfg.nx),-.5,device=dev)
 height[:,3:5,6:8]=torch.nan
 support=Support176(height,torch.isfinite(height),torch.zeros_like(height,dtype=torch.long))
 a=rasterize176(got,support,cfg,capture_ms=ids,epoch=ids,obstacle_min_height_m=torch.full((n,),.16,device=dev))
 b=raster183(got,support,cfg,capture_ms=ids,epoch=ids)
 assert torch.equal(a.state,b.state)
 assert torch.equal(a.seen_ms,b.seen_ms)
