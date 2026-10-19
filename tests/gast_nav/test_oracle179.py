"""D-5 first-hit counterexamples with the ACTIVE >.16 m low-object rule."""
import torch
from dataclasses import replace
from gd_lab.gast_nav.oracle_occgrid176 import OccConfig176,OracleRays176,Support176,oracle_first_hits176,rasterize176
from gd_lab.gast_nav.producer168 import SensorPrimitives


def example(z,boxes,covered=True):
 c=OccConfig176(x_min_m=-1,x_max_m=2,y_min_m=-1,y_max_m=1)
 origins=torch.tensor([[[-.8,0.,z]]]);directions=torch.tensor([[[1.,0.,0.]]])
 rays=OracleRays176(origins,directions,torch.zeros(1,1),torch.full((1,1),3.),torch.full((1,1),covered,dtype=torch.bool),torch.full((1,1),2,dtype=torch.long))
 n=len(boxes);scene=SensorPrimitives(torch.tensor([[b[0] for b in boxes]],dtype=torch.float32),torch.eye(3)[None,None].expand(1,n,3,3).clone(),torch.tensor([[b[1] for b in boxes]],dtype=torch.float32),torch.zeros(1,n,dtype=torch.long),torch.ones(1,n,dtype=torch.bool),torch.ones(1,dtype=torch.bool))
 t=torch.zeros(1,dtype=torch.long);h=oracle_first_hits176(scene,rays,scene_capture_id=t,ray_capture_id=t)
 shape=(1,c.ny,c.nx);support=Support176(torch.zeros(shape),torch.ones(shape,dtype=torch.bool),torch.zeros(shape,dtype=torch.long))
 f=rasterize176(h,support,c,capture_ms=t,epoch=t,obstacle_min_height_m=torch.tensor([.16]))
 return c,f


def state(c,f,x):
 return f.state[0,round((0-c.y_min_m)/c.cell_m),int((x-c.x_min_m)/c.cell_m+1e-5)]


def test_d5_thin_pillar_observed_above_low_object_limit_blocks_and_occludes():
 c,f=example(.20,[([.51,0,.2],[.01,.01,.2])])
 assert state(c,f,.505)==2 and state(c,f,1.)==0


def test_d5_low_hit_does_not_extend_ray_behind_object():
 c,f=example(.15,[([.51,0,.075],[.01,.1,.075])])
 assert state(c,f,.505)!=2 and state(c,f,1.)==0


def test_d5_wall_hides_person_and_coverage_invalid_is_unknown():
 boxes=[([0.,0.,.4],[.02,.5,.4]),([.8,0.,.4],[.2,.2,.4])]
 c,f=example(.2,boxes);assert state(c,f,-.02)==2 and state(c,f,.8)==0
 _,f=example(.2,boxes,False);assert not f.state.any()


def test_d5_above_016_hit_remains_occupied_in_any_route_segment():
 c,f=example(.17,[([.51,0,.1],[.01,.1,.1])]);assert state(c,f,.505)==2
