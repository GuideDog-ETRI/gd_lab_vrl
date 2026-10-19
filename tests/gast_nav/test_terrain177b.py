from dataclasses import replace
import torch,pytest
from test_terrain177 import terrain,head,cell,fuse
from gd_lab.gast_nav.terrain_fusion177 import FusionConfig177
from gd_lab.gast_nav.oracle_occgrid176 import OccConfig176
from gd_lab.gast_nav.gap177b import walkable_gap177b
from gd_lab.gast_nav.occgrid_gate176 import OccupancyInput176,certify_stopping_grid176
from gd_lab.gast_nav.map_types import WIDE_GRID
from gd_lab.gast_nav.swept174 import FutureSweep174
CFG=FusionConfig177(terrain_limits177b=True)

def gap_scene(width):
 t=terrain();t.gap[:,:,8:8+width]=True;t.source[:,:,8:8+width]=4;t.surface_z[:,:,8:8+width]=-.7
 return t

@pytest.mark.parametrize('cells,allowed',[(2,True),(3,False)])
def test_gap_width_and_no_synthetic_floor(cells,allowed):
 t=gap_scene(cells);r=fuse(head(),t,config=CFG);yx=cell(OccConfig176(),.1,0)
 assert bool(r.walkable_gap[0,*yx])==allowed
 assert not r.foot_support.observed[0,*yx]
 assert (int(r.frame.state[0,*yx])==1)==allowed
 if allowed:
  f=r.frame;obs=OccupancyInput176(f.state,torch.zeros_like(f.state,dtype=torch.float),f.support.reference_z,(1000-f.support.seen_ms).float()/1000,f.epoch,r.walkable_gap,(1000-r.walkable_gap_seen_ms).float()/1000)
  sweep=FutureSweep174(torch.tensor([[[.1,0.,0.]]]),torch.tensor([[[.02,.02,.1]]]),torch.ones(1,1,dtype=torch.bool),torch.ones(1,dtype=torch.bool),torch.zeros(1,0,2),torch.zeros(1,0,dtype=torch.bool))
  ok,_=certify_stopping_grid176(obs,sweep,WIDE_GRID,epoch=f.epoch,margin_m=0.)
  assert ok.item()
  ok,_=certify_stopping_grid176(replace(obs,walkable_gap=None),sweep,WIDE_GRID,epoch=f.epoch,margin_m=0.)
  assert not ok.item()

def test_gap_far_bank_unknown_and_unknown_obstacle_block():
 t=gap_scene(2);t.valid[:,:,10:]=False;t.source[:,:,10:]=0
 assert not fuse(head(),t,config=CFG).walkable_gap.any()
 r=fuse(head(state=0),gap_scene(2),config=CFG)
 assert not (r.frame.state==1).any()

def test_turn_and_stale_gap_do_not_get_exception():
 assert not fuse(head(),gap_scene(2),config=CFG,gap_axis_enabled=torch.tensor([False])).walkable_gap.any()
 t=gap_scene(2);t.seen_ms[t.gap]=499
 assert not fuse(head(),t,config=CFG).walkable_gap.any()

@pytest.mark.parametrize('height,segment,passable',[(.12,'FLAT',True),(-.12,'FLAT',True),(.16,'FLAT',True),(.20,'FLAT',False),(.22,'STAIR_UP',True),(.26,'STAIR_UP',False),(.15,'FLAT',True),(.30,'FLAT',False)])
def test_steps_and_boxes(height,segment,passable):
 t=terrain();t.surface_z[:,:,10:]=height
 r=fuse(head(),t,config=CFG,segment=segment);yx=cell(OccConfig176(),.4,0.)
 assert bool(r.foot_support.observed[0,*yx])==passable
 if height>0:assert (int(r.frame.state[0,*yx])==2)==(not passable)

def test_exact_gap_boundary_and_no_edge():
 g=torch.zeros(1,1,40,dtype=torch.bool);g[:,:,5:31]=True;s=~g;t=torch.full(g.shape,1000)
 assert walkable_gap177b(g,s,t,cell_m=.01)[0].sum()==26
 g[:,:,31]=True;s=~g
 assert not walkable_gap177b(g,s,t,cell_m=.01)[0].any()


def test_observed_roughness_and_unknown_not_filled():
 t=terrain();yy,xx=torch.meshgrid(torch.arange(11),torch.arange(17),indexing='ij');t.surface_z[0]=((xx+yy)%2)*.09
 r=fuse(head(),t,config=CFG);yx=cell(OccConfig176(),0.,0.)
 assert r.foot_support.observed[0,*yx]
 t.valid[0,5,8]=False
 r=fuse(head(),t,config=CFG)
 assert not r.foot_support.observed[0,*yx]

@pytest.mark.parametrize('height,limit,occupied',[(.15,.16,False),(.16,.16,False),(.20,.16,True),(.30,.16,True),(.22,.25,False),(.25,.25,False),(.26,.25,True)])
def test_head_first_hit_uses_same_walkability_limit(height,limit,occupied):
 from gd_lab.gast_nav.oracle_occgrid176 import OracleRays176,OracleReturns176,rasterize176
 cfg=OccConfig176();h=head(support=True)
 rays=OracleRays176(torch.tensor([[[0.,0.,height]]]),torch.tensor([[[1.,0.,0.]]]),torch.zeros(1,1),torch.ones(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long))
 returns=OracleReturns176(rays,torch.full((1,1),.4),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.bool))
 f=rasterize176(returns,h.support,cfg,capture_ms=h.capture_ms,epoch=h.epoch,obstacle_min_height_m=torch.tensor([limit]))
 yx=cell(cfg,.4,0.)
 assert int(f.state[0,*yx])==(2 if occupied else 1)
 # Head ray still terminates at the first low hit; no free behind it.
 assert int(f.state[0,*cell(cfg,.6,0.)])==0
