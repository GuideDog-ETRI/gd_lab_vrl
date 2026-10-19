from dataclasses import replace
import torch,pytest
from test_terrain177 import terrain,head,cell,fuse
from test_terrain177b import gap_scene
from gd_lab.gast_nav.terrain_fusion177 import FusionConfig177
from gd_lab.gast_nav.oracle_occgrid176 import OccConfig176
from gd_lab.gast_nav.sweep179 import MeasuredSweep179
CFG=FusionConfig177(terrain_limits177b=True,terrain_free180=True)

def test_terrain_free180_unknown_head_and_outside_roi():
 h=head(state=0);t=terrain();r=fuse(h,t,config=CFG);g=OccConfig176()
 assert r.frame.state[0,*cell(g,0,0)]==1
 assert r.frame.state[0,*cell(g,1.,0)]==0
 assert not h.state.any()
 assert r.frame.seen_ms[0,*cell(g,0,0)]==1000

@pytest.mark.parametrize('failure',['invalid','pending','stale','epoch'])
def test_terrain_free180_no_unobserved_or_stale_grant(failure):
 t=terrain()
 if failure=='invalid':t.valid[:]=False
 if failure=='pending':t.pending[:]=True
 if failure=='stale':t.seen_ms[:]=499
 if failure=='epoch':t=replace(t,epoch=t.epoch+1)
 h=head(state=0)
 # fuse helper uses terrain epoch; deliberately preserve head epoch by direct call for epoch test.
 if failure=='epoch':
  from gd_lab.gast_nav.terrain_fusion177 import fuse_terrain177
  r=fuse_terrain177(h,t,t.pose,torch.tensor([1000]),h.epoch,config=CFG,measured_current_support=torch.zeros(1))
 else:r=fuse(h,t,config=CFG)
 assert not (r.frame.state==1).any()

def test_terrain_free180_occupied_priority_and_gap():
 r=fuse(head(state=2),terrain(),config=CFG)
 assert (r.frame.state==2).all()
 r=fuse(head(state=0),gap_scene(2),config=CFG)
 ix=cell(OccConfig176(),.1,0)
 assert r.walkable_gap[0,*ix] and r.frame.state[0,*ix]==1
 assert not r.foot_support.observed[0,*ix]
 r=fuse(head(state=0),gap_scene(3),config=CFG)
 assert r.frame.state[0,*ix]==0

def test_terrain_free180_scope_and_deployment():
 import yaml
 from pathlib import Path
 root=Path(__file__).resolve().parents[2]
 cfg=yaml.safe_load((root/'configs/gast_nav/pipeline180.yaml').read_text())
 model=MeasuredSweep179(cfg);assert not model.gate and model.calibration['trials']==2880
 cfg['scope']='deployment'
 with pytest.raises(ValueError,match='diagnostic/training'):MeasuredSweep179(cfg)
 cfg['sweep_qualification_gate']=True
 assert MeasuredSweep179(cfg).gate

def test_terrain_free180_known_limitation_unseen_overhang():
 # D-6 explicitly accepts this situation: no fabricated obstacle detection.
 # Same observed floor + unknown head cannot distinguish an unseen table top.
 r=fuse(head(state=0),terrain(),config=CFG)
 assert r.frame.state[0,*cell(OccConfig176(),.3,.3)]==1
