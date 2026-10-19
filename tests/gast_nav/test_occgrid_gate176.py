from dataclasses import replace
import torch
from gd_lab.gast_nav.map_types import Grid
from gd_lab.gast_nav.swept174 import FutureSweep174
from gd_lab.gast_nav.occgrid_gate176 import OccupancyInput176,certify_stopping_grid176


def case():
 g=Grid(11,11,.1,-.5,-.5)
 state=torch.ones(2,11,11,dtype=torch.uint8);zero=torch.zeros_like(state,dtype=torch.float32)
 o=OccupancyInput176(state,zero.clone(),zero.clone(),zero.clone(),torch.zeros(2,dtype=torch.long))
 s=FutureSweep174(torch.zeros(2,1,3),torch.full((2,1,3),.04),torch.ones(2,1,dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.zeros(2,1,2),torch.ones(2,1,dtype=torch.bool))
 return g,o,s


def run(g,o,s,margin=0.):return certify_stopping_grid176(o,s,g,epoch=torch.zeros(2,dtype=torch.long),margin_m=margin)


def test_all_free_and_margin_intersects_whole_cells():
 g,o,s=case();assert run(g,o,s)[0].all()
 o.state[:,5,6]=2
 assert run(g,o,s)[0].all()
 ok,d=run(g,o,s,.02);assert not ok.any() and d['occupied'].all()


def test_unknown_support_and_stale_are_not_free():
 for field,value in [('state',0),('support',float('nan')),('age_s',.501),('support_age_s',.501)]:
  g,o,s=case();getattr(o,field)[:,5,5]=value;assert not run(g,o,s)[0].any()
 g,o,s=case();o.age_s.fill_(.5);assert run(g,o,s)[0].all()


def test_unmeasured_sweep_epoch_and_map_exit_close():
 g,o,s=case();assert not run(g,o,None)[0].any()
 assert not run(g,o,replace(s,qualified=torch.zeros(2,dtype=torch.bool)))[0].any()
 assert not run(g,replace(o,epoch=torch.ones(2,dtype=torch.long)),s)[0].any()
 s.centres[...,0]=.54;assert not run(g,o,s)[0].any()


def test_disjoint_parts_no_global_hull_and_no_body_exclusion():
 g,o,s=case();c=torch.zeros(2,2,3);c[:,0,0]=-.3;c[:,1,0]=.3
 s=replace(s,centres=c,half=torch.full_like(c,.02),required=torch.ones(2,2,dtype=torch.bool))
 o.state[:,5,5]=0;assert run(g,o,s)[0].all()
 o.state[:,5,2]=0;assert not run(g,o,s)[0].any()


def test_obstacle_and_support_ttls_are_independent():
 g,o,s=case();o.age_s.fill_(1.);o.support_age_s.fill_(.4)
 ok,_=certify_stopping_grid176(o,s,g,epoch=o.epoch,margin_m=0,ttl_s=2.,support_ttl_s=.5)
 assert ok.all()
 o.support_age_s.fill_(.501)
 ok,_=certify_stopping_grid176(o,s,g,epoch=o.epoch,margin_m=0,ttl_s=2.,support_ttl_s=.5)
 assert not ok.any()


def test_live_adapter_alignment_preserves_separate_timestamps_and_epoch():
 from types import SimpleNamespace
 from gd_lab.gast_nav.isaac_occgrid176 import IsaacOccGrid176
 from gd_lab.gast_nav.oracle_occgrid176 import OccFrame176,Support176
 from gd_lab.gast_nav.map_types import WIDE_GRID
 from gd_lab.gast_nav.map_config import load_config
 adapter=object.__new__(IsaacOccGrid176)
 adapter.opts={'obstacle_ttl_s':2.,'support_ttl_s':.5}
 adapter.nav=SimpleNamespace(pipeline=SimpleNamespace(config=load_config()))
 adapter.grid=WIDE_GRID;adapter.frame_pose=torch.zeros(1,4)
 shape=(1,WIDE_GRID.ny,WIDE_GRID.nx)
 state=torch.ones(shape,dtype=torch.uint8);time=torch.zeros(shape,dtype=torch.long)
 support=Support176(torch.zeros(shape),torch.ones(shape,dtype=torch.bool),time.clone())
 adapter.frame=OccFrame176(state,time,support,torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long))
 obs,grid=adapter.query_occupancy176(pose=adapter.frame_pose,time_ms=torch.tensor([1000]),epoch=torch.tensor([0]))
 assert (obs.age_s==1.).all() and (obs.support_age_s==1.).all()
 assert adapter.frame.seen_ms.max()==0 and adapter.frame.support.seen_ms.max()==0
 expired,_=adapter.query_occupancy176(pose=adapter.frame_pose,time_ms=torch.tensor([1000]),epoch=torch.tensor([1]))
 assert not expired.state.any() and not torch.isfinite(expired.support).any()


def test_missing_history_age_never_overflows_timestamp():
 from gd_lab.gast_nav.isaac_occgrid176 import original_timestamp176
 ages=torch.tensor([[[float('inf'),float('nan'),-.1,.4,.1]]])
 assert original_timestamp176(torch.tensor([300]),ages).tolist()==[[[-1,-1,-1,-1,200]]]
