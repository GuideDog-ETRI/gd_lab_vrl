from types import SimpleNamespace as NS
from dataclasses import replace
import torch
from test_terrain177 import terrain,head
from gd_lab.gast_nav import isaac_fusion177 as adapter
from gd_lab.gast_nav.occgrid_gate176 import OccupancyInput176
from gd_lab.gast_nav.oracle_occgrid176 import OccConfig176

def test_live_adapter_contact_missing_and_stream_health(monkeypatch):
 t=terrain(n=2);h=head(n=2,support=True)
 env=NS(_terrain177=t,scene=NS(env_origins=torch.zeros(2,3)))
 nav=NS(robot=NS(body_names=['foot']),foot_ids=[0],control=NS(commands=NS(commands=[NS(requested_mode='NAV'),NS(requested_mode='NAV')])),index=NS(lookup=lambda ids:[None,None]))
 producer=NS(env=env,nav=nav,frame=h,ocfg=OccConfig176(),opts={'obstacle_ttl_s':.5},stats={},last_query169={})
 points=torch.zeros(2,1,3);valid=torch.tensor([[True],[False]])
 monkeypatch.setattr(adapter,'actual_contacts174',lambda *args:(points,valid))
 import gd_lab.gast_nav.command_generator as cg
 monkeypatch.setattr(cg,'route_hint',lambda *args:(NS(segment_type='FLAT'),None))
 obs=OccupancyInput176(h.state,torch.zeros_like(h.state,dtype=torch.float32),h.support.reference_z,torch.zeros_like(h.state,dtype=torch.float32),h.epoch)
 out=adapter.fuse_query177(producer,obs,torch.zeros(2,4),torch.full((2,),1000),h.epoch)
 assert out.state.dtype==torch.uint8 and (out.state[0]==1).any() and not (out.state[1]==1).any()
 assert (h.state==1).all() and (obs.state==1).all() # no feedback into raw memory
 env._terrain177=replace(t,last_capture_ms=torch.full((2,),600))
 stale=adapter.fuse_query177(producer,obs,torch.zeros(2,4),torch.full((2,),1000),h.epoch)
 assert not stale.state.any()
