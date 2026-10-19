from dataclasses import replace
import pytest
import torch
from gd_lab.gast_nav.front_depth183 import optical_returns183,merge_camera_frame183,virtual_entry183
from gd_lab.gast_nav.oracle_occgrid176 import OccConfig176,Support176,OccFrame176,rasterize176
from gd_lab.gast_nav.temporal_map.memory import load_settings,validate_settings,TemporalMap


def rays(distance,*,length=1.,source=5,valid=True,self_hit=False):
 d=torch.tensor([[float(distance)]])
 return optical_returns183(torch.zeros(1,1,3),torch.tensor([[[1.,0.,0.]]]),d,pixel_length=torch.tensor([length]),near_z=.15,far_z=5.,measured=torch.full((1,1),valid),self_hit=torch.full((1,1),self_hit),source=source)


def test_optical_metric_near_far_and_invalid():
 r=rays(.2,length=2.)
 assert r.rays.near_m.item()==pytest.approx(.3) and not r.valid.item()
 assert r.rays.far_m.item()==10
 assert not rays(float('inf'),valid=False).valid.item()
 assert rays(float('inf')).valid.item()
 assert not rays(.5,self_hit=True).valid.item()
 assert not rays(float('nan')).valid.item()
 with pytest.raises(ValueError):rays(1.,source=3)


def test_front_camera_explicit_opt_in_and_stop():
 cfg=OccConfig176(x_min_m=0.,x_max_m=1.,y_min_m=-.1,y_max_m=.1)
 z=torch.full((1,2,10),-.3);t=torch.zeros(1,dtype=torch.long)
 sup=Support176(z,torch.ones_like(z,dtype=torch.bool),torch.zeros_like(z,dtype=torch.long))
 with pytest.raises(ValueError):rasterize176(rays(.45),sup,cfg,capture_ms=t,epoch=t)
 f=rasterize176(rays(.45),sup,replace(cfg,front_depth183=True),capture_ms=t,epoch=t,obstacle_min_height_m=torch.tensor([.16]))
 assert f.state[0,1,4]==2
 assert f.state[0,1,0]==0 # optical prefix
 assert (f.state[:,:,5:]==0).all() # no carving behind first hit
 f=rasterize176(rays(.1),sup,replace(cfg,front_depth183=True),capture_ms=t,epoch=t)
 assert (f.state==0).all() # blocker before near cannot reveal floor behind


def frame(state,height=0.,time=100):
 st=torch.tensor([[[state]]],dtype=torch.uint8);tt=torch.tensor([time]);seen=tt[:,None,None].clone()
 return OccFrame176(st,seen,Support176(torch.full((1,1,1),height),torch.ones((1,1,1),dtype=torch.bool),seen),tt*0,tt)


def test_head_occupied_priority_both_orders_and_support_conflict():
 a,b=frame(2),frame(1)
 assert merge_camera_frame183(a,b).state.item()==2
 assert merge_camera_frame183(b,a).state.item()==2
 assert merge_camera_frame183(b,frame(1,.5)).state.item()==0
 with pytest.raises(ValueError):merge_camera_frame183(a,frame(1,time=101))


def test_support_ttl_is_scoped_and_does_not_refresh():
 s=load_settings();s['support_ttl_s']=2.
 with pytest.raises(ValueError):validate_settings(s)
 s['static_oracle183']=True;validate_settings(s)
 s['context']='deployment'
 with pytest.raises(ValueError):validate_settings(s)
 s['context']='training_static';s['memory_free_ttl_s']=2.;s['world_cells']=[3,3]
 cfg=OccConfig176(x_min_m=-.05,x_max_m=.05,y_min_m=-.05,y_max_m=.05)
 m=TemporalMap(1,s,output_cfg=cfg)
 p=torch.zeros(1,4);f=frame(1)
 m.push(f,p,torch.ones_like(f.state),profile_hash='a'*64,occluded_mask=torch.zeros_like(f.state,dtype=torch.bool))
 assert m.query(p,torch.tensor([1600]),torch.tensor([0])).foot_support.observed.item()
 assert not m.query(p,torch.tensor([2101]),torch.tensor([0])).foot_support.observed.item()
 assert m.last_ms.item()==100


def test_virtual_entry_speed_and_axis_original_z_yaw():
 pose=torch.tensor([[1.,2.,.6,.7],[3.,4.,.7,-.8]])
 samples=list(virtual_entry183(pose,torch.tensor([0.,torch.pi/2]),torch.tensor([.4,.6]),torch.tensor([10000,10000])))
 assert len(samples)==10
 p,t=samples[0]
 assert torch.allclose(p[:,:2],torch.tensor([[-1.,2.],[3.,2.]]))
 assert torch.equal(p[:,2:],pose[:,2:]) and t.tolist()==[5000,6667]
 assert samples[-1][1].tolist()==[9500,9667]


def test_raw_tap_preserves_all_front_pixels_and_no_mutation():
 from gd_lab.gast_nav.front_depth183 import packet_returns183
 p=dict(camera_indices=(0,1),resolution=(1,2),origins=torch.zeros(1,2,3),
  dirs=torch.tensor([1.,0.,0.]).expand(1,2,1,2,3).clone(),ranges=torch.ones(1,2,1,2),
  pixel_length=torch.ones(1,2,1,2),measured=torch.ones(1,2,1,2,dtype=torch.bool),
  self_hit=torch.zeros(1,2,1,2,dtype=torch.bool),near_z=.15,far_z=5.)
 before=p['ranges'].clone();out=list(packet_returns183(p,torch.zeros(1,4)))
 assert len(out)==2 and out[0].distance_m.shape==(1,2)
 assert out[0].rays.source.unique().tolist()==[5] and out[1].rays.source.unique().tolist()==[6]
 assert torch.equal(before,p['ranges'])
 p['camera_indices']=(2,3)
 with pytest.raises(ValueError):list(packet_returns183(p,torch.zeros(1,4)))


def test_camera_merge_does_not_renew_retained_occupied():
 a=frame(2);a.seen_ms.fill_(40)
 result=merge_camera_frame183(a,frame(1))
 assert result.state.item()==2 and result.seen_ms.item()==40


def test_memory_fills_only_holes_and_retained_occupied_wins():
 from gd_lab.gast_nav.front_depth183 import supplement_memory183
 from gd_lab.gast_nav.occgrid_gate176 import OccupancyInput176
 state=torch.tensor([[[1,1,0,1,0]]],dtype=torch.uint8);age=torch.zeros(1,1,5)
 support=torch.tensor([[[0.,0.,float('nan'),0.,0.]]]);epoch=torch.tensor([1])
 current=OccupancyInput176(state,age,support,age,epoch)
 mem=OccupancyInput176(torch.tensor([[[0,2,1,1,1]]],dtype=torch.uint8),age+1.,torch.tensor([[[float('nan'),0.,0.,.5,.5]]]),age+1.,epoch)
 out=supplement_memory183(current,mem,obstacle_ttl_s=2.,support_ttl_s=2.)
 assert out.state.tolist()==[[[1,2,1,1,0]]]
 assert out.age_s[0,0,0]==0 and out.age_s[0,0,2]==1
 assert out.support.tolist()==[[[0.,0.,0.,0.,0.]]]
 # Different epoch never contributes, even with a numerically fresh timestamp.
 other=supplement_memory183(current,replace(mem,epoch=epoch+1),obstacle_ttl_s=2.,support_ttl_s=2.)
 assert other.state.tolist()==state.tolist() and torch.isnan(other.support[0,0,2])
