from dataclasses import replace
import torch
from gd_lab.gast_nav.map_types import Grid
from gd_lab.gast_nav.occgrid_gate176 import OccupancyInput176,certify_stopping_grid176
from gd_lab.gast_nav.swept174 import FutureSweep174
from gd_lab.gast_nav.pipeline179 import contact_support179
from gd_lab.gast_nav.sweep179 import measured_sweep179,MeasuredSweep179


def fixture():
 g=Grid(11,11,.1,-.5,-.5);state=torch.ones(1,11,11,dtype=torch.uint8);z=state.float()*0
 o=OccupancyInput176(state,z.clone(),z.clone(),z.clone(),torch.zeros(1,dtype=torch.long))
 c=torch.tensor([[[0.,0.,0.],[.3,0.,0.]]]);s=FutureSweep174(c,torch.full_like(c,.02),torch.ones(1,2,dtype=torch.bool),torch.ones(1,dtype=torch.bool),torch.zeros(1,0,2),torch.zeros(1,0,dtype=torch.bool))
 return g,o,s


def test_exclude_only_current_body_not_future_and_do_not_mutate_evidence():
 g,o,s=fixture();o.state[:,5,5]=0;body=torch.zeros_like(o.state,dtype=torch.bool);body[:,5,5]=True
 kw=dict(epoch=o.epoch,margin_m=0,current_body_mask=body)
 ok,d=certify_stopping_grid176(o,s,g,**kw);assert ok.all() and not d['required'][0,5,5] and d['required'][0,5,8]
 assert o.state[0,5,5]==0
 o.state[:,5,8]=0;assert not certify_stopping_grid176(o,s,g,**kw)[0].any()
 o.state[:,5,8]=2;assert not certify_stopping_grid176(o,s,g,**kw)[0].any()


def test_no_vacuous_or_unqualified_permission():
 g,o,s=fixture();body=torch.ones_like(o.state,dtype=torch.bool)
 assert not certify_stopping_grid176(o,s,g,epoch=o.epoch,margin_m=0,current_body_mask=body)[0].any()
 assert not certify_stopping_grid176(o,replace(s,qualified=torch.tensor([False])),g,epoch=o.epoch,margin_m=0)[0].any()


def test_contact_support_not_free_or_gap_fill_and_invalid_contacts_ignored():
 g,o,s=fixture();o.support[:]=torch.nan;o.support_age_s[:]=torch.inf;o.state[:]=0
 pts=torch.tensor([[[0.,0.,-.5],[.3,0.,-.2],[float('nan'),0.,0.]]]);valid=torch.tensor([[True,False,True]])
 result=contact_support179(o,pts,valid,g)
 assert result.support[0,5,5]==-.5 and result.support_age_s[0,5,5]==0
 assert torch.isfinite(result.support).sum()==1 and not result.state.any()
 assert not torch.isfinite(o.support).any()


def test_measured_velocity_and_next_swing_change_sweep():
 c=torch.zeros(1,2,3);h=torch.full_like(c,.03);v=torch.tensor([[.4,-.1]]);w=torch.tensor([-.3]);rel=torch.zeros_like(c)
 a,sa=measured_sweep179(c,h,v,w,rel)
 rel[:,1,0]=.5;b,sb=measured_sweep179(c,h,v,w,rel)
 assert sa['stop_distance'].item()>0 and sa['stop_angle'].item()<0
 assert (b.centres+b.half)[...,0].max()>(a.centres+a.half)[...,0].max()
 assert sb['next_swing_displacement'][0,1,0]>0


def test_missing_qualification_is_closed(tmp_path):
 model=MeasuredSweep179({'qualification_file':str(tmp_path/'absent.json')})
 assert model.domains==[]


def test_reward_certificate_independent_of_actor_observability():
 from gd_lab.gast_nav.rewards import reward_terms
 from gd_lab.gast_nav import schema as S
 n=2;z=torch.zeros(n);one=torch.ones(n);b=torch.ones(n,dtype=torch.bool)
 keys=('progress','overspeed','clearance','offset','stop_success','false_stop','forbidden_entry','turn_tracking','reverse_rate','reverse_accumulation','turn_drift','stair_alignment','dwell_motion','contact')
 kw=dict(mode=torch.tensor([S.MODES.index('NAV'),S.MODES.index('TURN')]),go=b,phase=torch.zeros(n,dtype=torch.long),feasible=b,observed=~b,supervisor_certified=b,v_path=one*.2,v_cap=one*.4,speed=one*.2,yaw_rate=one*-.2,turn_rate=one*-.3,positive_yaw=z,clearance=one,margin=one*.1,offset_error=z,yaw_error=z,turn_drift=z,contact=~b,entered_forbidden=~b,cfg={**{k:1. for k in keys},'stopped_speed_mps':.05})
 terms,_=reward_terms(**kw);assert terms['progress'][0]==.5 and terms['turn_tracking'][1]>0
 kw.update(observed=b,supervisor_certified=~b);terms,_=reward_terms(**kw)
 assert not terms['progress'].any() and not terms['turn_tracking'].any()


def test_contact_force_debounce_strict_threshold_and_duration():
 from gd_lab.gast_nav.pipeline179 import stable_contact179
 history=torch.zeros(2,11,3);history[...,2]=11.
 weight=torch.full((2,),100.)
 assert stable_contact179(history,weight,.005).all()
 assert not stable_contact179(history[:,:10],weight,.005).any()
 history[0,5,2]=10.;history[1,2,0]=float('nan')
 assert not stable_contact179(history,weight,.005).any()


def test_current_exclusion_requires_whole_cell_inside_one_primitive():
 from gd_lab.gast_nav.pipeline179 import current_body_cells179
 from gd_lab.gast_nav.body_projection import CollisionFootprint
 fp=object.__new__(CollisionFootprint)
 fp.parts=[(0,'sphere',torch.zeros(3),torch.eye(3),torch.tensor(.025))]
 g=Grid(3,3,.1,-.1,-.1);pos=torch.zeros(1,1,3);q=torch.tensor([[[1.,0.,0.,0.]]]);pose=torch.zeros(1,4)
 assert fp.mask(g.xy(device='cpu',dtype=torch.float32),pos,q,pose,.02)[0,1,1]
 assert not current_body_cells179(fp,g,pos,q,pose,.02).any()
 fp.parts=[(0,'sphere',torch.zeros(3),torch.eye(3),torch.tensor(.08))]
 mask=current_body_cells179(fp,g,pos,q,pose,.02)
 assert mask[0,1,1] and mask.sum()==1


def test_stale_qualification_settings_fail_closed(tmp_path):
 import json
 p=tmp_path/'audit.json';p.write_text(json.dumps({'settings':{'deceleration':.6},'domains':[{'qualified':True}]}))
 assert MeasuredSweep179({'deceleration':.5,'qualification_file':str(p)}).domains==[]
