import torch
from gd_lab.gast_nav.support174 import actual_contacts174

def test_contact_uses_actual_point_not_link_center():
 from types import SimpleNamespace
 value=torch.tensor([[[[.1,.2,0.]]]])
 env=SimpleNamespace(scene={'ground_contact174_foot':SimpleNamespace(data=SimpleNamespace(contact_pos_w=value))})
 p,valid=actual_contacts174(env,['foot'],torch.zeros(1,4),lambda p,pose:p)
 assert valid.all() and p[0,0,2]==0

def test_unknown_wait_zero_before_hold():
 from test_env160 import tile,CFG,RUNTIME
 from gd_lab.gast_nav.control_runtime import ControlRuntime
 from gd_lab.gast_nav.landing_fsm import LandingObservation
 from gd_lab.gast_nav.supervisor import HealthInput
 r=ControlRuntime([tile('nav_corridor')],dict(CFG,supervisor_mode='enforce'),RUNTIME,seed=42);r.reset([0],now_ms=0,yaws=[0.])
 o=LandingObservation(True,False,False,False,True,0.,0.,False,False)
 for time in (10,100,1000,2990):
  x=r.before_policy(0,now_ms=time,dt=.01,xy=(0.,0.),yaw=0.,health=HealthInput(0,0,0),landing_observation=o,odom_epoch=0,legacy_velocity=(.5,0.,0.),unknown_wait=True)
  assert x.controller=='NAV_POLICY' and x.velocity==(0.,0.,0.) and not x.effective_go and x.reason=='UNKNOWN_WAIT'
 x=r.before_policy(0,now_ms=3000,dt=.01,xy=(0.,0.),yaw=0.,health=HealthInput(0,0,0,risk=True),landing_observation=o,odom_epoch=0,legacy_velocity=(.5,0.,0.),unknown_wait=True)
 assert x.controller=='HOLD_3906'

def test_historical_support_expiry_and_change():
 from test_free_evidence167 import cap,measurement,GRID,CFG
 from gd_lab.gast_nav.support174 import historical_support174
 from gd_lab.gast_nav.map_config import load_config
 from dataclasses import replace
 cfg=replace(CFG,pose_sigma=0.,pose_delay_s=0.,historical_obstacle_speed_bound_mps=0.,same_sample_pose_cancellation=True)
 old=cap(t=900);empty=cap(t=1000,support=torch.full((1,1,1),torch.nan))
 m=measurement([empty,old],settings=cfg)
 h,age=historical_support174(m,GRID,load_config());assert h.item()==0 and abs(age.item()-.1)<1e-6
 changed=cap(t=1000,support=torch.ones(1,1,1)*.2)
 h,age=historical_support174(measurement([changed,old],settings=cfg),GRID,load_config());assert abs(h.item()-.2)<1e-6 and age.item()==0
 h,_=historical_support174(measurement([old],now=1600,settings=cfg),GRID,load_config());assert not torch.isfinite(h).any()

def test_three_dimensional_sweep_and_unqualified_fail_closed():
 from test_free_evidence167 import cap,measurement,GRID,CFG,rays
 from gd_lab.gast_nav.swept174 import FutureSweep174,prove_future174
 from gd_lab.gast_nav.map_config import load_config
 from dataclasses import replace
 cfg=replace(CFG,pose_sigma=0.,pose_delay_s=0.,support_error_m=0.,sensor_angle_error_deg=0.,range_error_floor_m=0.,range_error_fraction=0.)
 m=measurement(settings=cfg);support=torch.zeros(1,1,1)
 sweep=FutureSweep174(torch.tensor([[[.5,0.,.5]]]),torch.full((1,1,3),.02),torch.ones(1,1,dtype=torch.bool),torch.ones(1,dtype=torch.bool),torch.tensor([[[.5,0.]]]),torch.ones(1,1,dtype=torch.bool))
 ok,_=prove_future174(sweep,m,rays(),support,GRID,load_config());assert ok.all()
 ok,_=prove_future174(replace(sweep,qualified=torch.zeros(1,dtype=torch.bool)),m,rays(),support,GRID,load_config());assert not ok.any()
 ok,_=prove_future174(replace(sweep,centres=torch.tensor([[[.5,0.,4.]]])),m,rays(),support,GRID,load_config());assert not ok.any()
 # Changed support must not reuse a capture's old reference.
 ok,_=prove_future174(sweep,m,rays(),support+.2,GRID,load_config());assert not ok.any()

def test_temporary_envelope_rejects_unmeasured_command_and_posture(tmp_path):
 from types import SimpleNamespace as NS
 from gd_lab.gast_nav.envelope174 import DiagnosticEnvelope174
 from gd_lab.gast_nav.map_types import Grid
 d=NS(body_pos_w=torch.zeros(1,1,3),body_quat_w=torch.tensor([[[1.,0.,0.,0.]]]),root_lin_vel_b=torch.zeros(1,3),root_ang_vel_b=torch.zeros(1,3))
 posture=torch.zeros(1,1,3)
 model=dict(centres=torch.tensor([[0.,0.,.10]]),half=torch.full((1,3),.05),postures=posture,body_names=['foot'],part_names=['foot'],speed_max=.2,omega_max=.2,qualified_geometry=True,posture_format='collision_corners_sphere_centres')
 path=tmp_path/'models.pt';torch.save({'nav_corridor:NAV:0.2':model},path)
 cmd=NS(requested_mode='NAV',v_cap=.2);tile=NS(kind='nav_corridor')
 nav=NS(footprint=NS(parts=[(0,'sphere',torch.zeros(3),torch.eye(3),torch.tensor(.03))]),robot=NS(data=d,body_names=['foot']),pose=lambda:torch.zeros(1,4),env=NS(scene=NS(env_origins=torch.zeros(1,3))),index=NS(lookup=lambda _: [tile]),control=NS(commands=NS(commands=[cmd])))
 e=DiagnosticEnvelope174(path,'cpu');grid=Grid(5,5,.1,-.2,-.2);support=torch.zeros(1,5,5)
 assert e.build(nav,support,grid).qualified.all()
 # Rotating a spherical foot must not count as a different collision posture.
 d.body_quat_w[:]=torch.tensor([2**-.5,0.,0.,2**-.5])
 assert e.build(nav,support,grid).qualified.all()
 d.body_quat_w[:]=torch.tensor([1.,0.,0.,0.])
 cmd.v_cap=.25;assert not e.build(nav,support,grid).qualified.any()
 cmd.v_cap=.2;d.body_pos_w+=.1;assert not e.build(nav,support,grid).qualified.any()
 d.body_pos_w.zero_();assert not e.build(nav,support*torch.nan,grid).qualified.any()

def test_actual_contact_force_window_is_required():
 from types import SimpleNamespace as NS
 data=NS(contact_pos_w=torch.zeros(1,1,1,3),net_forces_w_history=torch.ones(1,6,1,3)*10)
 env=NS(scene={'ground_contact174_foot':NS(data=data)},_contact174_force_gate=True,_contact_weight174=torch.tensor([100.]))
 _,ok=actual_contacts174(env,['foot'],torch.zeros(1,4),lambda x,p:x);assert ok.all()
 data.net_forces_w_history[:,0]=0
 _,ok=actual_contacts174(env,['foot'],torch.zeros(1,4),lambda x,p:x);assert not ok.any()
