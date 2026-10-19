import copy
from pathlib import Path
import torch
import yaml
from gd_lab.gast_nav.stair_end184 import StairEnd184
from gd_lab.gast_nav.control_runtime import ControlRuntime
from gd_lab.gast_nav.supervisor import HealthInput
from gd_lab.gast_nav.landing_fsm import LandingObservation
from test_env160 import CFG,RUNTIME,tile


def test_shadow_keeps_commands_but_not_safety_overrides():
 cfg=copy.deepcopy(CFG);cfg['supervisor_mode']='shadow';cfg['commands']['go_probability']=1.
 rt=copy.deepcopy(RUNTIME);rt['command']['omega_max_verified']=1.
 r=ControlRuntime([tile('nav_corridor')],cfg,rt,seed=42);r.reset([0],now_ms=0,yaws=[0.])
 o=LandingObservation(True,True,True,True,True,0.,0.,False,False)
 for t in range(1,6):
  out=r.before_policy(0,now_ms=t*10,dt=.01,xy=(0.,0.),yaw=0.,health=HealthInput(1000,1000,1000,risk=True),landing_observation=o,odom_epoch=0,legacy_velocity=(.5,0.,0.),unknown_wait=True)
 assert out.controller=='NAV_POLICY' and out.effective_go and out.velocity[0]>0
 assert out.phase=='APPROACH' # no GT end label enters actor phase
 out=r.before_policy(0,now_ms=60,dt=.01,xy=(0.,0.),yaw=0.,health=HealthInput(0,0,0,nav_invalid=True),landing_observation=o,odom_epoch=0,legacy_velocity=(0,0,0))
 assert out.controller=='HOLD_3906'


def detector():
 return StairEnd184(1,'cpu',yaml.safe_load((Path(__file__).parents[2]/'configs/gast_nav/shadow184.yaml').read_text()))

def tick(d,z,pitch=0.,stair=True,go=True,request=1,contacts=True):
 return d.tick(foot_z=torch.tensor([z],dtype=torch.float),contacts=torch.full((1,4),contacts),pitch=torch.tensor([pitch]),roll=torch.zeros(1),stair_surface=torch.tensor([stair]),nav=torch.ones(1,dtype=torch.bool),go=torch.tensor([go]),request_id=torch.tensor([request]),dt=.05)


def test_stair_end_latched_until_new_go_not_heartbeat():
 d=detector();tick(d,[0]*4);tick(d,[.1,.1,0,0],pitch=.12)
 for _ in range(5):tick(d,[.1]*4)
 assert d.wait.item() and d.total_events.item()==1
 for _ in range(8):tick(d,[.1]*4)
 assert d.wait.item() and d.total_events.item()==1
 tick(d,[.1]*4,request=2);assert not d.wait.item()
 d.reset([0]);assert not d.contact_seen.any()


def test_flat_swing_ramp_and_down_first_step_are_not_end():
 for stair in (False,True):
  d=detector()
  for k in range(20):tick(d,[.01*k,.01*k,.01*k-.08,.01*k-.08],pitch=.15,stair=stair)
  for k in range(1,10):tick(d,[.2,.2,.11+.01*k,.11+.01*k],pitch=.15*(1-k/9),stair=stair)
  for _ in range(5):tick(d,[.2]*4,stair=stair)
  # Without a >=4cm stance-height jump, constant ramp cannot arm.
  assert not d.wait.item()
 d=detector();tick(d,[0]*4);tick(d,[-.1,-.1,0,0],pitch=-.12)
 for _ in range(8):tick(d,[-.1,-.1,0,0],pitch=-.12)
 assert not d.wait.item()


def test_short_tread_needs_continuous_flat_but_wide_middle_landing_counts():
 d=detector();tick(d,[0]*4);tick(d,[.1,.1,0,0],pitch=.12)
 tick(d,[.1]*4);tick(d,[.1]*4,contacts=False)
 assert not d.wait.item()
 for _ in range(4):tick(d,[.1]*4)
 assert d.wait.item() # no final-floor vs landing distinction


def test_shadow_progress_does_not_require_certificate():
 from gd_lab.gast_nav.rewards import reward_terms
 z=torch.zeros(2);one=torch.ones(2);b=one.bool()
 kw=dict(mode=one.long(),go=b,phase=z.long(),feasible=~b,observed=~b,v_path=one*.2,v_cap=one*.5,speed=one*.2,yaw_rate=z,turn_rate=z,positive_yaw=z,clearance=one,margin=one*.1,offset_error=z,yaw_error=z,turn_drift=z,contact=z,entered_forbidden=z,cfg=CFG['rewards'])
 terms,_=reward_terms(**kw,supervisor_mode='shadow');torch.testing.assert_close(terms['progress'],one*.4)
 terms,_=reward_terms(**kw,supervisor_mode='enforce');assert not terms['progress'].any()


def tick_ahead(d,z,ahead,go=True,request=1):
 return d.tick(foot_z=torch.tensor([z],dtype=torch.float),contacts=torch.full((1,4),True),pitch=torch.zeros(1),roll=torch.zeros(1),stair_surface=torch.ones(1,dtype=torch.bool),nav=torch.ones(1,dtype=torch.bool),go=torch.tensor([go]),request_id=torch.tensor([request]),dt=.05,ahead_flat=torch.tensor([ahead]))


def test_ahead_flat_rule_stops_only_when_flat_ground_ahead():
 # 2026-10-19 user rule: after stair stepping, flat ground ahead -> stop (no pitch needed).
 d=detector();tick_ahead(d,[0]*4,False);tick_ahead(d,[.1,.1,0,0],False)
 for _ in range(6):tick_ahead(d,[.1]*4,False)   # wide mid-stair tread: next step visible ahead
 assert not d.wait.item()
 for _ in range(4):tick_ahead(d,[.1]*4,True)    # landing / final floor ahead
 assert d.wait.item() and d.total_events.item()==1
 tick_ahead(d,[.1]*4,True,request=2);assert not d.wait.item()


def test_ahead_flat_needs_prior_stair_stepping():
 d=detector()
 for _ in range(10):tick_ahead(d,[0]*4,True)    # flat start, or top of stairs before descending
 assert not d.wait.item()
