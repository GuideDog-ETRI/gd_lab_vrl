"""UPDATE160 only: no Isaac app, no GPU, no optimizer steps (enforced)."""
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from gd_lab.gast_nav import schema as S
from gd_lab.gast_nav.train_inertia import TrainInertia, ExternalForceAccumulator, sum_external_wrenches, limited_sine_amplitude
from gd_lab.gast_nav.command_generator import CommandGenerator, route_projection
from gd_lab.gast_nav.scene_stub import SceneTileMeta
from gd_lab.gast_nav.rewards import reward_terms, legacy_reward_masks, intervention_penalty
from gd_lab.gast_nav.obs_terms import (sensor_pattern, randomize_sensor_pattern, world_to_sample, mask_self_returns,
                                      fit_observed_surface, observed_support, ObservationPipeline, critic_privileged)
from gd_lab.gast_nav.map_types import Grid, RayBatch, DynamicCapsules
ROOT=Path(__file__).resolve().parents[2]
CFG=yaml.safe_load((ROOT/'configs/gast_nav/environment.yaml').read_text())
RUNTIME=yaml.safe_load((ROOT/'configs/gast_nav/runtime.yaml').read_text())

@pytest.fixture(autouse=True)
def prohibit_optimizer(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('UPDATE160 forbids every optimizer step')
    pending=[torch.optim.Optimizer];seen=set()
    while pending:
        cls=pending.pop()
        if cls in seen:continue
        seen.add(cls);pending.extend(cls.__subclasses__())
        monkeypatch.setattr(cls,'step',forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()

def test_accelerating_floor_point_mass_equivalence():
    # Integrate inertial coordinates and subtract the accelerated floor; compare
    # with relative coordinates under -m*a for multiple unequal payload masses.
    dtype=torch.float64;dt=.005
    mass=torch.tensor([[2.,8.,14.]],dtype=dtype)
    floor_x=torch.zeros(3,dtype=dtype);floor_v=floor_x.clone()
    x=torch.zeros(3,3,dtype=dtype);v=x.clone();relative=x.clone();relative_v=x.clone()
    applied=torch.tensor([[3.,-1.,0.],[1.,2.,0.],[-4.,1.,0.]],dtype=dtype)
    for k in range(1000):
        a=torch.tensor([math.sin(k*dt),.3*math.cos(k*dt),0.],dtype=dtype)
        floor_v+=a*dt;floor_x+=floor_v*dt
        v+=applied/mass[0,:,None]*dt;x+=v*dt
        force,_=sum_external_wrenches(mass,a[None],applied[None])
        relative_v+=force[0]/mass[0,:,None]*dt;relative+=relative_v*dt
    torch.testing.assert_close(x-floor_x,relative,atol=1e-11,rtol=1e-11)

def test_profile_jerk_sine_and_reset():
    model=TrainInertia(32,CFG['inertia']);model.reset(range(32),torch.ones(32,dtype=torch.bool))
    p=model.parameters
    assert bool((2*math.pi*p['frequency']*p['amp'] <= 10.000001).all())
    previous=model.acceleration.clone()
    for _ in range(1500):
        a=model.step(.005)
        assert float((a-previous).norm(dim=-1).max()) <= .050001
        previous=a.clone()
    model.reset([0,3],[False,False])
    assert not bool(model.acceleration[[0,3]].any())
    assert not bool(model.step(.005)[[0,3]].any())
    out=model.gravity_bias(torch.tensor([[0.,0.,-1.]]).repeat(32,1))
    torch.testing.assert_close(out.norm(dim=-1),torch.ones(32))

def test_sine_clamps_high_frequency():
    actual=limited_sine_amplitude(torch.tensor([1.5]),torch.tensor([2.]),10.)
    assert actual.item()==pytest.approx(10/(4*math.pi))

def test_single_force_writer_actual_mass_and_reset():
    mass=torch.tensor([[1.,7.]])
    calls=[]
    robot=SimpleNamespace(root_physx_view=SimpleNamespace(get_masses=lambda:mass),
                          set_external_force_and_torque=lambda f,t,**kw:calls.append((f.clone(),t.clone(),kw)))
    acc=ExternalForceAccumulator(1,2,device='cpu')
    handle=torch.tensor([[[3.,2.,1.],[0.,0.,0.]]]);torque=handle*.2
    acc.submit_handle(handle,torque)
    acc.apply(robot,torch.tensor([[1.,0.,0.]]))
    torch.testing.assert_close(calls[0][0],handle-torch.tensor([[[1.,0.,0.],[7.,0.,0.]]]))
    assert calls[0][2]=={'is_global':True}
    mass[0,1]=10.;acc.apply(robot,torch.tensor([[1.,0.,0.]]))
    assert calls[-1][0][0,1,0]==-10.
    assert acc.write_count==len(calls)==2
    acc.reset([0]);acc.apply(robot,torch.zeros(1,3))
    assert not bool(calls[-1][0].any()) and not bool(calls[-1][1].any())

def tile(kind):
    return SceneTileMeta(kind,0,((0.,0.),(10.,0.)),((0.,1.),),
                         (('TURN_SPOT' if kind=='nav_turn_room' else 'PASSAGE',0.,10.),),
                         None,None,(),'flat',kind=='nav_turn_room')

def test_command_episode_distribution_defaults_and_handshake():
    kinds=sum(([kind]*int(frac*10000) for kind,frac in CFG['scene_mix'].items()),[])
    gen=CommandGenerator([tile(k) for k in kinds],CFG,RUNTIME['command'],seed=42)
    gen.reset(range(len(kinds)),now_ms=0,yaws=[0.]*len(kinds))
    for task,expected in CFG['commands']['task_probabilities'].items():
        assert gen.task.count(task)/len(kinds)==pytest.approx(expected,abs=.015)
    defaults=sum(c.c.margin==.1 and c.c.y_off==0. and c.c.turn_cap==.6 for c in gen.commands)
    assert defaults/len(kinds)==pytest.approx(.3,abs=.015)
    idx=kinds.index('nav_turn_room')
    _,applied,features,_=gen.tick(idx,now_ms=10,dt=.01,xy=(0,0),stable=False)
    assert not applied['effective_go'] # switch handshake or conservative rejection
    assert len(features)==16
    for k in range(2,5):
        _,applied,_,_=gen.tick(idx,now_ms=k*10,dt=.01,xy=(0,0),stable=True)
    assert applied['mode']=='TURN'
    assert applied['hold_reason']=='TURN_UNVERIFIED' # never invent hardware certification

def test_command_condition_rate_and_stop_start_mode_switch():
    gen=CommandGenerator([tile('nav_corridor')],CFG,RUNTIME['command'],seed=4)
    gen.reset([0],now_ms=0,yaws=[0.]);gen.task[0]='stop_start'
    from gd_lab.gast_nav.commands import Conditions
    gen.target[0]=Conditions(.2,.2,1.)
    before=gen.commands[0].c
    cmd,_,_,_=gen.tick(0,now_ms=10,dt=.01,xy=(1,0),stable=True)
    for key in S.CONDITION_KEYS:
        assert abs(getattr(cmd.c,key)-getattr(before,key))<=CFG['commands']['condition_rate_per_s'][key]*.01+1e-9
    cmd,_,_,_=gen.tick(0,now_ms=5010,dt=5.,xy=(1,0),stable=True)
    assert cmd.requested_mode=='TURN'

def test_route_projection_frame():
    s,off,yaw,remaining=route_projection(((0,0),(0,2),(2,2)),(.2,1))
    assert (s,off,yaw,remaining)==pytest.approx((1.,-.2,math.pi/2,3.))

def reward_case(feasible,observed,speed=0.):
    n=len(feasible);z=torch.zeros(n);one=torch.ones(n);go=torch.ones(n,dtype=torch.bool)
    return reward_terms(mode=torch.full((n,),S.MODES.index('NAV')),go=go,phase=z.long(),
                        feasible=torch.tensor(feasible),observed=torch.tensor(observed),v_path=one*.2,v_cap=one*.5,
                        speed=z+speed,yaw_rate=z,turn_rate=z,positive_yaw=z,clearance=one,margin=one*.1,
                        offset_error=z,yaw_error=z,turn_drift=z,contact=z,entered_forbidden=one,cfg=CFG['rewards'])

def test_two_axis_reward_truth_table():
    terms,weighted=reward_case([True,True,False,False],[True,False,True,False])
    torch.testing.assert_close(terms['progress'],torch.tensor([.4,0.,0.,0.]))
    torch.testing.assert_close(terms['false_stop'],torch.tensor([1.,0.,0.,0.]))
    torch.testing.assert_close(terms['stop_success'],torch.tensor([0.,0.,1.,1.]))
    torch.testing.assert_close(weighted['forbidden_entry'],torch.tensor([0.,0.,-2.,-2.]))

def test_masks_and_single_intervention():
    phase=torch.tensor([S.PHASES.index(p) for p in ('APPROACH','PRE_BRAKE','LANDING_ENTERED','DWELL')])
    masks=legacy_reward_masks(torch.tensor([0,1,2,1]),phase,torch.ones(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool))
    assert masks['xy_tracking'].tolist()==[True,False,False,False]
    assert masks['stair_stall'].tolist()==[True,False,False,False]
    assert intervention_penalty(torch.tensor([1,1,0]),torch.tensor([0,1,1]),-1.).tolist()==[-1.,0.,0.]

def test_patterns_mount_jitter_and_yaw_transform():
    settings=CFG['sensors'];o,d=sensor_pattern(settings)
    half=len(d)//2
    assert bool((d[:half,0]>0).all()) and bool((d[half:,0]<0).all())
    sensor=SimpleNamespace(ray_starts=o[None].repeat(2,1,1),ray_directions=d[None].repeat(2,1,1))
    randomize_sensor_pattern(sensor,[1],settings,generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(sensor.ray_starts[0],o)
    assert float((sensor.ray_starts[1]-o).abs().max())<=settings['mount_jitter_m']+1e-7
    torch.testing.assert_close(sensor.ray_directions.norm(dim=-1),torch.ones(2,len(o)))
    pose=torch.tensor([[1.,2.,3.,math.pi/2]])
    torch.testing.assert_close(world_to_sample(torch.tensor([[[1.,3.,3.]]]),pose),torch.tensor([[[1.,0.,0.]]]),atol=1e-6,rtol=0)

def test_self_and_dynamic_first_hit():
    from gd_lab.gast_nav.ray_geometry import resolve_first_hits
    rays=RayBatch(torch.zeros(1,1,3),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[4.]]),torch.zeros(1,1,dtype=torch.bool))
    def capsule(x):return DynamicCapsules(torch.tensor([[[x,0.,-.5]]]),torch.tensor([[[x,0.,.5]]]),torch.tensor([[.1]]))
    masked=mask_self_returns(rays,capsule(1.))
    assert masked.self_hit.item() and masked.distances.item()==pytest.approx(.9)
    assert not resolve_first_hits(masked,capsules=capsule(2.)).occupied_hit.item()
    assert resolve_first_hits(masked,capsules=capsule(.5)).occupied_hit.item()

def test_observed_support_plane_and_missing_cells():
    grid=Grid(5,5,.1,-.2,-.2)
    xy=grid.xy(device='cpu',dtype=torch.float32).reshape(1,-1,2)
    points=torch.cat((xy,torch.full((1,25,1),-.4)),-1);valid=torch.ones(1,25,dtype=torch.bool)
    h,obs,normals=fit_observed_surface(points,valid,grid,CFG['sensors'])
    torch.testing.assert_close(normals[...,2],torch.ones_like(h))
    valid[0,12]=False
    support=observed_support(points,valid,points[:,:4],torch.ones(1,4,dtype=torch.bool),'FLAT',grid,CFG['sensors'])
    assert torch.isnan(support[0,2,2])
    assert torch.isfinite(support).sum()==24
    no_contact=observed_support(points,valid,points[:,:4],torch.zeros(1,4,dtype=torch.bool),'FLAT',grid,CFG['sensors'])
    assert torch.isnan(no_contact).all()

def test_pipeline_unknown_shapes_reset_and_privileged_separation():
    # Real UPDATE159 pipeline, empty observed scene: never fill floor from GT.
    pipe=ObservationPipeline(1,CFG['sensors'])
    rays=RayBatch(torch.zeros(1,1,3),torch.tensor([[[1.,0.,0.]]]),torch.full((1,1),torch.inf),torch.zeros(1,1,dtype=torch.bool))
    pose=torch.zeros(1,4)
    wide,near=pipe.capture(rays=rays,terrain_points=torch.zeros(1,1,3),terrain_valid=torch.zeros(1,1,dtype=torch.bool),
                          contact_points=torch.zeros(1,4,3),contact_valid=torch.zeros(1,4,dtype=torch.bool),segment='FLAT',pose=pose,time_ms=torch.tensor([0]))
    assert wide.shape==(1,27,43,49) and near.shape==(1,3,25,37)
    assert torch.isfinite(wide).all() and torch.isfinite(near).all()
    assert not wide[:,0::3].any() and not wide[:,1::3].any() and near[:,2].all()
    privileged=critic_privileged(sdf=torch.ones(1,2),feasible=torch.ones(1,dtype=torch.bool),observed=torch.zeros(1,dtype=torch.bool),
                                acceleration=torch.ones(1,3),controller=torch.tensor([1]))
    assert privileged.shape==(1,10)
    pipe.reset([0]);assert pipe.mapper.epoch.item()==1 and pipe.near.unknown.all()

def test_control_state_switch_hold_and_zero_history():
    from gd_lab.gast_nav.control_runtime import ControlRuntime, zero_legacy_command_history
    from gd_lab.gast_nav.landing_fsm import LandingObservation
    from gd_lab.gast_nav.supervisor import HealthInput
    from tensordict import TensorDict
    runtime=ControlRuntime([tile('nav_corridor')],CFG,RUNTIME,seed=42)
    runtime.reset([0],now_ms=0,yaws=[0.])
    o=LandingObservation(True,False,False,False,True,0.,0.,False,False)
    def tick(time,health):
        return runtime.before_policy(0,now_ms=time,dt=.01,xy=(0.,0.),yaw=0.,health=health,
                                     landing_observation=o,odom_epoch=0,legacy_velocity=(.5,0.,0.))
    normal=tick(10,HealthInput(0,0,0))
    assert normal.controller=='NAV_POLICY'
    held=tick(20,HealthInput(400,400,0))
    assert held.controller=='HOLD_3906' and held.intervention_started
    assert held.velocity==(0.,0.,0.) and not held.effective_go
    assert not tick(30,HealthInput(400,400,0)).intervention_started
    fail=tick(40,HealthInput(0,0,0,infer_failed=True))
    assert fail.controller=='DAMP' and fail.fallback_action=='VISION_HOLD'
    obs=TensorDict({'policy':torch.ones(2,230)},batch_size=[2])
    held_obs=zero_legacy_command_history(obs,term_dims=[3,3,3,12,12,12,1],history_length=5)
    assert not held_obs['policy'][:,30:45].any() and obs['policy'].all()
    assert held_obs['policy'][:,:30].all() and held_obs['policy'][:,45:].all()

def test_reverse_penalty_has_no_threshold():
    n=1;z=torch.zeros(n);one=torch.ones(n)
    terms,_=reward_terms(mode=torch.tensor([2]),go=one.bool(),phase=z.long(),feasible=one.bool(),observed=one.bool(),
                        v_path=z,v_cap=one,speed=z,yaw_rate=one*1e-5,turn_rate=-one*.2,positive_yaw=one*1e-7,
                        clearance=one,margin=one*.1,offset_error=z,yaw_error=z,turn_drift=z,contact=z,
                        entered_forbidden=z,cfg=CFG['rewards'])
    assert terms['reverse_rate'].item()==pytest.approx(1e-5)
    assert terms['reverse_accumulation'].item()==pytest.approx(1e-7)
