from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gd_lab.gast_nav.scene_config import load_scene_config
from gd_lab.gast_nav.scene_types import SceneIndex,StaticBox
from gd_lab.gast_nav.scene_geometry import point_box_distance
from gd_lab.gast_nav.terrains import build_scene
from gd_lab.gast_nav.turn_clutter import turn_free_polygon,cylinder_slot,segment_clear
from gd_lab.gast_nav.moving_cylinder import MovingCylinder,NoCylinderPathError
from gd_lab.gast_nav.scene_adapter import EpisodeCylinder
from gd_lab.gast_nav.feasibility import turn_feasible


@pytest.fixture(scope='module')
def corpus():
    settings=load_scene_config(); count=50; rows=[]; stats=[]
    for level in range(settings['terrain']['levels']):
        samples=[]
        for seed in range(count):
            _,_,m=build_scene('nav_turn_room',level/(settings['terrain']['levels']-1),seed=seed)
            polygon=np.asarray(turn_free_polygon(m,config=settings))
            area=abs(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))/2
            assert area>0 and np.isfinite(polygon).all()
            samples.append(m); rows.append(m)
        cylinder=EpisodeCylinder(SceneIndex(samples),stage='C',seed=161,config=settings)
        placed=int(cylinder.enabled.sum())
        assert placed/count>=.9, (level,placed,count)
        stats.append({'level':level,'scenes':count,'generation_passed':count,'polygon_passed':count,
                      'cylinder_placed':placed,'cylinder_rate':placed/count,
                      'all_success_one_sided_95_lower':.05**(1/count) if placed==count else None,
                      'radius_min':min(m.turn_spot[1] for m in samples),'radius_max':max(m.turn_spot[1] for m in samples),
                      'turn_feasible':sum(turn_feasible(m).feasible for m in samples),
                      'skip_report':cylinder.skip_report()})
    result={'seed_count_per_level':count,'levels':stats}
    _out=Path(__file__).resolve().parent/'.runtime'; _out.mkdir(parents=True,exist_ok=True); (_out/'turn161_stats.json').write_text(json.dumps(result,indent=2)+'\n')
    return rows,stats,settings


def test_polygon_workspace_generation_and_cylinder_rates(corpus):
    rows,stats,settings=corpus
    assert len(rows)==500 and len(stats)==10
    assert all(s['generation_passed']==s['scenes'] and s['polygon_passed']==s['scenes'] for s in stats)
    assert all(s['cylinder_rate']>=.9 for s in stats)


def test_actual_radius_tracks_level_delta_exactly_and_is_monotone(corpus):
    rows,stats,settings=corpus
    target=settings['oracle']['turn_swept_radius_m']+settings['oracle']['margin_m']
    for m in rows:
        delta=.3-.4*m.level/9
        assert m.turn_spot[1]==pytest.approx(target+delta,abs=1e-6)
        assert min(point_box_distance(m.turn_spot[0],b) for b in m.obstacles)==pytest.approx(m.turn_spot[1])
    assert all(b['radius_max']<a['radius_min'] for a,b in zip(stats,stats[1:]))
    assert stats[0]['radius_min']>target and stats[-1]['radius_max']<target
    assert stats[0]['turn_feasible']==50 and stats[-1]['turn_feasible']==0


def test_clutter_variety_is_not_four_wall_rectangle(corpus):
    rows,_,settings=corpus
    for m in rows:
        clutter=[b for b in m.obstacles if b.kind!='wall']
        assert 4<=len(clutter)<=12
        assert any(b.kind=='pillar' for b in clutter)
        assert any(.4<=2*b.half[2]<=.5 for b in clutter)
        assert all(.4<=2*b.half[2]<=1.2 for b in clutter)
        assert len({round(b.yaw,3) for b in clutter})>=3
        assert m.route_polyline[-1]==m.turn_spot[0]
        polygon=np.array(turn_free_polygon(m,config=settings))
        radii=np.linalg.norm(polygon-np.array(m.turn_spot[0]),axis=1)
        assert radii.max()-radii.min()>.1
        assert radii.min()==pytest.approx(m.turn_spot[1],abs=1e-6)


def test_reserved_cylinder_segment_and_sway_envelope_are_clear(corpus):
    rows,_,settings=corpus
    radius=max(settings['cylinder']['radius'])+settings['cylinder']['clearance_m']+settings['cylinder']['sway_max_offset_m']
    for m in rows:
        a,b=cylinder_slot(m,settings)
        for obstacle in m.obstacles:
            assert segment_clear(a,b,obstacle,radius,settings['terrain']['geometry_epsilon'])
        for p in (a,b):
            assert math.dist(p,m.turn_spot[0])>settings['cylinder']['robot_radius_m']+max(settings['cylinder']['radius'])+settings['cylinder']['clearance_m']


def test_oldest_and_hardest_turn_meshes_remain_closed_and_reproducible():
    for difficulty in (0.,1.):
        meshes,origin,m=build_scene('nav_turn_room',difficulty,seed=99)
        other,_,n=build_scene('nav_turn_room',difficulty,seed=99)
        assert m==n and np.array_equal(meshes[0].vertices,other[0].vertices)
        assert all(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0 for mesh in meshes)


def moving(n=1):
    _,_,m=build_scene('nav_door',0.,seed=8)
    obj=MovingCylinder(SceneIndex([m],[0]*n),stage='C',seed=18)
    obj.position[:]=torch.tensor([-2.,0.,.85]); obj.endpoints[:,0]=obj.position
    obj.endpoints[:,1]=torch.tensor([-1.,0.,.85]); obj.target[:]=1; obj.speed[:]=0.
    return obj


def test_sway_sign_gain_and_exact_low_pass_discretization():
    obj=moving(); dt=.02; accel=torch.tensor([[1.,-.5]])
    alpha=1-math.exp(-2*math.pi*obj.settings['sway_cutoff_hz']*dt)
    before=obj.position.clone(); offset=obj.update(dt,accel)
    torch.testing.assert_close(obj.a_train_lp,alpha*accel)
    torch.testing.assert_close(offset,-obj.settings['sway_gain_s2']*alpha*accel)
    assert torch.equal(obj.position,before)  # update is filter-only, no collider move
    obj.step(dt,torch.tensor([[10.,10.,.3]]))
    torch.testing.assert_close(obj.position[:,:2]-before[:,:2],offset)
    torch.testing.assert_close(obj.velocity[:,:2],offset/dt)


def test_sway_euclidean_cap_not_per_axis():
    obj=moving(); limit=obj.settings['sway_max_offset_m']
    offset=obj.update(10.,torch.tensor([[1e6,1e6]]))
    assert float(offset.norm())==pytest.approx(limit,rel=1e-5)
    assert float(offset[0,0])==pytest.approx(float(offset[0,1]))
    assert float(offset.norm())<=limit+1e-6


def test_sway_high_frequency_attenuation_and_zero_relaxation():
    obj=moving(); dt=.005; samples=[]
    for i in range(500):
        a=math.sin(2*math.pi*10*i*dt)
        samples.append(float(obj.update(dt,torch.tensor([[a,0.]]))[0,0]))
    assert max(abs(x) for x in samples[100:])<obj.settings['sway_gain_s2']*.2
    obj.update(2.,torch.zeros(1,2))
    assert float(obj.sway_offset.norm())<1e-4


def test_sway_is_absolute_offset_not_accumulating_drift():
    obj=moving(); far=torch.tensor([[10.,10.,.3]]); start=obj.position.clone()
    for _ in range(100):
        obj.update(.05,torch.tensor([[.5,0.]])); obj.step(.05,far)
    assert float(obj.position[0,0]-start[0,0])==pytest.approx(-.04,abs=1e-5)
    assert abs(float(obj.position[0,1]-start[0,1]))<1e-6


def test_post_offset_robot_collision_stops_and_applied_offset_is_not_advanced():
    obj=moving(); obj.position[:]=torch.tensor([0.,0.,.85]); obj.endpoints[:]=obj.position[:,None]
    obj.radius[:]=.25; limit=.25+obj.settings['robot_radius_m']+obj.settings['clearance_m']
    robot=torch.tensor([[limit+.03,0.,.3]])
    obj.update(10.,torch.tensor([[-2.,0.]]))  # shifts towards robot by 0.12
    before=obj.position.clone(); obj.step(.1,robot)
    assert bool(obj.blocked.all()) and torch.equal(obj.position,before)
    assert not bool(obj.applied_sway_offset.any()) and not bool(obj.velocity.any())


def test_post_offset_static_obstacle_stops_before_wall():
    obj=moving(); meta=obj.index.tiles[0]
    obstacle=StaticBox((-1.5,0.,.5),(.1,.3,.5),0.,'box')
    obj.index=SceneIndex([replace(meta,obstacles=(obstacle,))])
    obj.position[:]=torch.tensor([-1.92,0.,.85]); obj.endpoints[:]=obj.position[:,None]; obj.radius[:]=.25
    obj.update(10.,torch.tensor([[-2.,0.]])); before=obj.position.clone()
    obj.step(.1,torch.tensor([[10.,10.,.3]]))
    assert bool(obj.blocked.all()) and torch.equal(obj.position,before)


def test_sway_partial_reset_and_inactive_rows():
    obj=moving(n=2); obj.update(.1,torch.tensor([[1.,0.],[0.,1.]]))
    preserved=obj.a_train_lp[1].clone(); obj.reset([0],stage='B')
    assert not bool(obj.sway_offset[0].any()) and not bool(obj.a_train_lp[0].any())
    assert torch.equal(obj.a_train_lp[1],preserved)
    obj.update(.1,torch.ones(2,2)); assert not bool(obj.sway_offset[0].any())
    assert bool(obj.sway_offset[1].any())


@pytest.mark.parametrize('accel',[torch.zeros(1,3),torch.tensor([[float('nan'),0.]])])
def test_sway_invalid_acceleration_rejected(accel):
    with pytest.raises(ValueError,match='a_train_xy'): moving().update(.1,accel)


def test_lpf_constant_input_is_invariant_to_dt_partition():
    a=moving(); b=moving(); acceleration=torch.tensor([[.4,-.3]])
    a.update(.1,acceleration)
    for _ in range(20): b.update(.005,acceleration)
    torch.testing.assert_close(a.a_train_lp,b.a_train_lp)
    torch.testing.assert_close(a.sway_offset,b.sway_offset)


def test_sway_reversal_uses_nominal_endpoint_and_capsule_tracks_actual_pose():
    obj=moving(); start=obj.position.clone(); obj.speed[:]=1.; obj.pingpong[:]=True
    obj.endpoints[:,1]=start+torch.tensor([.1,0.,0.])
    obj.update(2.,torch.tensor([[.5,0.]]))
    obj.step(.2,torch.tensor([[10.,10.,.3]]))
    assert int(obj.target[0])==0
    assert obj.position[0,0]==pytest.approx(float(start[0,0])+.1+float(obj.sway_offset[0,0]))
    capsule=obj.capsules()
    torch.testing.assert_close(capsule.starts[:,0,:2],obj.position[:,:2])
    torch.testing.assert_close(capsule.ends[:,0,:2],obj.position[:,:2])


def test_sway_blocked_then_clear_resume_never_accumulates_hidden_progress():
    obj=moving(); obj.position[:]=torch.tensor([0.,0.,.85]); obj.endpoints[:]=obj.position[:,None]; obj.radius[:]=.25
    limit=.25+obj.settings['robot_radius_m']+obj.settings['clearance_m']
    obj.update(10.,torch.tensor([[-2.,0.]])); obj.step(.1,torch.tensor([[limit+.03,0.,.3]]))
    assert obj.position[0,0]==0
    obj.step(.1,torch.tensor([[10.,10.,.3]]))
    assert obj.position[0,0]==pytest.approx(obj.settings['sway_max_offset_m'])
    obj.update(10.,torch.zeros(1,2)); obj.step(.1,torch.tensor([[10.,10.,.3]]))
    assert obj.position[0,0]==pytest.approx(0.,abs=1e-6)


def test_static_collision_cache_rebind_and_batch_separation():
    obj=moving(2); meta=obj.index.tiles[0]
    obstacle=StaticBox((-1.5,0.,.5),(.1,.3,.5),0.,'box')
    blocked=replace(meta,obstacles=(obstacle,)); clear=replace(meta,obstacles=())
    obj.index=SceneIndex([blocked,clear],[0,1]); obj.position[:]=torch.tensor([-1.92,0.,.85]); obj.endpoints[:]=obj.position[:,None]; obj.radius[:]=.25
    obj.update(10.,torch.tensor([[-2.,0.],[-2.,0.]])); obj.step(.1,torch.tensor([[10.,10.,.3],[10.,10.,.3]]))
    assert obj.blocked.tolist()==[True,False]
    obj.index.bind([0],[1]); obj.step(.1,torch.tensor([[10.,10.,.3],[10.,10.,.3]]))
    assert not bool(obj.blocked.any())


def test_invalid_sway_config_is_rejected_before_mutating_filter():
    obj=moving(); obj.settings=dict(obj.settings); obj.settings['sway_cutoff_hz']=float('nan')
    with pytest.raises(ValueError,match='sway settings'): obj.update(.1,torch.ones(1,2))
    assert not bool(obj.a_train_lp.any())
