from dataclasses import replace
import copy
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import trimesh
import yaml
from gd_lab.gast_nav.scene_config import load_scene_config
from gd_lab.gast_nav.scene_types import StaticBox, SceneTileMeta, SceneIndex, project_route, route_point
from gd_lab.gast_nav.terrains import build_scene, FUNCTIONS, KINDS, META_KEY
from gd_lab.gast_nav.scene_geometry import point_box_distance, floor_height
from gd_lab.gast_nav.feasibility import passage_feasible, turn_feasible, batch_feasibility
from gd_lab.gast_nav.moving_cylinder import MovingCylinder


def settings_file(tmp_path, mutate=None):
    settings=load_scene_config()
    settings['corridor']['box_count']=[0,0]; settings['corridor']['bend_deg']=[0.,0.]; settings['corridor']['pillar_probability']=0.
    settings['stairs']['box_count']=[0,0]
    if mutate: mutate(settings)
    path=tmp_path/'scenes.yaml'; path.write_text(yaml.safe_dump(settings))
    return SimpleNamespace(size=tuple(settings['terrain']['size']),settings_path=str(path),seed=42),settings


def height_at(meshes,xy,*,max_z=math.inf):
    # Independent CPU vertical ray / triangle projection; no rtree/simulator.
    triangles=np.concatenate([m.triangles for m in meshes])
    a=triangles[:,0]; b=triangles[:,1]-a; c=triangles[:,2]-a
    v=np.array(xy)-a[:,:2]
    det=b[:,0]*c[:,1]-b[:,1]*c[:,0]
    good=np.abs(det)>1e-10
    u=np.zeros_like(det); w=np.zeros_like(det)
    u[good]=(v[good,0]*c[good,1]-v[good,1]*c[good,0])/det[good]
    w[good]=(b[good,0]*v[good,1]-b[good,1]*v[good,0])/det[good]
    z=a[:,2]+u*b[:,2]+w*c[:,2]
    good &= (u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)&(z<=max_z)
    return float(z[good].max()) if good.any() else math.nan


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('difficulty',[0.,.5,1.])
def test_scene_mesh_watertight_normals_and_bounds(kind,difficulty):
    meshes,origin,meta=build_scene(kind,difficulty,seed=7)
    assert meta.kind==kind and np.array_equal(origin,[8.,6.,0.])
    assert meshes and all(m.is_watertight and m.is_winding_consistent and m.volume>0 for m in meshes)
    vertices=np.concatenate([m.vertices for m in meshes])
    assert np.isfinite(vertices).all() and (vertices[:,:2]>=-1e-8).all() and (vertices[:,:2]<=[16.+1e-8,12.+1e-8]).all()
    assert meta.route_polyline[0][0] < meta.route_polyline[-1][0]
    assert meshes[0].metadata[META_KEY]['kind']==kind


@pytest.mark.parametrize('kind',KINDS)
def test_callback_matches_isaac_signature_origin_and_metadata(kind):
    cfg=SimpleNamespace(size=(16.,12.),seed=12,settings_path=None)
    meshes,origin=FUNCTIONS[kind](.4,cfg)
    reference,reference_origin,meta=build_scene(kind,.4,cfg)
    assert np.array_equal(origin,reference_origin)
    assert np.allclose(trimesh.util.concatenate(meshes).bounds,trimesh.util.concatenate(reference).bounds)
    assert meshes[0].metadata[META_KEY]['route_polyline']==meta.route_polyline


@pytest.mark.parametrize('kind',KINDS)
def test_seed_reproducible_without_global_rng_mutation(kind):
    state=np.random.get_state()
    a,origin,m=build_scene(kind,.3,seed=100)
    b,_,n=build_scene(kind,.3,seed=100)
    assert m==n and np.array_equal(a[0].vertices,b[0].vertices)
    restored=np.random.get_state()
    assert state[0]==restored[0] and np.array_equal(state[1],restored[1]) and state[2:]==restored[2:]


@pytest.mark.parametrize('kind',KINDS)
def test_all_metadata_obstacles_correspond_to_exact_mesh_boxes(kind):
    meshes,origin,meta=build_scene(kind,.5,seed=5)
    for obstacle in meta.obstacles:
        rotation=trimesh.transformations.rotation_matrix(obstacle.yaw,[0,0,1]); rotation[:3,3]=np.array(obstacle.center)+origin
        expected=trimesh.creation.box(np.array(obstacle.half)*2,rotation)
        assert any(np.allclose(expected.bounds,m.bounds,atol=1e-8) and np.isclose(expected.volume,m.volume) for m in meshes)


@pytest.mark.parametrize('kind',KINDS)
def test_difficulty_monotone_narrowing_and_level_bounds(kind,tmp_path):
    cfg,_=settings_file(tmp_path)
    widths=[]; levels=[]
    for d in np.linspace(0.,1.,11):
        _,_,m=build_scene(kind,float(d),cfg)
        # UPDATE161 turn difficulty is clearance, no longer rectangle width.
        widths.append(m.turn_spot[1] if kind=='nav_turn_room' else min(w for _,w in m.width_profile)); levels.append(m.level)
    assert all(b<=a+1e-7 for a,b in zip(widths,widths[1:])),widths
    assert levels[0]==0 and levels[-1]==9 and levels==sorted(levels)


def test_corridor_actual_width_and_safe_approach(tmp_path):
    cfg,c=settings_file(tmp_path)
    meshes,origin,m=build_scene('nav_corridor',1.,cfg)
    sidewalls=[o for o in m.obstacles if o.kind=='wall']
    positive=min(o.center[1]-o.half[1] for o in sidewalls if o.center[1]>0)
    negative=max(o.center[1]+o.half[1] for o in sidewalls if o.center[1]<0)
    assert positive-negative==pytest.approx(.5)
    assert min(w for _,w in m.width_profile)==pytest.approx(.5)
    assert m.width_profile[0][1]>=1.5
    assert min(point_box_distance(m.route_polyline[0],b) for b in sidewalls)>=.8-1e-7


def test_door_aperture_threshold_and_both_gaps_mesh(tmp_path):
    def change(c): c['door']['gap']=[.2,.2]; c['door']['threshold']=[.03,.03]; c['door']['thickness']=[.1,.1]
    cfg,c=settings_file(tmp_path,change)
    meshes,origin,m=build_scene('nav_door',1.,cfg)
    assert min(w for _,w in m.width_profile)==pytest.approx(.5)
    assert height_at(meshes,origin[:2],max_z=.5)==pytest.approx(.03)
    for x in (-.65,.65):
        assert height_at(meshes,origin[:2]+[x,0.],max_z=.5)==pytest.approx(-.7)
        assert height_at(meshes,origin[:2]+[x-.15,0.],max_z=.5)==pytest.approx(0.)
    # Header over centre is present, not a semantic-only opening.
    assert height_at(meshes,origin[:2])>=1.9


def test_door_zero_gap_is_continuous_floor(tmp_path):
    cfg,_=settings_file(tmp_path,lambda c:c['door'].update(gap=[0.,0.],threshold=[0.,0.]))
    meshes,origin,_=build_scene('nav_door',0.,cfg)
    assert height_at(meshes,origin[:2]+[.65,0.],max_z=.5)==pytest.approx(0.)


def test_stair_landing_bounds_and_heights_match_mesh(tmp_path):
    cfg,_=settings_file(tmp_path)
    meshes,origin,m=build_scene('nav_stair_landing',.5,cfg)
    (x0,y0),(x1,y1)=m.landing
    centre=(np.array([x0,y0])+np.array([x1,y1]))/2
    top=floor_height(m,centre)
    assert top>0 and x1-x0==pytest.approx(1.3)
    assert height_at(meshes,centre+origin[:2],max_z=top+.01)==pytest.approx(top)
    points=np.linspace(m.route_polyline[0][0],m.route_polyline[-1][0],31)
    z=[height_at(meshes,origin[:2]+[x,0.],max_z=top+.01) for x in points]
    assert z[0]==pytest.approx(0.) and z[-1]==pytest.approx(0.) and max(z)==pytest.approx(top)
    types=[s[0] for s in m.segments]
    assert types==['FLAT','STAIR_UP','LANDING','STAIR_DOWN','FLAT']
    landing_segment=next(s for s in m.segments if s[0]=='LANDING')
    assert m.route_polyline[0][0]+landing_segment[1]==pytest.approx(x0)
    assert m.route_polyline[0][0]+landing_segment[2]==pytest.approx(x1)


def simple_meta(width=.7,*,turn_radius=None):
    return SceneTileMeta('nav_corridor' if turn_radius is None else 'nav_turn_room',0,((-1.,0.),(1.,0.)),((0.,width),(2.,width)),(('PASSAGE',0.,2.),),None,None if turn_radius is None else ((0.,0.),turn_radius),(),'flat',False)


@pytest.mark.parametrize('delta,expected',[(-1e-7,False),(0.,True),(1e-7,True)])
def test_passage_feasibility_exact_swept_plus_margin(delta,expected):
    from gd_lab.gast_nav.scene_config import load_scene_config
    o=load_scene_config()['oracle']; W,m=o['swept_width_m'],o['margin_m']
    r=passage_feasible(simple_meta(W+2*m+delta))
    assert r.feasible==expected and r.required==pytest.approx(W+2*m)


@pytest.mark.parametrize('delta,expected',[(-1e-7,False),(0.,True),(1e-7,True)])
def test_turn_feasibility_exact_radius_plus_margin(delta,expected):
    from gd_lab.gast_nav.scene_config import load_scene_config
    o=load_scene_config()['oracle']; R,m=o['turn_swept_radius_m'],o['margin_m']
    r=turn_feasible(simple_meta(1.5,turn_radius=R+m+delta))
    assert r.feasible==expected and r.required==pytest.approx(R+m)


def test_feasibility_uses_remaining_route_not_passed_bottleneck():
    m=replace(simple_meta(),width_profile=((0.,1.5),(.5,.5),(.6,.5),(.60001,1.5),(2.,1.5)))
    assert not passage_feasible(m,(-1.,0.)).feasible
    assert passage_feasible(m,(.5,0.)).feasible


def test_turn_feasibility_recomputes_at_current_position(tmp_path):
    cfg,_=settings_file(tmp_path)
    _,_,m=build_scene('nav_turn_room',0.,cfg)
    assert turn_feasible(m).feasible
    obstacle=m.obstacles[0]
    assert not turn_feasible(m,obstacle.center[:2]).feasible
    assert m.turn_spot[1]==pytest.approx(min(point_box_distance(m.turn_spot[0],b) for b in m.obstacles))


def test_scene_index_lookup_binding_padded_tensors_and_grid(tmp_path):
    cfg,_=settings_file(tmp_path)
    metas=[build_scene(k,.4,cfg)[2] for k in KINDS]
    index=SceneIndex(metas,[3,0,2,1,0])
    assert [m.kind for m in index.lookup(torch.tensor([0,4]))]==['nav_stair_landing','nav_corridor']
    tensors=index.tensors('cpu')
    assert tensors['route_points'].shape[0]==5 and tensors['landing_mask'].tolist()==[True,False,False,False,False]
    assert tensors['train_inertia'].tolist()==[False,False,True,False,False]
    assert tensors['spawn_xy'][2].tolist()==pytest.approx(metas[2].turn_spot[0])
    index.bind([0],[1]); assert index.lookup([0])[0].kind=='nav_door'
    grid=SceneIndex.from_grid({(0,0):metas[0],(0,1):metas[1]},[0,0],[1,0])
    assert [m.kind for m in grid.lookup([0,1])]==['nav_door','nav_corridor']
    with pytest.raises(IndexError): index.lookup([-1])


def test_route_projection_arc_distance_and_left_sign():
    m=replace(simple_meta(),route_polyline=((0.,0.),(1.,0.),(1.,1.)))
    s,lateral,yaw=project_route(m,(1.1,.5))
    assert s==pytest.approx(1.5) and lateral==pytest.approx(-.1) and yaw==pytest.approx(math.pi/2)
    assert route_point(m,1.5)[0]==pytest.approx((1.,.5))


def test_metadata_validation_rejects_unsafe_inputs():
    with pytest.raises(ValueError): StaticBox((0.,0.,0.),(-1.,1.,1.),0.,'box')
    with pytest.raises(ValueError): build_scene('wrong',.5)
    with pytest.raises(ValueError): build_scene('nav_corridor',1.01)
    with pytest.raises(ValueError): build_scene('nav_door',.5,SimpleNamespace(size=(1.,1.)))


def cylinder_fixture(tmp_path,*,stage='C',n=1):
    cfg,c=settings_file(tmp_path)
    meta=build_scene('nav_door',0.,cfg)[2]
    index=SceneIndex([meta],[0]*n)
    return MovingCylinder(index,stage=stage,seed=8,config=c),c


@pytest.mark.parametrize('stage',['A0','A','B'])
def test_cylinder_absent_before_stage_c(tmp_path,stage):
    obj,c=cylinder_fixture(tmp_path,stage=stage)
    assert not bool(obj.enabled.any()) and obj.capsules() is None
    assert obj.position[0,2]==c['cylinder']['disabled_z_m']


def test_stage_c_one_cylinder_dimensions_and_world_capsule(tmp_path):
    obj,c=cylinder_fixture(tmp_path,n=3)
    assert bool(obj.enabled.all()) and bool(((obj.radius>=.2)&(obj.radius<=.3)).all())
    assert bool((obj.height==1.7).all()) and bool(((obj.speed>=0)&(obj.speed<=.8)).all())
    cap=obj.capsules(); assert cap.starts.shape==(3,1,3)
    assert torch.allclose(cap.starts[:,0,2],obj.position[:,2]-.85)
    assert torch.allclose(cap.ends[:,0,2],obj.position[:,2]+.85)
    assert torch.equal(cap.radii[:,0],obj.radius)


def set_motion(obj,start,end,speed=1.,pingpong=False):
    obj.position[:]=torch.tensor(start)
    obj.endpoints[:,0]=torch.tensor(start); obj.endpoints[:,1]=torch.tensor(end)
    obj.speed[:]=speed; obj.radius[:]=.25; obj.enabled[:]=True; obj.target[:]=1; obj.pingpong[:]=pingpong


def test_cylinder_continuous_swept_collision_stops_even_if_endpoint_clear(tmp_path):
    obj,_=cylinder_fixture(tmp_path)
    set_motion(obj,[-2.,0.,.85],[2.,0.,.85],speed=4.)
    before=obj.position.clone(); obj.step(1.,torch.tensor([[0.,0.,.3]]))
    assert bool(obj.blocked.all()) and torch.equal(obj.position,before) and not bool(obj.velocity.any())


def test_cylinder_does_not_stop_for_robot_at_other_height_or_far_away(tmp_path):
    obj,_=cylinder_fixture(tmp_path)
    set_motion(obj,[-2.,0.,.85],[2.,0.,.85])
    obj.step(.5,torch.tensor([[0.,5.,.3]]))
    assert not bool(obj.blocked.any()) and obj.position[0,0]==pytest.approx(-1.5)
    obj.step(.5,torch.tensor([[0.,0.,10.]]))
    assert not bool(obj.blocked.any())


def test_cylinder_predicted_robot_motion_blocks_and_overlap_flagged(tmp_path):
    obj,_=cylinder_fixture(tmp_path)
    set_motion(obj,[0.,0.,.85],[0.,1.,.85],speed=0.)
    obj.step(1.,torch.tensor([[3.,0.,.3]]),torch.tensor([[-4.,0.,0.]]))
    assert bool(obj.blocked.all()) and not bool(obj.initial_overlap.any())
    obj.step(.1,torch.tensor([[0.,0.,.3]]))
    assert bool(obj.initial_overlap.all())


def test_cylinder_pingpong_no_overshoot_and_straight_holds_at_end(tmp_path):
    obj,_=cylinder_fixture(tmp_path)
    set_motion(obj,[0.,0.,.85],[.1,0.,.85],speed=1.,pingpong=True)
    far=torch.tensor([[0.,10.,.3]])
    obj.step(1.,far); assert obj.position[0,0]==pytest.approx(.1) and int(obj.target[0])==0
    obj.step(1.,far); assert obj.position[0,0]==pytest.approx(0.) and int(obj.target[0])==1
    obj.pingpong[:]=False; obj.step(1.,far); pos=obj.position.clone(); obj.step(1.,far)
    assert torch.equal(pos,obj.position) and not bool(obj.velocity.any())


def test_cylinder_collider_writer_matches_capsule_pose_and_actual_velocity(tmp_path):
    obj,_=cylinder_fixture(tmp_path)
    set_motion(obj,[0.,0.,.85],[2.,0.,.85]); obj.step(.1,torch.tensor([[0.,10.,.3]]))
    class Fake:
        def write_root_pose_to_sim(self,pose,env_ids): self.pose=pose; self.ids=env_ids
        def write_root_velocity_to_sim(self,velocity,env_ids): self.velocity=velocity
    fake=Fake(); obj.write_colliders(fake)
    assert torch.equal(fake.pose[:,:3],obj.position) and fake.pose[0,3]==1
    assert torch.equal(fake.velocity[:,:3],obj.velocity)
    assert obj.collider_specs()[0]['kinematic_enabled']


def test_cylinder_no_safe_spawn_is_explicit_not_silent_skip(tmp_path):
    cfg,c=settings_file(tmp_path)
    # Preserve the NO_PATH regression with an explicitly tiny legacy fixture;
    # UPDATE161 generated turn workspaces intentionally reserve a cylinder slot.
    m=replace(simple_meta(.4,turn_radius=.15),route_polyline=((-0.1,0.),(0.1,0.)))
    with pytest.raises(ValueError,match='NO_CYLINDER_PATH'):
        MovingCylinder(SceneIndex([m]),seed=0,config=c)


def test_batch_feasibility_cpu_adapter(tmp_path):
    cfg,c=settings_file(tmp_path)
    metas=[build_scene('nav_corridor',1.,cfg)[2],build_scene('nav_turn_room',0.,cfg)[2]]
    result=batch_feasibility(SceneIndex(metas),torch.tensor([[-4.,0.],[0.,0.]]),torch.tensor([False,True]),config=c)
    assert result['feasible'].tolist()==[False,True]


def test_scene_modules_do_not_import_isaac_or_simulator():
    # Fresh interpreter: policy-side tests in this process import isaaclab.
    import subprocess, sys
    mods=('scene_types','scene_config','scene_geometry','terrains','feasibility','moving_cylinder')
    code=('import sys; '+'; '.join(f'import gd_lab.gast_nav.{m}' for m in mods)+
          "; print(any(k=='isaaclab' or k.startswith('omni.') for k in sys.modules))")
    out=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True).stdout.strip()
    assert out=='False'


def test_isaac_generator_hook_keeps_metadata_through_centring_and_placement(tmp_path,monkeypatch):
    # Verify owned adapter algebra with a duck-typed base, not an Isaac smoke.
    import sys,types
    from gd_lab.gast_nav.terrains import make_isaac_generator_class
    class Config(SimpleNamespace):
        def copy(self): return copy.deepcopy(self)
    class Base:
        def __init__(self,cfg): self.cfg=cfg
        def _get_terrain_mesh(self,difficulty,cfg): return 'legacy',np.zeros(3)
        def _add_sub_terrain(self,mesh,origin,row,col,cfg):
            translation=np.array([(row+.5)*self.cfg.size[0],(col+.5)*self.cfg.size[1],0.])
            mesh.apply_translation(translation); self.origin_world=origin+translation
    module=types.ModuleType('isaaclab.terrains'); module.TerrainGenerator=Base
    package=types.ModuleType('isaaclab'); package.__path__=[]
    monkeypatch.setitem(sys.modules,'isaaclab',package); monkeypatch.setitem(sys.modules,'isaaclab.terrains',module)
    cls=make_isaac_generator_class()
    gen=cls(Config(size=(16.,12.),seed=9,use_cache=False))
    cfg=Config(size=(16.,12.),seed=0,settings_path=None,function=FUNCTIONS['nav_door'])
    mesh,origin=gen._get_terrain_mesh(.5,cfg)
    assert np.array_equal(origin,[0.,0.,0.])
    assert np.allclose(mesh.bounds[0,:2],[-8.,-6.]) and np.allclose(mesh.bounds[1,:2],[8.,6.])
    gen._add_sub_terrain(mesh,origin,2,3,cfg)
    assert gen.scene_grid[(2,3)].kind=='nav_door'
    assert np.array_equal(gen.origin_world,[40.,42.,0.])
    assert gen.scene_grid[(2,3)].route_polyline[0]==(-3.,0.)
    gen.cfg.use_cache=True
    with pytest.raises(ValueError,match='use_cache=False'): gen._get_terrain_mesh(.5,cfg)
    legacy=Config(function=lambda d,c:None)
    assert gen._get_terrain_mesh(.5,legacy)[0]=='legacy'


def test_cylinder_stage_c_origin_and_partial_reset_isolation(tmp_path):
    cfg,c=settings_file(tmp_path)
    meta=build_scene('nav_door',0.,cfg)[2]
    origins=torch.tensor([[0.,0.,0.],[20.,30.,1.]])
    obj=MovingCylinder(SceneIndex([meta],[0,0]),stage='C',seed=10,env_origins=origins,config=c)
    second=obj.position[1].clone(); obj.reset([0])
    assert torch.equal(obj.position[1],second)
    assert obj.position[1,2]==pytest.approx(1.85)
    assert obj.capsules().starts[1,0,2]==pytest.approx(1.)


def test_cylinder_path_stays_clear_of_static_obstacles(tmp_path):
    cfg,c=settings_file(tmp_path)
    metas=[build_scene(kind,0.,cfg)[2] for kind in ('nav_door','nav_corridor','nav_stair_landing')]
    obj=MovingCylinder(SceneIndex(metas),stage='C',seed=9,config=c)
    for i,meta in enumerate(metas):
        for t in np.linspace(0.,1.,11):
            point=(1-t)*obj.endpoints[i,0]+t*obj.endpoints[i,1]
            z=float(point[2]-.85)
            for box in meta.obstacles:
                if box.kind=='step' or box.center[2]+box.half[2]<=z: continue
                assert point_box_distance(point[:2].tolist(),box)>float(obj.radius[i])


def test_default_randomized_mesh_and_metadata_seed_corpus():
    for seed in range(6):
        for kind in KINDS:
            _,_,meta=build_scene(kind,seed/5.,seed=seed)
            assert all(math.isfinite(s+w) and w>=0 for s,w in meta.width_profile)
            assert meta.width_profile[0][0]==0.
            assert meta.width_profile[-1][0]==pytest.approx(meta.route_length)


def test_meta_tensors_do_not_modify_shared_schema():
    from gd_lab.gast_nav import schema
    assert schema.SEGMENT_TYPES==('FLAT','PASSAGE','STAIR_UP','LANDING','STAIR_DOWN','TURN_SPOT')
