"""B-1 full-volume proof and appendix D counterexamples 1..6, CPU only."""
from dataclasses import replace
import math
import json
from pathlib import Path
import pytest
import torch
from gd_lab.gast_nav import schema as S
from gd_lab.gast_nav.map_types import Grid,RayBatch,WIDE_GRID
from gd_lab.gast_nav.free_evidence167 import (FreeEvidenceConfig,FreeFrusta,FreeCapture,FreeMeasurement,
    FreeEvidenceHistory,frusta_from_d4xx,volume_free_proof,pose_budget)
from gd_lab.gast_nav.obstacle_map import ray_evidence,ObstacleMap
from gd_lab.gast_nav.near_clearance import build_near,extended_near_grid
from gd_lab.gast_nav.ray_geometry import resolve_first_hits

GRID=Grid(1,1,.1,.5,0.)
CFG=FreeEvidenceConfig()

def prism(x0=-1.,x1=2.,y0=-1.,y1=1.,z0=0.,z1=3.,valid=True):
    # Orthographic free-volume fixture with the same six-plane containment API.
    normals=torch.tensor([[[[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.],[1.,0.,0.],[-1.,0.,0.]]]])
    return FreeFrusta(torch.zeros(1,1,3),normals,torch.tensor([[[y0,-y1,z0,-z1,x0,-x1]]],dtype=torch.float32),torch.tensor([[x1]],dtype=torch.float32),torch.tensor([[valid]]))

def cap(f=None,*,grid=GRID,t=1000,support=None,epoch=0,pose=None):
    return FreeCapture(f or prism(),torch.zeros(1,4) if pose is None else pose,
        torch.zeros(1,grid.ny,grid.nx) if support is None else support,torch.tensor([t]),torch.tensor([epoch]))

def measurement(captures=None,*,now=1000,grid=GRID,pose=None,travel=None,settings=CFG,epoch=0):
    captures=(cap(grid=grid),) if captures is None else tuple(captures)
    return FreeMeasurement(captures,torch.zeros(1,4) if pose is None else pose,torch.tensor([now]),torch.tensor([epoch]),
        torch.zeros(1,len(captures)) if travel is None else travel,torch.zeros(1),torch.zeros(1),settings)

def rays(x=0.,z=.15,dist=math.inf,self_hit=False):
    return resolve_first_hits(RayBatch(torch.tensor([[[x,0.,z]]]),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[dist]]),torch.tensor([[self_hit]])))

def proof(m,*,grid=GRID,support=None):
    return volume_free_proof(torch.zeros(1,grid.ny,grid.nx) if support is None else support,m,grid=grid)


def test_positive_full_volume_replaces_sparse_subbins():
    support=torch.zeros(1,1,1)
    legacy,_=ray_evidence(rays(),support,grid=GRID)
    measured,_,age=ray_evidence(rays(),support,grid=GRID,measurement=measurement(),return_age=True)
    assert (legacy==S.UNKNOWN).all() and (measured==S.FREE).all() and (age==0).all()


def test_no_measurement_unknown_not_legacy_free_and_occupied_wins():
    from tests.gast_nav.test_maps import full_rays
    support=torch.zeros(1,1,1)
    state,_=ray_evidence(resolve_first_hits(full_rays()),support,grid=GRID,certified=True)
    assert (state==S.UNKNOWN).all()
    state,_=ray_evidence(rays(dist=.5),support,grid=GRID,measurement=measurement())
    assert state[0,0,0,0]==S.OCCUPIED


def test_pose_range_support_errors_shrink_not_expand():
    thin=cap(prism(y0=-.06,y1=.06))
    assert not proof(measurement([thin])).free.any()
    zero=replace(CFG,pose_sigma=0.,support_error_m=0.,sensor_angle_error_deg=0.,range_error_floor_m=0.,range_error_fraction=0.)
    assert proof(measurement([thin],settings=zero)).free.all()
    far=cap(prism(x1=.56))
    assert not proof(measurement([far])).free.any()


def test_union_complete_boxes_from_multiple_captures_and_age_not_refreshed():
    # Each alone misses part of the cell; two eroded overlapping volumes tile it.
    a=cap(prism(y1=.045),t=1000);b=cap(prism(y0=-.045),t=900)
    cfg=replace(CFG,historical_obstacle_speed_bound_mps=0.,pose_sigma=0.,sensor_angle_error_deg=0.)
    assert not proof(measurement([a],settings=cfg)).free.all()
    assert not proof(measurement([b],settings=cfg)).free.all()
    result=proof(measurement([a,b],settings=cfg))
    assert result.free.all() and torch.allclose(result.age_s,torch.full_like(result.age_s,.1))


def test_unknown_motion_bound_disallows_old_free_even_if_ttl_valid():
    assert not proof(measurement([cap(t=900)])).free.any()


def test_counterexample1_person_enters_occluded_old_free():
    cfg=replace(CFG,historical_obstacle_speed_bound_mps=1.)
    old=cap(prism(y0=-.20,y1=.20),t=800)
    assert proof(measurement([replace(old,time_ms=torch.tensor([1000]))],settings=cfg)).free.all()
    # Arrival from an unseen boundary can reach the cell in 0.2s. Reachability
    # erosion makes it UNKNOWN even with NO current person ray hit.
    assert not proof(measurement([old],settings=cfg)).free.any()


def test_counterexample2_expired_wall_cannot_be_reused_as_free():
    cfg=replace(CFG,historical_obstacle_speed_bound_mps=0.)
    old=cap(prism(x1=.45),t=400) # wall first-hit before this cell
    state,_,age=ray_evidence(rays(),torch.zeros(1,1,1),grid=GRID,measurement=measurement([old],settings=cfg),return_age=True)
    assert (state==S.UNKNOWN).all() and torch.isinf(age).all()
    assert not proof(measurement([cap(t=400)],settings=cfg)).free.any() # even old free expires


def test_counterexample3_single_unseen_5cm_pillar_cell_not_ignored():
    grid=Grid(3,1,.05,.45,0.)
    # Measured free volumes leave an actual 5cm gap, not a centre-point test.
    a=prism(x1=.475);b=prism(x0=.525)
    f=FreeFrusta(*(torch.cat([getattr(a,k),getattr(b,k)],1) for k in ('origins','normals','offsets','far_m','volume_valid')))
    result=proof(measurement([cap(f,grid=grid)],grid=grid),grid=grid)
    assert not result.free[:,:,0,1].any()
    assert not result.free[:,:2].all() # strict low-band certification must refuse


def test_counterexample4_one_cell_wide_long_unknown_strip():
    grid=Grid(9,5,.05,.3,-.1)
    a=prism(y1=-.02499);b=prism(y0=.02499)
    f=FreeFrusta(*(torch.cat([getattr(a,k),getattr(b,k)],1) for k in ('origins','normals','offsets','far_m','volume_valid')))
    ideal=replace(CFG,pose_sigma=0.,support_error_m=0.,sensor_angle_error_deg=0.)
    result=proof(measurement([cap(f,grid=grid)],grid=grid,settings=ideal),grid=grid)
    assert result.free[:,:,0,:].all() and result.free[:,:,-1,:].all()
    assert not result.free[:,:,2,:].any()
    required=torch.ones(1,5,9,dtype=torch.bool)
    known=result.free[:,:2].all(1)
    assert not bool((known|~required).all())


def test_counterexample5_gap_support_unknown_never_filled_by_free_rays():
    result=proof(measurement(),support=torch.full((1,1,1),torch.nan))
    assert not result.free.any() and torch.isinf(result.age_s).all()


@pytest.mark.parametrize('which',['support','pose','epoch'])
def test_counterexample6_changed_support_odom_epoch_fail_closed(which):
    cfg=replace(CFG,historical_obstacle_speed_bound_mps=0.)
    m=measurement([cap(t=900)],settings=cfg)
    support=torch.zeros(1,1,1)
    if which=='support':support+=.2
    if which=='pose':m=replace(m,pose=torch.tensor([[.1,0.,0.,0.]]),travel_m=torch.tensor([[.1]]))
    if which=='epoch':m=replace(m,epoch=torch.tensor([1]))
    assert not proof(m,support=support).free.any()


def test_self_first_hit_truncates_footprint_no_free_behind():
    # The producer supplies nearest return INCLUDING self, never replaces it by inf.
    f=prism(x1=.2)
    state,_=ray_evidence(rays(dist=.2,self_hit=True),torch.zeros(1,1,1),grid=GRID,measurement=measurement([cap(f)]))
    assert (state==S.UNKNOWN).all()


def test_d4xx_intrinsics_extrinsics_min_depth_and_invalid_pixel():
    depth=torch.full((1,1,4,4),2.)
    k=torch.tensor([[[100.,0.,1.5],[0.,100.,1.5],[0.,0.,1.]]]);rot=torch.eye(3)[None,None];t=torch.tensor([[[.3,.2,.4]]])
    blocks=torch.tensor([[0,0,0,4,4]])
    f=frusta_from_d4xx(depth,k,rot,t,blocks=blocks,volume_valid=torch.tensor([[True]]))
    assert torch.equal(f.origins[:,0],t[:,0]) and f.volume_valid.all() and f.far_m.item()==2.
    depth[0,0,1,1]=.7
    assert frusta_from_d4xx(depth,k,rot,t,blocks=blocks,volume_valid=torch.tensor([[True]])).far_m.item()==pytest.approx(.7)
    depth[0,0,1,1]=torch.nan
    assert not frusta_from_d4xx(depth,k,rot,t,blocks=blocks,volume_valid=torch.tensor([[True]])).volume_valid.any()
    assert not frusta_from_d4xx(torch.full_like(depth,2.),k,rot,t,blocks=blocks,volume_valid=torch.tensor([[False]])).volume_valid.any()


def test_sensor_sampling_guard_and_mount_not_fixed():
    depth=torch.full((1,1,2,2),2.);k=torch.tensor([[[10.,0.,.5],[0.,10.,.5],[0.,0.,1.]]]);rot=torch.eye(3)[None,None];t=torch.tensor([[[-.4,.3,.2]]])
    f=frusta_from_d4xx(depth,k,rot,t,blocks=torch.tensor([[0,0,0,2,2]]),volume_valid=torch.tensor([[True]]))
    assert not f.volume_valid.any() # too sparse to resolve 5cm minimum object


def test_history_clones_reset_no_epoch_or_age_refresh():
    cache=FreeEvidenceHistory();f=prism();support=torch.zeros(1,1,1);pose=torch.zeros(1,4)
    cache.push(f,pose,support,torch.tensor([1000]),torch.tensor([0]));support.fill_(.5)
    assert cache.captures[0].support.item()==0
    with pytest.raises(ValueError):cache.push(f,pose,support,torch.tensor([1000]),torch.tensor([0]))
    cache.reset([0]);m=cache.measurement(pose,torch.tensor([1000]),torch.tensor([0]),torch.zeros(1,1),torch.zeros(1),torch.zeros(1))
    assert not proof(m).free.any()


def test_p2_budget_uses_path_slip_delay_and_matches_defaults():
    p,a=pose_budget(torch.tensor([1.]),torch.tensor([2.]),torch.tensor([.6]),torch.tensor([.5]),CFG)
    assert p.item()==pytest.approx(math.sqrt(2)*3*math.sqrt(.015**2+2*.005**2)+.03*2+.04*.6)
    assert a.item()==pytest.approx(math.radians(3*math.sqrt(.3**2+2*.2**2))+.04*.5)


def test_near_internal_proof_does_not_change_wire_unknown_contract():
    grid,_=extended_near_grid();h=torch.zeros(1,grid.ny,grid.nx)
    raw=RayBatch(torch.tensor([[[0.,0.,.15]]]),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[math.inf]]),torch.tensor([[False]]))
    f,state,support,age=build_near(raw,h,certified=True,return_proof=True)
    assert f.unknown.all() and (state==S.UNKNOWN).all() and support.all() and torch.isinf(age).all()


def test_future_duplicate_capture_and_nonunit_planes_rejected():
    with pytest.raises(ValueError,match='future'):proof(measurement([cap(t=1100)]))
    with pytest.raises(ValueError,match='unique'):proof(measurement([cap(),cap()]))
    f=prism();bad=replace(f,normals=f.normals*2)
    with pytest.raises(ValueError,match='unit'):proof(measurement([cap(bad)]))


def test_map_frame_age_provenance_and_packet_layout_unchanged():
    from gd_lab.gast_nav.map_types import NearFrame
    from gd_lab.gast_nav.packets import encode_obs1,decode_obs1,PACKET_BYTES
    grid=WIDE_GRID;h=torch.zeros(1,grid.ny,grid.nx)
    cfg=replace(CFG,historical_obstacle_speed_bound_mps=0.)
    m=measurement([cap(prism(x0=-20,x1=20,y0=-20,y1=20,z0=-10,z1=10),grid=grid)],grid=grid,now=1100,settings=cfg)
    mapper=ObstacleMap(1)
    raw=RayBatch(torch.tensor([[[0.,0.,.15]]]),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[math.inf]]),torch.tensor([[False]]))
    frame=mapper.update(raw,h,m.pose,m.now_ms,measurement=m,certified=True)
    known=frame.state==S.FREE
    assert known.any() and torch.allclose(frame.age_s[known],torch.full_like(frame.age_s[known],.1))
    near=NearFrame(torch.full((1,2,25,37),255,dtype=torch.uint8),torch.zeros(1,25,37,dtype=torch.bool),torch.full((1,25,37),.1))
    packet=encode_obs1(frame,near)
    decoded=decode_obs1(packet)
    assert len(packet)==PACKET_BYTES and torch.equal(decoded.wide.state,frame.state)
    assert torch.allclose(decoded.wide.age_s[known],frame.age_s[known])
    mapper.reset([0])
    with pytest.raises(ValueError,match='mismatch'):mapper.update(raw,h,m.pose,m.now_ms+100,measurement=m,certified=True)


def test_near_low_band_proof_independent_of_actor_b3_unknown():
    grid,_=extended_near_grid();h=torch.zeros(1,grid.ny,grid.nx)
    cfg=replace(CFG,historical_obstacle_speed_bound_mps=0.,sensor_angle_error_deg=0.)
    m=measurement([cap(prism(x0=-5,x1=5,y0=-5,y1=5,z1=1.0),grid=grid)],grid=grid,now=1100,settings=cfg)
    raw=RayBatch(torch.tensor([[[0.,0.,.15]]]),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[math.inf]]),torch.tensor([[False]]))
    frame,state,support,age=build_near(raw,h,measurement=m,certified=True,return_proof=True)
    assert (state[:,:2]==S.FREE).all() and (state[:,2]==S.UNKNOWN).all()
    assert frame.unknown.all() and support.all() and torch.allclose(age[:,:2],torch.full_like(age[:,:2],.1))


def test_bad_travel_and_unbounded_planes_rejected():
    m=measurement(pose=torch.tensor([[.01,0.,0.,0.]]))
    with pytest.raises(ValueError,match='travel'):proof(m)
    f=prism();normal=f.normals.clone();normal[:]=torch.tensor([1.,0.,0.])
    with pytest.raises(ValueError,match='planes'):proof(measurement([cap(replace(f,normals=normal))]))


def test_float64_pose_supported_without_changing_depth_dtype():
    m=measurement();m=replace(m,pose=m.pose.double(),captures=(replace(m.captures[0],pose=m.captures[0].pose.double()),))
    assert proof(m).free.all()


def test_far_range_config_cap_and_center_without_volume_guarantee():
    grid=Grid(1,1,.1,6.5,0.)
    m=measurement([cap(prism(x1=20.,y0=-20.,y1=20.,z1=10.),grid=grid)],grid=grid)
    assert not proof(m,grid=grid).free.any()
    assert not proof(measurement([cap(prism(valid=False))])).free.any()


def test_current_measurement_pose_time_binding():
    h=torch.zeros(1,WIDE_GRID.ny,WIDE_GRID.nx);m=measurement([cap(grid=WIDE_GRID)],grid=WIDE_GRID)
    raw=RayBatch(torch.tensor([[[0.,0.,.15]]]),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[math.inf]]),torch.tensor([[False]]))
    mapper=ObstacleMap(1)
    with pytest.raises(ValueError,match='mismatch'):mapper.update(raw,h,m.pose+torch.tensor([[.1,0.,0.,0.]]),m.now_ms,measurement=m)


def test_no_legacy_result_can_be_mislabelled_as_proof():
    grid,_=extended_near_grid();raw=RayBatch(torch.zeros(1,1,3),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[math.inf]]),torch.tensor([[False]]))
    with pytest.raises(ValueError,match='requires certified'):build_near(raw,torch.zeros(1,grid.ny,grid.nx),return_proof=True)


def test_expired_occupied_in_actual_map_never_becomes_free_certificate():
    h=torch.zeros(1,WIDE_GRID.ny,WIDE_GRID.nx);pose=torch.zeros(1,4);mapper=ObstacleMap(1)
    wall=RayBatch(torch.tensor([[[0.,0.,.15]]]),torch.tensor([[[1.,0.,0.]]]),torch.tensor([[.5]]),torch.tensor([[False]]))
    blind=replace(wall,distances=torch.tensor([[math.inf]]))
    a=mapper.update(wall,h,pose,torch.tensor([1000]),certified=True)
    assert (a.state==S.OCCUPIED).any()
    b=mapper.update(blind,h,pose,torch.tensor([1400]),certified=True)
    assert (b.state==S.OCCUPIED).any() and not (b.state==S.FREE).any()
    c=mapper.update(blind,h,pose,torch.tensor([1600]),certified=True)
    assert (c.state==S.UNKNOWN).all() and torch.isinf(c.age_s).all()


@pytest.mark.parametrize('is_self',[False,True])
def test_current_first_hit_trims_contradictory_full_frustum_shadow(is_self):
    grid=Grid(3,1,.1,.4,0.);support=torch.zeros(1,1,3)
    m=measurement([cap(grid=grid)],grid=grid)
    state,_=ray_evidence(rays(dist=.5,self_hit=is_self),support,grid=grid,measurement=m)
    assert (state[:,:,:,2]==S.UNKNOWN).all() # x=.6 beyond wall/self
    if not is_self:assert state[0,0,0,1]==S.OCCUPIED
    # The producer/cached source arrays are immutable.
    assert m.captures[0].frusta.far_m.item()==2.
