from dataclasses import replace
from pathlib import Path
import copy,sys,hashlib,pickle,math
import numpy as np
import torch,pytest
from gd_lab.gast_nav.terrain_tap177 import TerrainFrame177,tap_v25or177
from gd_lab.gast_nav.terrain_fusion177 import FusionConfig177,fuse_terrain177
from gd_lab.gast_nav.oracle_occgrid176 import *


@pytest.fixture
def native_v25or(monkeypatch):
    # Replace the producer's staging-only environment setup locally, per test.
    path=Path(__file__).resolve().parents[2]/'tools/rl_replay'
    monkeypatch.syspath_prepend(str(path))
    import structure130
    for key,value in dict(dual_grid=True,dual_or=True,d_deep=.7).items():
        monkeypatch.setitem(structure130.S,key,value)


def terrain(t=1000,n=1):
    shape=(n,11,17);z=torch.zeros(shape);valid=torch.ones(shape,dtype=torch.bool);false=torch.zeros_like(valid)
    return TerrainFrame177(z,valid,false.clone(),false.clone(),torch.full(shape,t,dtype=torch.long),torch.ones(shape,dtype=torch.uint8),torch.zeros((n,4),dtype=torch.float64),torch.zeros(n,dtype=torch.long),torch.full((n,),t,dtype=torch.long),torch.full((n,),t,dtype=torch.long))

def head(grid=None,state=FREE,t=1000,n=1,support=False,z=0.):
    grid=grid or OccConfig176();shape=(n,grid.ny,grid.nx);s=torch.full(shape,state,dtype=torch.uint8)
    stamp=torch.full(shape,t,dtype=torch.long)
    su=Support176(torch.full(shape,float(z)) if support else torch.full(shape,torch.nan),torch.full(shape,support,dtype=torch.bool),stamp.clone() if support else torch.full_like(stamp,-1))
    return OccFrame176(s,torch.where(s!=UNKNOWN,stamp,-1),su,torch.zeros(n,dtype=torch.long),torch.full((n,),t,dtype=torch.long))

def fuse(h,t,now=1000,grid=None,pose=None,segment=None,floor=None,**kw):
    n=len(t.epoch);pose=t.pose if pose is None else pose
    return fuse_terrain177(h,t,pose,torch.full((n,),now,dtype=torch.long),t.epoch,grid=grid,measured_current_support=torch.zeros(n) if floor is None else floor,segment_type=segment,**kw)

def cell(grid,x=0,y=0):return (int(math.floor((y-grid.y_min_m)/grid.cell_m+1e-6)),int(math.floor((x-grid.x_min_m)/grid.cell_m+1e-6)))

def test_near_support_fusion_fills_head_missing_without_making_obstacle_free():
    t=terrain();h=head();e=fuse(h,t);c=OccConfig176();yx=cell(c)
    assert e.frame.state[0,*yx]==FREE and e.foot_support.observed[0,*yx]
    assert e.frame.seen_ms[0,*yx]==1000 and e.foot_support.seen_ms[0,*yx]==1000
    # Outside local +/- .85,.55, head support was missing and stays unknown.
    assert e.frame.state[0,*cell(c,1.2)]==UNKNOWN
    assert not (fuse(head(state=UNKNOWN),t).frame.state==FREE).any()
    assert torch.isnan(h.support.reference_z).all() # input unchanged


def test_gap_edge_vetoes_head_free_even_when_head_has_floor_support():
    t=terrain();t.gap[:,:,11]=True;t.source[:,:,11]=4;t.surface_z[:,:,11]=-.7
    e=fuse(head(support=True),t);c=OccConfig176()
    assert e.gap.any()
    assert (e.terrain_source_bits[e.gap]==16).all()
    assert (e.frame.state[e.gap]==UNKNOWN).all() and not e.foot_support.observed[e.gap].any()
    assert e.frame.state[0,*cell(c,.2)]==FREE
    assert e.frame.state[0,*cell(c,.3)]==UNKNOWN
    # Native held gap still vetoes without being relabelled as a fresh support.
    old=replace(t,seen_ms=t.seen_ms.clone());old.seen_ms[old.gap]=0
    e=fuse(head(support=True),old)
    assert e.gap.any() and (e.terrain_seen_ms[e.gap]==0).all()


@pytest.mark.parametrize('height',[.04,.08,.20,.45])
def test_low_box_under_body_is_occupied_on_flat(height):
    t=terrain();t.surface_z[:,4:7,7:10]=height
    e=fuse(head(support=True),t);c=OccConfig176();yx=cell(c)
    assert e.frame.state[0,*yx]==OCCUPIED
    assert not e.foot_support.observed[0,*yx]
    assert e.frame.support.reference_z[0,*yx]==0 # not box-top reference
    # If self mask hid it, no point/height is invented from scene geometry.
    t.valid[:,4:7,7:10]=False
    e=fuse(head(support=True),t)
    assert e.frame.state[0,*yx]==UNKNOWN


@pytest.mark.parametrize('direction',[1,-1])
def test_connected_stair_treads_accept_step_exception(direction):
    t=terrain();t.surface_z[:,:,:,] = 0
    t.surface_z[:,:,10:13]=direction*.2;t.surface_z[:,:,13:]=direction*.4
    e=fuse(head(),t,segment='STAIR_UP' if direction>0 else 'STAIR_DOWN');c=OccConfig176()
    assert e.frame.state[0,*cell(c,.3)]==FREE
    assert e.frame.state[0,*cell(c,.6)]==FREE
    assert abs(float(e.foot_support.reference_z[0,*cell(c,.6)])-direction*.4)<1e-5
    assert not e.terrain_occupied.any() and not e.gap.any()
    flat=fuse(head(),t,segment='FLAT')
    assert flat.frame.state[0,*cell(c,.3)]!=FREE


def test_disconnected_high_platform_not_seeded_as_stair_floor():
    t=terrain();t.surface_z[:,:,10:]=.45
    e=fuse(head(),t,segment='STAIR_UP')
    assert e.terrain_occupied.any() and not e.foot_support.observed[e.terrain_occupied].any()


def test_support_ttl_original_observation_and_cached_result():
    t=terrain();t=replace(t,sample_ms=torch.tensor([1400]))
    h=head(t=1400);e=fuse(h,t,now=1400)
    c=OccConfig176();req=torch.zeros_like(h.state,dtype=torch.bool);req[0,*cell(c)]=True
    assert e.foot_support.seen_ms[0,*cell(c)]==1000
    assert certify176(e.frame,req,torch.tensor([1500]),e.cert_cfg,required_outside=torch.tensor([False]))[0].all()
    assert not certify176(e.frame,req,torch.tensor([1501]),e.cert_cfg,required_outside=torch.tensor([False]))[0].any()
    e=fuse(h,t,now=1501)
    assert e.frame.state[0,*cell(c)]==UNKNOWN
    with pytest.raises(ValueError):FusionConfig177(support_ttl_s=2.)


def test_conflict_and_existing_occupied_have_priority():
    t=terrain();h=head(support=True,z=.05);e=fuse(h,t)
    assert e.support_conflict.any() and (e.frame.state[e.support_conflict]==UNKNOWN).all()
    h=head(state=OCCUPIED,support=True,z=.13)
    e=fuse(h,t)
    assert (e.frame.state==OCCUPIED).all() and (e.frame.support.reference_z==.13).all()
    assert torch.equal(e.frame.seen_ms,h.seen_ms)


def test_coordinate_rotation_translation_base_z_and_005_grid():
    t=terrain();t.gap[:,:,11]=True;t.source[:,:,11]=4;t.surface_z[:,:,11]=-.7
    h=head(support=True);pose=t.pose.clone();pose[:,0]=.1;pose[:,2]=.2;pose[:,3]=math.pi/2
    # Fresh independent head floor support is in the CURRENT base-z frame.
    h=replace(h,support=replace(h.support,reference_z=torch.full_like(h.support.reference_z,-.2)))
    e=fuse(h,t,pose=pose,floor=torch.zeros(1))
    assert e.gap.any();c=OccConfig176()
    assert e.frame.state[0,*cell(c,0,-.2)]==UNKNOWN
    good=e.foot_support.observed
    assert torch.allclose(e.foot_support.reference_z[good],torch.full_like(e.foot_support.reference_z[good],-.2))
    c=OccConfig176(cell_m=.05);e=fuse(head(c),terrain(),grid=c)
    assert e.frame.state.shape==(1,c.ny,c.nx) and e.foot_support.observed.any()
    # Partial ROI border never extends the terrain support beyond its extent.
    assert not e.foot_support.observed[0,*cell(c,1.,0)]


def test_epoch_unknown_pending_missing_and_future_fail_closed():
    t=terrain();t.pending[:,5,8]=True;t.valid[:,5,8]=False;t.source[:,5,8]=5
    e=fuse(head(support=True),t);c=OccConfig176();assert e.frame.state[0,*cell(c)]==UNKNOWN
    stale=replace(t,epoch=torch.ones(1,dtype=torch.long))
    assert not (fuse(head(),stale).frame.state==FREE).any()
    with pytest.raises(ValueError,match='future'):fuse(head(),replace(t,sample_ms=torch.tensor([1001])))
    with pytest.raises(ValueError,match='measured floor'):fuse(head(),terrain(),floor=torch.tensor([torch.nan]))
    with pytest.raises(ValueError):fuse(head(),terrain(),segment='INVALID')


def test_legacy_sampler_tap_uses_exact_selected_donor_and_is_read_only(native_v25or):
    # Imports contain CPU mapper definitions only; never instantiate Isaac/Lidar.
    path=Path('/home/user/gd_project/archive/gd_lab_vrl_worktrees/gast_nav_v1/tools/rl_replay');sys.path.insert(0,str(path))
    from lidar126 import Maps
    from structure130 import S
    m=Maps(1,'L0S');raw=m.direct.maps[0];raw.recenter(np.zeros(2))
    raw.z[:]=0.;raw.var[:]=0.;raw.last[:]=1.;raw.finite[:]=1.;raw.last_capture=1.
    # Unique deeper direct subcell under body0, force h0d to choose a sub-point.
    ij,inside=raw.indices(np.array([[-.025,.025]]));ix,iy=ij[0]
    raw.z[iy,ix]=-.2;raw.last[iy,ix]=.8;raw.finite[iy,ix]=.8
    pose=np.zeros((1,4));scanner=np.array([.5]);packed=m.sample(pose[:,[0,1,3]],scanner,1.2)
    before=pickle.dumps(m);original=packed.copy();rng=np.random.get_state()
    t=tap_v25or177(m,packed,pose,scanner,sample_ms=np.array([1200],np.int64),epoch=np.array([0],np.int64),structure_rules=S)
    assert t.valid[0,5,8] and abs(float(t.surface_z[0,5,8])+.2)<1e-5
    assert t.seen_ms[0,5,8]==800 and t.last_capture_ms[0]==1000
    assert pickle.dumps(m)==before and np.array_equal(packed,original)
    assert all(np.array_equal(a,b) for a,b in zip(rng,np.random.get_state()))
    # Structural GAP at the same query cell has a synthetic depth, not support.
    mem=m.struct[0];key=tuple(np.floor(np.array([.3,0])/mem.res+1e-7).astype(int))
    mem.gap[key]=(0.,.2,.5,1.)
    packed=m.sample(pose[:,[0,1,3]],scanner,1.2)
    before=pickle.dumps(m)
    t=tap_v25or177(m,packed,pose,scanner,sample_ms=np.array([1200],np.int64),epoch=np.array([0],np.int64),structure_rules=S)
    assert t.gap[0,5,11] and t.source[0,5,11]==4
    assert t.valid[0,5,11] and abs(float(t.surface_z[0,5,11])+.7)<1e-5
    assert t.seen_ms[0,5,11]==1000 and pickle.dumps(m)==before


def test_head_gap_and_no_unobserved_support_or_interpolation():
    t=terrain();h=head(support=True);gap=torch.zeros_like(h.state,dtype=torch.bool)
    c=OccConfig176();gap[0,*cell(c)]=True
    e=fuse(h,t,head_gap_mask=gap)
    assert e.gap[0,*cell(c)] and e.frame.state[0,*cell(c)]==UNKNOWN
    t.valid[:]=False;t.valid[:,5,8]=True
    e=fuse(h,t) # isolated cell cannot produce a measured plane normal
    assert not e.foot_support.observed[e.terrain_roi].any()
    # Explicit contact seed does not fill a missing height.
    t.valid[:]=False
    e=fuse(h,t,contact_seed_mask=torch.ones_like(t.valid))
    assert not e.foot_support.observed[e.terrain_roi].any()


def test_high_surface_does_not_become_low_obstacle_or_create_head_free():
    t=terrain();t.surface_z[:,4:7,7:10]=.8
    e=fuse(head(),t);c=OccConfig176()
    assert e.frame.state[0,*cell(c)]==UNKNOWN and not e.terrain_occupied[0,*cell(c)]
    # Head unknown cannot become free even with measured clean terrain across ROI.
    t=terrain();e=fuse(head(state=UNKNOWN),t)
    assert e.foot_support.observed.any() and not (e.frame.state==FREE).any()


def test_memory_obstacle_three_seconds_does_not_extend_foot_support():
    # New head capture elsewhere, old free in this region: genuine memory case.
    h=replace(head(t=1000),capture_ms=torch.tensor([1700]))
    t=terrain(t=1700);cfg=FusionConfig177(obstacle_ttl_s=3.)
    e=fuse(h,t,now=1800,config=cfg);c=OccConfig176()
    req=torch.zeros_like(h.state,dtype=torch.bool);req[0,*cell(c)]=True
    assert e.frame.seen_ms[0,*cell(c)]==1000
    assert e.frame.support.seen_ms[0,*cell(c)]==1700
    assert e.head_capture_ms[0]==1700
    assert certify176(e.frame,req,torch.tensor([1800]),e.cert_cfg,required_outside=torch.tensor([False]))[0].all()
    assert not certify176(e.frame,req,torch.tensor([2201]),e.cert_cfg,required_outside=torch.tensor([False]))[0].any()


def test_batched_modes_and_mismatched_epoch_do_not_mix_envs():
    from gd_lab.gast_nav.schema import SEGMENT_TYPES
    t=terrain(n=3);t.surface_z[:,:,10:]=.2
    h=head(n=3);segment=torch.tensor([SEGMENT_TYPES.index('FLAT'),SEGMENT_TYPES.index('STAIR_UP'),SEGMENT_TYPES.index('STAIR_UP')])
    # The third terrain frame belongs to an old episode.
    t=replace(t,epoch=torch.tensor([0,0,1]))
    e=fuse_terrain177(h,t,t.pose,torch.tensor([1000]*3),torch.tensor([0]*3),measured_current_support=torch.zeros(3),segment_type=segment,config=FusionConfig177(chunk_envs=2))
    c=OccConfig176();yx=cell(c,.3)
    assert e.frame.state[0,*yx]==OCCUPIED
    assert e.frame.state[1,*yx]==FREE
    assert e.frame.state[2,*yx]==UNKNOWN


def test_shipped_config_matches_constructor_contract():
    assert FusionConfig177.load()==replace(FusionConfig177(),terrain_limits177b=True)


@pytest.mark.parametrize('world_z',[-2.,1.2])
def test_native_clipped_height_tap_recovers_metric_donor(world_z, native_v25or):
    path=Path('/home/user/gd_project/archive/gd_lab_vrl_worktrees/gast_nav_v1/tools/rl_replay');sys.path.insert(0,str(path))
    from lidar126 import Maps
    from structure130 import S
    m=Maps(1,'L0S');raw=m.direct.maps[0];raw.recenter(np.zeros(2))
    raw.z[:]=world_z;raw.var[:]=0.;raw.last[:]=1.;raw.finite[:]=1.;raw.last_capture=1.
    pose=np.zeros((1,4));scanner=np.array([.5])
    packed=m.sample(pose[:,[0,1,3]],scanner,1.2)
    assert np.all(np.abs(packed[0,:187])==5.) # inversion alone is ambiguous
    t=tap_v25or177(m,packed,pose,scanner,sample_ms=np.array([1200],np.int64),epoch=np.array([0],np.int64),structure_rules=S)
    assert t.valid.all() and torch.allclose(t.surface_z,torch.full_like(t.surface_z,world_z))
    assert (t.seen_ms==1000).all()
