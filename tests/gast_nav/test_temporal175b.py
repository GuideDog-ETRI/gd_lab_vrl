from dataclasses import replace
import copy,math
import pytest,torch
from gd_lab.gast_nav.temporal_map import TemporalMap,load_settings,select_rear_source
from gd_lab.gast_nav.temporal_map.sensors import HeadSensors
from gd_lab.gast_nav.temporal_map.manifest import effective_manifest
from gd_lab.gast_nav.temporal_map.oracle import HeadOracle175B,rasterize_head175b,measured_shadow175b,yaw_rotation
from gd_lab.gast_nav.oracle_occgrid176 import *
from gd_lab.gast_nav.producer168 import SensorPrimitives


def config():return OccConfig176(x_min_m=-1.05,x_max_m=1.05,y_min_m=-1.05,y_max_m=1.05)
def settings(ttl=3):
    s=load_settings();s['memory_free_ttl_s']=ttl;s['world_cells']=[41,41];return s

def make(n=1,ttl=3,device='cpu'):
    c=config();m=TemporalMap(n,settings(ttl),device=device,output_cfg=c)
    p=torch.zeros((n,4),device=device,dtype=torch.float64)
    return m,p,c

def packet(c,t=1000,n=1,state=FREE,z=0.,support=True,device='cpu',epoch=0):
    sh=(n,c.ny,c.nx);d=device
    st=torch.full(sh,state,dtype=torch.uint8,device=d);stamp=torch.full(sh,t,dtype=torch.long,device=d)
    su=Support176(torch.full(sh,float(z),device=d),torch.full(sh,support,dtype=torch.bool,device=d),stamp.clone())
    f=OccFrame176(st,torch.where(st!=UNKNOWN,stamp,-1),su,torch.full((n,),epoch,device=d,dtype=torch.long),torch.full((n,),t,device=d,dtype=torch.long))
    return f,torch.where(st!=UNKNOWN,1,0).to(torch.uint8)

def push(m,p,c,t=1000,**kw):
    f,b=packet(c,t,n=m.n,device=m.device,**kw);m.push(f,p,b,profile_hash='a'*64,occluded_mask=torch.zeros_like(f.state,dtype=torch.bool));return f

def query(m,p,t,epoch=0,**kw):return m.query(p,torch.full((m.n,),t,device=m.device,dtype=torch.long),torch.full((m.n,),epoch,device=m.device,dtype=torch.long),**kw)
def center(a):return a[:,a.shape[1]//2,a.shape[2]//2]


def test_initial_memory_empty_no_prior():
    m,p,c=make();e=query(m,p,0)
    assert not e.obstacle_state.any() and not e.foot_support.observed.any()
    assert not e.frame.support.observed.any()


@pytest.mark.parametrize('ttl',[.5,1.,2.,3.])
def test_obstacle_ttl_independent_foot_half_second_including_cached_frame(ttl):
    m,p,c=make(ttl=ttl);push(m,p,c)
    e=query(m,p,1000)
    req=torch.zeros_like(e.frame.state,dtype=torch.bool);req[:,c.ny//2,c.nx//2]=True
    assert certify176(e.frame,req,torch.tensor([1500]),e.cert_cfg,required_outside=torch.tensor([False]))[0].all()
    assert not certify176(e.frame,req,torch.tensor([1501]),e.cert_cfg,required_outside=torch.tensor([False]))[0].any()
    live=query(m,p,1000+int(ttl*1000));dead=query(m,p,1001+int(ttl*1000))
    assert center(live.obstacle_state)==FREE and center(dead.obstacle_state)==UNKNOWN
    assert not query(m,p,1501).foot_support.observed.any()
    # Occupied uses exactly the same obstacle comparison TTL, not foot TTL.
    mm,pp,cc=make(ttl=ttl);push(mm,pp,cc,state=OCCUPIED)
    assert center(query(mm,pp,1000+int(ttl*1000)).obstacle_state)==OCCUPIED
    assert center(query(mm,pp,1001+int(ttl*1000)).obstacle_state)==UNKNOWN


def test_world_translation_rotation_z_and_source_time_provenance():
    m,p,c=make();t=10**12;f,b=packet(c,t);b.fill_(2);m.push(f,p,b,profile_hash='a'*64)
    # Entire source footprint conservative: world rotation cannot invent free.
    pp=p.clone();pp[:,0]=.6;pp[:,3]=math.pi/4;pp[:,2]=.4
    e=query(m,pp,t+100);mask=e.obstacle_state==FREE
    assert mask.any() and not mask.all()
    assert (e.frame.seen_ms[mask]==t).all() and (e.source_bits[mask]==2).all()
    assert torch.allclose(e.frame.support.reference_z[mask],torch.full_like(e.frame.support.reference_z[mask],-.4))
    assert (e.capture_id[mask]==0).all() and e.current_free[mask].all()
    # Next capture only unknown; old observations become memory, not refreshed.
    push(m,pp,c,t=t+200,state=UNKNOWN,support=False)
    e=query(m,pp,t+300);mask=e.obstacle_state==FREE
    assert mask.any() and e.memory_free[mask].all() and not e.current_free.any()
    assert (e.frame.seen_ms[mask]==t).all()


def test_bounded_recentering_no_wrap_and_epoch_reset():
    m,p,c=make();push(m,p,c)
    far=p.clone();far[:,0]=10;push(m,far,c,t=1100,state=UNKNOWN,support=False)
    assert not query(m,far,1100).obstacle_state.any()
    push(m,p,c,t=1200,state=UNKNOWN,support=False,epoch=1)
    assert not query(m,p,1200,epoch=1).obstacle_state.any()
    with pytest.raises(ValueError,match='non-increasing'):push(m,p,c,t=1200,epoch=1)


def test_stairs_support_change_invalidates_old_free_preserves_occupied_classification():
    m,p,c=make();f,b=packet(c);f.state[:,c.ny//2,c.nx//2]=OCCUPIED;m.push(f,p,b,profile_hash='a'*64)
    # A newly observed tread changes support, but no new obstacle measurement.
    push(m,p,c,t=1100,state=UNKNOWN,z=.2,support=True)
    e=query(m,p,1100)
    assert center(e.obstacle_state)==OCCUPIED
    assert center(e.frame.support.reference_z)==0 # historical classification unchanged
    assert not center(e.frame.support.observed) # cannot assert old basis == fresh foot
    assert abs(float(center(e.foot_support.reference_z))-.2)<1e-6
    assert not (e.obstacle_state==FREE).any()
    # Fresh free cannot delete occupied before obstacle TTL.
    push(m,p,c,t=1200,state=FREE,z=.2)
    assert center(query(m,p,1200).obstacle_state)==OCCUPIED
    push(m,p,c,t=4101,state=FREE,z=.2)
    assert center(query(m,p,4101).obstacle_state)==FREE


def test_person_enters_memory_visible_shadow_and_hidden_dynamic_context():
    m,p,c=make();push(m,p,c)
    f,b=packet(c,t=1100,state=UNKNOWN,support=True)
    iy=c.ny//2;ix=c.nx//2
    f.state[:,iy,ix]=OCCUPIED;f.seen_ms[:,iy,ix]=1100;b[:,iy,ix]=2
    shadow=torch.zeros_like(f.state,dtype=torch.bool);shadow[:,iy,ix:]=True
    m.push(f,p,b,profile_hash='b'*64,occluded_mask=shadow)
    e=query(m,p,1100)
    assert center(e.obstacle_state)==OCCUPIED
    assert (e.obstacle_state[:,iy,ix+1:]==UNKNOWN).all()
    # Unobserved rear intrusion is impossible to detect: explicit dynamic mode
    # must fail closed for remembered free, never consult GT person position.
    e=query(m,p,1100,dynamic_context=True)
    assert e.memory_free.any() and not (e.frame.state[e.memory_free]==FREE).any()


def test_pose_and_provenance_fail_closed():
    m,p,c=make();f,b=packet(c);f.seen_ms.fill_(1001)
    with pytest.raises(ValueError,match='provenance'):m.push(f,p,b,profile_hash='a'*64)
    with pytest.raises(ValueError,match='GT pose'):query(m,p[:,:3],1000)
    for key,value in [('support_ttl_s',3),('pose_level','P2'),('initial_prior_for_certification',True)]:
        s=settings();s[key]=value
        with pytest.raises(ValueError):TemporalMap(1,s,output_cfg=c)
    with pytest.raises(ValueError,match='unavailable'):select_rear_source(settings())
    token=object();assert select_rear_source(settings(),temporal_map=token) is token


def test_native_pattern_mount_and_manifest():
    s=settings();h=HeadSensors(s);manifest=effective_manifest(s)
    assert len(manifest['sha256'])==64
    _,d,_,near,src=h.directions('ouster');assert d.shape==(32768,3) and near==.4 and src==1
    _,mid,_,near,src=h.directions('mid360');assert mid.shape==(20000,3) and src==2
    el=torch.asin(mid[:,2]);assert float(el.min())>=math.radians(-7.21) and float(el.max())<=math.radians(52.21)
    assert not torch.equal(mid,h.directions('mid360',capture_index=1)[1])
    assert torch.equal(mid,h.directions('mid360',capture_index=40)[1])
    keep=h.coverage.keep_directions('mid360',mid);az=torch.atan2(mid[:,1],mid[:,0])
    deg=az.abs()*180/math.pi
    el_deg=torch.rad2deg(torch.atan2(mid[:,2],mid[:,:2].norm(dim=-1)))
    expected=(deg<=105)&(el_deg>=-7)&(el_deg<=52)
    away=(deg-105).abs()>.0001
    assert torch.equal(keep[away],expected[away])
    assert h.coverage.mid_max_range==1.5
    assert h.directions('d455')[1].shape==(921600,3)
    r,t=h.transforms['livox_frame'];assert r[2,0]<-.09 # cumulative 0.74 +4.81 down
    assert .25<float(t[0])<.3


def test_oracle_valid_nohit_selfhit_near_preserved_no_rng():
    s=settings();s['sensor_sim']['d455_resolution']=[8,6];h=HeadSensors(s)
    state=torch.random.get_rng_state().clone()
    def nohit(o,d,far):return torch.full(o.shape[:2],torch.inf),torch.zeros(o.shape[:2],dtype=torch.bool),torch.ones(o.shape[:2],dtype=torch.bool)
    samples=list(h.capture('d455',nohit,n=1));assert samples[0].valid.all() and torch.isinf(samples[0].distances).all()
    assert torch.equal(state,torch.random.get_rng_state())
    def selfhit(o,d,far):return torch.full(o.shape[:2],.01),torch.ones(o.shape[:2],dtype=torch.bool),torch.ones(o.shape[:2],dtype=torch.bool)
    a=list(h.capture('d455',selfhit,n=1))[0];assert not a.valid.any() and a.self_hit.all() and (a.distances==.01).all()
    with pytest.raises(ValueError,match='explicit'):list(h.capture('d455',lambda o,d,f:nohit(o,d,f)[:2],n=1))
    bad=copy.deepcopy(s);bad['sensor_sim']['range_noise_std_m']=.01
    with pytest.raises(ValueError,match='no noise'):HeadSensors(bad)


def test_end_to_end_actual_head_rays_first_hit_shadow_no_support_prior():
    s=settings();s['sensor_sim']['d455_resolution']=[8,6];adapter=HeadOracle175B(s);c=config()
    # Scene fixture only: floor at -0.55 base-relative and a front thin wall.
    centres=torch.tensor([[[0,0,-.6],[.8,0,0]]],dtype=torch.float32)
    half=torch.tensor([[[4,4,.05],[.025,1,.5]]],dtype=torch.float32)
    scene=SensorPrimitives(centres,torch.eye(3)[None,None].expand(1,2,-1,-1).clone(),half,torch.zeros((1,2),dtype=torch.long),torch.ones((1,2),dtype=torch.bool),torch.tensor([True]))
    p=torch.zeros((1,4));ids=torch.tensor([0]);su=Support176(torch.full((1,c.ny,c.nx),-.55),torch.ones((1,c.ny,c.nx),dtype=torch.bool),torch.full((1,c.ny,c.nx),1000,dtype=torch.long))
    kwargs=dict(capture_id=ids,scene_capture_id=ids,self_mask=torch.zeros((1,2),dtype=torch.bool),capture_index=0,sensors=('d455',))
    stream=adapter.returns(scene,torch.eye(3)[None],p,**kwargs)
    f,b,shadow=rasterize_head175b(stream,su,c,capture_ms=torch.tensor([1000]),epoch=ids)
    assert (f.state==OCCUPIED).any() and (b[f.state!=UNKNOWN]==4).all() and shadow.any()
    # Incomplete scene inventory never creates free/occupied.
    scene=replace(scene,complete=torch.tensor([False]))
    f,_,_=rasterize_head175b(adapter.returns(scene,torch.eye(3)[None],p,**kwargs),su,c,capture_ms=torch.tensor([1000]),epoch=ids)
    assert not f.state.any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_cpu_parity():
    outputs=[]
    for dev in ('cpu','cuda'):
        m,p,c=make(n=2,device=dev);push(m,p,c)
        pp=p.clone();pp[:,0]=.2;pp[:,3]=math.pi/3
        push(m,pp,c,t=1200,state=UNKNOWN,support=False)
        e=query(m,pp,1300)
        outputs.append((e.obstacle_state.cpu(),e.frame.seen_ms.cpu(),e.frame.support.observed.cpu(),e.frame.support.reference_z.cpu()))
    for a,b in zip(outputs[0],outputs[1]):
        if a.is_floating_point():assert torch.allclose(a,b,equal_nan=True,atol=1e-6)
        else:assert torch.equal(a,b)


def test_old_free_with_fresh_measured_foot_can_certify_without_refreshing_obstacle():
    m,p,c=make(ttl=3);push(m,p,c)
    push(m,p,c,t=1700,state=UNKNOWN,support=True)
    e=query(m,p,1800)
    req=torch.zeros_like(e.frame.state,dtype=torch.bool);req[:,c.ny//2,c.nx//2]=True
    assert center(e.memory_free) and center(e.frame.seen_ms)==1000
    assert center(e.foot_support.seen_ms)==1700
    assert certify176(e.frame,req,torch.tensor([1800]),e.cert_cfg,required_outside=torch.tensor([False]))[0].all()
    # Cached frame's support still expires relative to the REAL 1700 observation.
    assert not certify176(e.frame,req,torch.tensor([2201]),e.cert_cfg,required_outside=torch.tensor([False]))[0].any()


def test_rotation_thin_occupied_is_never_turned_into_free():
    m,p,c=make();f,b=packet(c,state=UNKNOWN,support=False)
    f.state[:,c.ny//2,c.nx//2]=OCCUPIED;f.seen_ms[:,c.ny//2,c.nx//2]=1000;b[:,c.ny//2,c.nx//2]=4
    m.push(f,p,b,profile_hash='c'*64)
    pp=p.clone();pp[:,3]=math.pi/4;pp[:,0]=.037
    e=query(m,pp,1100)
    assert (e.obstacle_state==OCCUPIED).any() and not (e.obstacle_state==FREE).any()
    occupied=e.obstacle_state==OCCUPIED
    assert (e.source_bits[occupied]==4).all() and (e.profile_index[occupied]==0).all()


def test_shadow_near_self_hit_retracts_memory_even_before_near_cut():
    c=config();sh=(1,c.ny,c.nx);near=torch.tensor([[.15]])
    r=OracleRays176(torch.tensor([[[0.,0.,.2]]]),torch.tensor([[[1.,0.,0.]]]),near,torch.tensor([[1.]]),torch.tensor([[True]]),torch.tensor([[4]]))
    ret=OracleReturns176(r,torch.tensor([[.1]]),torch.tensor([[False]]),torch.tensor([[True]]))
    su=Support176(torch.zeros(sh),torch.ones(sh,dtype=torch.bool),torch.full(sh,1000,dtype=torch.long))
    mask=measured_shadow175b(ret,su,c,capture_ms=torch.tensor([1000]),epoch=torch.tensor([0]))
    assert mask[0,c.ny//2,c.nx//2+5] # no free behind self, even hit<near
    assert not rasterize176(ret,su,c,capture_ms=torch.tensor([1000]),epoch=torch.tensor([0])).state.any()


def test_head_filters_share_update176_native_ray_contract():
    from gd_lab.gast_nav.oracle_patterns176 import mounted_rays176,load_gdm_profile176
    s=settings();h=HeadSensors(s);g=s['gdm']
    profile=load_gdm_profile176(g['trav_rbq'],g['sensor_configuration'],effective_parameters=dict(ouster_min_range_m=.4,ouster_max_range_m=6.,livox_min_range=.05,livox_max_range=1.5,livox_user_sector_deg=150.,livox_vertical_fov_deg=s['sensor_sim']['livox_vertical_fov_deg']))
    for sensor,src in [('ouster',0),('mid360',1)]:
        frame,d,zoff,near,_=h.directions(sensor)
        rings=torch.arange(32,64).repeat(1024) if sensor=='ouster' else torch.full((len(d),),-1)
        r,t=h.transforms[frame];t=t+r[:,2]*zoff
        rays,_=mounted_rays176(d,torch.full((len(d),),src),rings,torch.zeros(len(d),dtype=torch.long),r[None,None],t[None,None],profile)
        own=h.coverage.keep_directions(sensor,d)
        # Floating point boundary differences may only be <1e-4 degrees.
        assert (rays.coverage_valid[0]^own).sum()<=2
        assert (rays.near_m==near).all()
        assert (rays.far_m[rays.coverage_valid]==(6. if src==0 else 1.5)).all()


def test_head_screen_door_clips_body_x_without_local_direction_crop():
    s=settings();adapter=HeadOracle175B(s)
    # Geometric fixture uses a backward-slanted but allowed +100deg Mid beam.
    adapter.sensors.pattern=torch.tensor([[math.radians(100)],[0.]])
    # Same factory semantics, one small bounded chunk for this fixture.
    adapter.sensors.sim=dict(adapter.sensors.sim,ray_chunk=20000)
    scene=SensorPrimitives(torch.tensor([[[10.,10.,10.]]]),torch.eye(3)[None,None],torch.tensor([[[.1,.1,.1]]]),torch.zeros((1,1),dtype=torch.long),torch.ones((1,1),dtype=torch.bool),torch.tensor([True]))
    ids=torch.tensor([0]);p=torch.zeros((1,4))
    ret,_=next(adapter.returns(scene,torch.eye(3)[None],p,capture_id=ids,scene_capture_id=ids,self_mask=torch.zeros((1,1),dtype=torch.bool),capture_index=0,screen_door=True,sensors=('mid360',)))
    assert ret.valid.any() # negative direction x is not blanket removed
    endpoints=ret.rays.origins+ret.rays.directions*ret.rays.far_m[...,None]
    assert (endpoints[...,0]>=-1e-6).all()


def test_missing_visibility_cleanup_does_not_reuse_old_free():
    m,p,c=make();push(m,p,c)
    f,b=packet(c,t=1100,state=UNKNOWN,support=True)
    m.push(f,p,b,profile_hash='a'*64) # no explicit current shadow result
    assert not (query(m,p,1100).obstacle_state==FREE).any()
