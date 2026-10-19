# UPDATE179: upstream176 fixtures against live gast_nav, local read-only calibration paths.
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from gd_lab.gast_nav.oracle_occgrid176 import *
from gd_lab.gast_nav.producer168 import SensorPrimitives
ROOT=Path(__file__).resolve().parents[2]

def cfg(res=.1):return OccConfig176(cell_m=res,x_min_m=-1.,x_max_m=2.,y_min_m=-1.,y_max_m=1.)

def rays(origins=None,directions=None,device='cpu',sources=None):
    origins=origins or [[-.8,0.,.15]];directions=directions or [[1.,0,0]]
    o=torch.tensor(origins,device=device,dtype=torch.float32)[None];d=torch.tensor(directions,device=device,dtype=torch.float32)[None]
    return OracleRays176(o,d,torch.zeros(o.shape[:2],device=device),torch.full(o.shape[:2],3.,device=device),torch.ones(o.shape[:2],device=device,dtype=torch.bool),torch.tensor(sources or [2]*len(origins),device=device)[None])

def scene(boxes,device='cpu'):
    p=len(boxes);t=lambda x:torch.tensor(x,dtype=torch.float32,device=device)[None]
    return SensorPrimitives(t([v[0] for v in boxes]),torch.eye(3,device=device)[None,None].expand(1,p,-1,-1).clone(),t([v[1] for v in boxes]),torch.zeros((1,p),dtype=torch.long,device=device),torch.ones((1,p),dtype=torch.bool,device=device),torch.ones(1,dtype=torch.bool,device=device))

def hits(g,r,**kw):
    ids=torch.zeros(1,device=r.origins.device,dtype=torch.long)
    return oracle_first_hits176(g,r,scene_capture_id=ids,ray_capture_id=ids,**kw)

def support(c,device='cpu',z=0.,time=0):
    shape=(1,c.ny,c.nx)
    # Fixture explicit measured support. Production must use observed_support176.
    return Support176(torch.full(shape,z,device=device),torch.ones(shape,device=device,dtype=torch.bool),torch.full(shape,time,device=device,dtype=torch.long))

def frame(h,c,s=None,t=0,epoch=0):
    return rasterize176(h,s or support(c,h.rays.origins.device),c,capture_ms=torch.tensor([t],device=h.rays.origins.device),epoch=torch.tensor([epoch],device=h.rays.origins.device))

def cell(c,x,y=0):return int(math.floor((y-c.y_min_m)/c.cell_m+1e-5)),int(math.floor((x-c.x_min_m)/c.cell_m+1e-5))

def test_thin_pillar_smaller_than_cell_010_005():
    for res in (.1,.05):
        c=cfg(res);g=scene([([.51,0,.2],[.01,.01,.2])]);h=hits(g,rays())
        assert abs(float(h.distance_m[0,0]) - 1.30)<1e-5
        f=frame(h,c);y,x=cell(c,.505)
        assert f.state[0,y,x]==OCCUPIED
        assert f.state[0,cell(c,0)[0],cell(c,0)[1]]==FREE
        assert f.state[0,cell(c,1.)[0],cell(c,1.)[1]]==UNKNOWN


def test_wall_occludes_person_and_coverage_mask():
    c=cfg();g=scene([([0.,0,.5],[.02,.5,.5]),([.8,0,.4],[.2,.2,.4])]);r=rays()
    h=hits(g,r);assert float(h.distance_m[0,0])<.81
    f=frame(h,c);assert f.state[0,*cell(c,.8)]==UNKNOWN
    masked=frame(hits(g,replace(r,coverage_valid=torch.zeros_like(r.coverage_valid))),c)
    assert not (masked.state!=UNKNOWN).any()


def test_self_hit_before_near_no_free_behind():
    c=cfg();g=scene([([-.7,0,.15],[.02,.02,.02]),([.8,0,.5],[.1,.5,.5])]);r=rays()
    h=hits(g,replace(r,near_m=torch.full_like(r.near_m,.2)),self_mask=torch.tensor([[True,False]]))
    assert not h.valid.any();assert not (frame(h,c).state!=UNKNOWN).any()


def test_height_filter_and_nearest_high_hit_stops_ray():
    c=cfg();r=rays(origins=[[-.8,0.,.8]])
    h=hits(scene([([0.,0.,.8],[.02,.2,.05]),([1.,0.,.5],[.1,.2,.5])]),r)
    f=frame(h,c)
    assert not (f.state==OCCUPIED).any()
    assert f.state[0,*cell(c,1.)]==UNKNOWN
    # Bottom 3cm ignored, actual 0.66m upper bound included.
    for z,occupied in ((.02,False),(.03,True),(.66,True),(.67,False)):
        rr=rays(origins=[[-.8,0,z]])
        ff=frame(hits(scene([([0.,0.,z],[.02,.2,.001])]),rr),c)
        assert bool((ff.state==OCCUPIED).any())==occupied


def test_support_gap_ttl_and_empty_requirement():
    c=cfg();r=rays();h=hits(scene([([1.5,0,.2],[.02,.2,.2])]),r);s=support(c)
    s.observed[:, :, c.nx//2]=False
    f=frame(h,c,s)
    req=torch.zeros_like(f.state,dtype=torch.bool);req[:,:,c.nx//2]=True
    assert not certify176(f,req,torch.tensor([0]),c,required_outside=torch.tensor([False]))[0].any()
    assert not certify176(f,torch.zeros_like(req),torch.tensor([0]),c,required_outside=torch.tensor([False]))[0].any()
    state,age,_=expire176(f,torch.tensor([501]),c)
    assert (state==UNKNOWN).all() and (age==15).all()
    state,age,_=expire176(f,torch.tensor([1]),c)
    assert (age[state!=UNKNOWN]==1).all() # ceil(1ms/100ms)


def test_person_enters_previously_free_current_unknown():
    c=cfg();r=rays();old=frame(hits(scene([([1.5,0,.2],[.02,.2,.2])]),r),c)
    now=frame(hits(scene([([0.,0,.4],[.05,.3,.4])]),r),c,t=100)
    merged=merge_aligned176(now,old,c,alignment_valid=torch.tensor([True]))
    assert merged.state[0,*cell(c,.8)]==UNKNOWN # old free not reused behind person
    assert merged.state[0,*cell(c,-.04)]==OCCUPIED


def test_occupied_ttl_preserves_old_support_and_epoch():
    c=cfg();r=rays();old=frame(hits(scene([([0.,0,.2],[.02,.2,.2])]),r),c)
    now=frame(hits(scene([([1.5,0,.2],[.02,.2,.2])]),r),c,support(c,z=.2,time=100),t=100)
    out=merge_aligned176(now,old,c,alignment_valid=torch.tensor([True]));occupied=old.state==OCCUPIED
    assert (out.state[occupied]==OCCUPIED).all()
    assert (out.support.reference_z[occupied]==0).all()
    assert (out.seen_ms[occupied]==0).all()
    reset=merge_aligned176(replace(now,epoch=torch.tensor([1])),old,c,alignment_valid=torch.tensor([True]))
    assert torch.equal(reset.state,now.state)
    expired=merge_aligned176(replace(now,capture_ms=torch.tensor([501])),old,c,alignment_valid=torch.tensor([True]))
    assert torch.equal(expired.state,now.state)


def test_stair_tread_vs_flat_box_observed_support():
    # Existing C2 growth, route hint controls tread acceptance.
    heights=torch.tensor([[[0.,.2,.4]]]);seen=torch.ones_like(heights,dtype=torch.long)*100
    normals=torch.zeros((*heights.shape,3));normals[...,2]=1
    observed=torch.ones_like(heights,dtype=torch.bool);seed=torch.tensor([[[True,False,False]]])
    stair=observed_support176(heights,observed,normals,seed,seen_ms=seen,segment_type='STAIR_UP',measured_current_support=torch.tensor([0.]))
    flat=observed_support176(heights,observed,normals,seed,seen_ms=seen,segment_type='FLAT',measured_current_support=torch.tensor([0.]))
    assert stair.observed.all();assert not flat.observed[0,0,1:].any()
    assert torch.allclose(stair.reference_z,heights)
    assert torch.allclose(flat.reference_z,torch.zeros_like(heights))


def test_no_gt_oracle_errors_and_incomplete():
    c=cfg();r=rays();g=scene([([0.,0,.2],[.02,.2,.2])]);h=hits(g,r)
    assert perturb_returns176(h,c) is h
    with pytest.raises(ValueError):perturb_returns176(h,c,hook=lambda v:v)
    with pytest.raises(ValueError):perturb_returns176(h,replace(c,noise_stage='tuning'))
    assert perturb_returns176(h,replace(c,noise_stage='tuning'),hook=lambda v:replace(v,valid=torch.zeros_like(v.valid))).valid.sum()==0
    assert not hits(replace(g,complete=torch.zeros_like(g.complete)),r).valid.any()
    with pytest.raises(ValueError):hits(g,replace(r,source=torch.tensor([[3]])))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU optional')
def test_cpu_gpu_parity():
    c=cfg(.05);boxes=[([.51,0,.2],[.01,.01,.2])]
    cpu=frame(hits(scene(boxes),rays()),c)
    gpu=frame(hits(scene(boxes,'cuda'),rays(device='cuda')),c)
    assert torch.equal(cpu.state,gpu.state.cpu())
    assert torch.equal(cpu.seen_ms,gpu.seen_ms.cpu())


def test_native_gdm_pattern_filters_and_rear_boundary():
    from gd_lab.gast_nav.oracle_patterns176 import mounted_rays176,apply_self_sectors176,load_gdm_profile176
    ref=Path('/home/user/gd_project/gdm_docker/workspace/src')
    profile=load_gdm_profile176(ref/'trav_map/trav_map/config/setups/rbq/rbq.yaml',Path('/home/user/gd_project/sim_docker/_sensor_configuration.yaml'),effective_parameters=dict(ouster_min_range_m=.4,ouster_max_range_m=6.,livox_min_range=.05,livox_max_range=1.5,livox_user_sector_deg=150.,livox_vertical_fov_deg=[-7.,52.]))
    # Native calibration fixture, not a manufactured OS1 beam pattern.
    directions=torch.tensor([[-1.,0,0],[1.,0,0],[1.,0,0],[-1.,0,0],[1.,0,0]])
    source=torch.tensor([0,0,1,1,2]);ring=torch.tensor([40,10,-1,-1,-1]);mount=torch.tensor([0,0,1,1,2])
    r,local=mounted_rays176(directions,source,ring,mount,torch.eye(3)[None,None].expand(1,3,-1,-1),torch.tensor([[[.28,0,.1],[.26,0,.18],[-.39,.24,.1]]]),profile)
    assert r.coverage_valid.tolist()==[[False,False,True,False,True]]
    assert abs(float(r.far_m[0,4])-.08)<1e-6
    # Real self hit remains present; cannot become a farther background return.
    h=OracleReturns176(r,torch.full((1,5),.1),torch.ones((1,5),dtype=torch.bool),torch.zeros((1,5),dtype=torch.bool))
    angled=local.clone();angled[:,:,2]=.3
    selected=apply_self_sectors176(h,angled,profile)
    assert selected.self_hit[0,2]
    assert torch.equal(selected.distance_m,h.distance_m)


def test_drift_reuse_rejected_and_support_is_immutable():
    c=replace(cfg(),retain_free_across_captures=True);r=rays()
    old=frame(hits(scene([([1.5,0,.2],[.02,.2,.2])]),r),c)
    invisible=replace(r,coverage_valid=torch.zeros_like(r.coverage_valid))
    now=frame(hits(scene([([1.5,0,.2],[.02,.2,.2])]),invisible),c,t=100)
    out=merge_aligned176(now,old,c,alignment_valid=torch.tensor([False]))
    assert (out.state==UNKNOWN).all()
    s=support(c);f=frame(hits(scene([([.5,0,.2],[.02,.2,.2])]),r),c,s)
    s.reference_z.fill_(999)
    assert (f.support.reference_z==0).all()


def test_rotated_first_hits_and_grid_are_consistent():
    c=replace(cfg(),x_max_m=1.);r=rays(origins=[[-.8,.05,.15]])
    g=scene([([.51,.05,.2],[.01,.01,.2])]);original=frame(hits(g,r),c)
    rot=torch.tensor([[0.,-1,0],[1.,0,0],[0,0,1]])
    moved=replace(g,centres=g.centres@rot.T,rotations=rot[None,None]@g.rotations)
    rr=replace(r,origins=r.origins@rot.T,directions=r.directions@rot.T)
    result=frame(hits(moved,rr),c)
    assert torch.equal(result.state,torch.rot90(original.state,-1,(1,2)))


def test_observed_stair_treads_project_as_support_not_obstacles():
    c=OccConfig176(x_min_m=0.,x_max_m=.3,y_min_m=0.,y_max_m=.1)
    heights=torch.tensor([[[0.,.2,.4]]]);normals=torch.zeros((*heights.shape,3));normals[...,2]=1
    observed=torch.ones_like(heights,dtype=torch.bool);seeds=torch.tensor([[[True,False,False]]]);seen=torch.zeros_like(heights,dtype=torch.long)
    stair=observed_support176(heights,observed,normals,seeds,seen_ms=seen,segment_type='STAIR_UP',measured_current_support=torch.tensor([0.]))
    flat=observed_support176(heights,observed,normals,seeds,seen_ms=seen,segment_type='FLAT',measured_current_support=torch.tensor([0.]))
    boxes=[([.05,.05,-.025],[.05,.05,.025]),([.15,.05,.175],[.05,.05,.025]),([.25,.05,.375],[.05,.05,.025])]
    rr=rays(origins=[[.05,.05,.6],[.15,.05,.6],[.25,.05,.6]],directions=[[0.,0.,-1.]]*3)
    h=hits(scene(boxes),rr)
    tread=frame(h,c,stair);box=frame(h,c,flat)
    assert (tread.state==FREE).all()
    assert box.state[0,0,1]==OCCUPIED and box.state[0,0,2]==OCCUPIED
    # A gap has no measured support even though an oracle ray traverses it.
    gap=replace(stair,observed=stair.observed.clone());gap.observed[0,0,1]=False
    gapframe=frame(h,c,gap)
    required=torch.ones_like(gapframe.state,dtype=torch.bool)
    assert not certify176(gapframe,required,torch.tensor([0]),c,required_outside=torch.tensor([False]))[0].any()


def test_confirmed_no_return_vs_dropout_and_outside_swept():
    c=cfg();r=rays()
    g=scene([([10.,10.,.2],[.01,.01,.2])])
    h=hits(g,r)
    assert h.valid.all() and torch.isinf(h.distance_m).all()
    f=frame(h,c);required=f.state==FREE
    assert certify176(f,required,torch.tensor([0]),c,required_outside=torch.tensor([False]))[0].all()
    assert not certify176(f,required,torch.tensor([0]),c,required_outside=torch.tensor([True]))[0].any()
    missing=frame(replace(h,valid=torch.zeros_like(h.valid)),c)
    assert (missing.state==UNKNOWN).all()


def test_fresh_occupied_keeps_new_observation_timestamp():
    c=cfg();r=rays();h=hits(scene([([0.,0,.2],[.02,.2,.2])]),r)
    old=frame(h,c,t=0)
    new=frame(h,c,support(c,time=100),t=100)
    merged=merge_aligned176(new,old,c,alignment_valid=torch.tensor([True]))
    occupied=new.state==OCCUPIED
    assert (merged.seen_ms[occupied]==100).all()
    assert (merged.support.seen_ms[occupied]==100).all()
