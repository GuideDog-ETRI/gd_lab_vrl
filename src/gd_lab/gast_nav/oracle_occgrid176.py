"""UPDATE176: first-hit oracle measurements -> single 2D supervisor grid.

This module never supplies actor OBS1, GT support or a global occupancy map.
No frustum erosion/volume inclusion proof is used for free-cell decisions.
"""
from dataclasses import dataclass,replace
import math
import torch

UNKNOWN,FREE,OCCUPIED=0,1,2

@dataclass(frozen=True)
class OccConfig176:
    body_height_m: float=.66
    floor_min_m: float=.03
    cell_m: float=.10
    x_min_m: float=-2.45
    x_max_m: float=2.45
    y_min_m: float=-2.15
    y_max_m: float=2.15
    ttl_s: float=.5
    retain_free_across_captures: bool=False
    noise_stage: str='oracle'
    support_ttl_s: float | None=None # appended, positional-compatible
    front_depth183: bool=False # explicit supervisor-only front D430 opt-in

    def __post_init__(self):
        values=(self.body_height_m,self.floor_min_m,self.cell_m,self.x_min_m,self.x_max_m,self.y_min_m,self.y_max_m,self.ttl_s)
        if not all(math.isfinite(v) for v in values) or self.cell_m not in (.1,.05) or not 0<=self.floor_min_m<self.body_height_m or self.ttl_s<=0:raise ValueError('grid config')
        if self.x_max_m<=self.x_min_m or self.y_max_m<=self.y_min_m:raise ValueError('bounds')
        for extent in (self.x_max_m-self.x_min_m,self.y_max_m-self.y_min_m):
            if abs(extent/self.cell_m-round(extent/self.cell_m))>1e-5:raise ValueError('whole cells required')
        if self.noise_stage not in ('oracle','tuning'):raise ValueError('noise stage')
        if self.support_ttl_s is not None and (not math.isfinite(self.support_ttl_s) or self.support_ttl_s<=0):raise ValueError('support TTL')
    @property
    def nx(self):return round((self.x_max_m-self.x_min_m)/self.cell_m)
    @property
    def ny(self):return round((self.y_max_m-self.y_min_m)/self.cell_m)
    @classmethod
    def load(cls,path):
        import yaml
        with open(path) as stream:return cls(**yaml.safe_load(stream))

@dataclass(frozen=True)
class OracleRays176:
    origins: torch.Tensor # [N,R,3], true sensor sample yaw frame
    directions: torch.Tensor # [N,R,3], normalization applied
    near_m: torch.Tensor # [N,R], excluded prefix is NEVER carved free
    far_m: torch.Tensor # [N,R]
    coverage_valid: torch.Tensor # [N,R], calibrated pattern + gdm sectors/self masks
    source: torch.Tensor # [N,R] int64: 0 Ouster, 1 Mid360, 2 assumed rear; 4 HEAD D455; source 3 forbidden; 5/6 front D430 only with front_depth183 opt-in

@dataclass(frozen=True)
class OracleReturns176:
    rays: OracleRays176
    distance_m: torch.Tensor # [N,R], true nearest metric hit; inf means no return
    valid: torch.Tensor # [N,R], false for incomplete/masked measurement
    self_hit: torch.Tensor # [N,R], terminate but never occupied

@dataclass(frozen=True)
class Support176:
    reference_z: torch.Tensor # [N,Y,X], base-relative observed/reference height, NaN unknown
    observed: torch.Tensor # [N,Y,X] bool; not a GT floor/prior fill
    seen_ms: torch.Tensor # [N,Y,X] int64, original actual observation timestamp

@dataclass(frozen=True)
class OccFrame176:
    state: torch.Tensor # [N,Y,X] uint8
    seen_ms: torch.Tensor # [N,Y,X] int64, -1 unknown
    support: Support176
    epoch: torch.Tensor # [N] int64
    capture_ms: torch.Tensor # [N] int64


def _validate_rays(r, *, front_depth183=False):
    n,b,_=r.origins.shape
    if r.origins.shape!=(n,b,3) or r.directions.shape!=r.origins.shape:raise ValueError('rays')
    for v in (r.origins,r.directions,r.near_m,r.far_m):
        if v.device!=r.origins.device or v.dtype!=r.origins.dtype or not bool(torch.isfinite(v).all()):raise ValueError('ray float metadata')
    if r.near_m.shape!=(n,b) or r.far_m.shape!=(n,b) or bool(((r.near_m<0)|(r.far_m<=r.near_m)).any()):raise ValueError('range')
    if r.coverage_valid.shape!=(n,b) or r.coverage_valid.dtype!=torch.bool or r.source.shape!=(n,b) or r.source.dtype!=torch.int64:raise ValueError('coverage/source')
    allowed=(r.source>=0)&((r.source<=2)|(r.source==4))
    if front_depth183:allowed |= (r.source==5)|(r.source==6)
    if r.coverage_valid.device!=r.origins.device or r.source.device!=r.origins.device or not bool(allowed.all()):raise ValueError('D4xx/unknown source forbidden')
    norm=r.directions.norm(dim=-1)
    if bool((norm<1e-10).any()):raise ValueError('zero direction')
    return replace(r,directions=r.directions/norm[...,None])


def oracle_first_hits176(scene,rays,*,scene_capture_id,ray_capture_id,self_mask=None):
    """Private primitives -> actual nearest BOX/CYLINDER/SPHERE intersections.

    Real sensor patterns/extrinsics and gdm coverage masks are supplied by the
    simulator, not invented here. Oracle removes errors, NOT occlusion/coverage.
    Unknown/unsupported inventory fails closed. Optional self_mask [N,P] uses
    existing simulator self-hit convention, no new mesh model. A self hit before
    near invalidates the ray; filtering it must never reveal farther geometry.
    """
    from .producer168 import validate as validate_primitives
    r=_validate_rays(rays);n,b=r.near_m.shape
    if scene.centres.shape[0]!=n or scene.centres.device!=r.origins.device or scene.centres.dtype!=r.origins.dtype:raise ValueError('scene/ray batch')
    for ids in (scene_capture_id,ray_capture_id):
        if ids.shape!=(n,) or ids.dtype!=torch.int64 or ids.device!=r.origins.device:raise ValueError('capture IDs')
    if not torch.equal(scene_capture_id,ray_capture_id):raise ValueError('unsynchronised capture')
    # Reuse generic primitive validation with an identity mathematical chart.
    eye=torch.eye(3,device=r.origins.device,dtype=r.origins.dtype)
    _,complete=validate_primitives(scene,eye[None],eye[None,None].expand(n,1,-1,-1),r.origins.new_zeros((n,1,3)),torch.tensor([[0,0,0,1,1]],device=r.origins.device))
    if self_mask is None:self_mask=torch.zeros_like(scene.active)
    if self_mask.shape!=scene.active.shape or self_mask.dtype!=torch.bool or self_mask.device!=scene.active.device:raise ValueError('self mask')
    out=[];self_out=[]
    inv=scene.rotations.transpose(-1,-2);h=scene.half[:,None]
    for start in range(0,b,256):
        origin=torch.einsum('npij,nbpj->nbpi',inv,r.origins[:,start:start+256,None]-scene.centres[:,None])
        d=torch.einsum('npij,nbj->nbpi',inv,r.directions[:,start:start+256])
        parallel=d.abs()<1e-10;den=torch.where(parallel,1.,d)
        aa=(-h-origin)/den;bb=(h-origin)/den
        entry=torch.where(parallel,-torch.inf,torch.minimum(aa,bb)).amax(-1).clamp_min(0)
        exit=torch.where(parallel,torch.inf,torch.maximum(aa,bb)).amin(-1)
        box=torch.where(~(parallel&(origin.abs()>h)).any(-1)&(exit>=entry),entry,torch.inf)
        def roots(a,bb,c):
            disc=bb*bb-a*c;den=torch.where(a>1e-12,a,1.)
            return (-bb-disc.clamp_min(0).sqrt())/den,(-bb+disc.clamp_min(0).sqrt())/den,(disc>=0)&(a>1e-12)
        t0,_,ok=roots(d.square().sum(-1),(origin*d).sum(-1),origin.square().sum(-1)-h[...,0].square())
        sphere=torch.where(origin.square().sum(-1)<=h[...,0].square(),0.,torch.where(ok&(t0>=0),t0,torch.inf))
        t0,t1,ok=roots(d[...,:2].square().sum(-1),(origin[...,:2]*d[...,:2]).sum(-1),origin[...,:2].square().sum(-1)-h[...,0].square())
        cyl=torch.full_like(box,torch.inf)
        for tt in (t0,t1):cyl=torch.minimum(cyl,torch.where(ok&(tt>=0)&((origin[...,2]+tt*d[...,2]).abs()<=h[...,2]),tt,torch.inf))
        for sign in (-1.,1.):
            dz=d[...,2];tt=(sign*h[...,2]-origin[...,2])/torch.where(dz.abs()>1e-12,dz,1.)
            xy=origin[...,:2]+tt[...,None]*d[...,:2]
            cyl=torch.minimum(cyl,torch.where((dz.abs()>1e-12)&(tt>=0)&(xy.square().sum(-1)<=h[...,0].square()),tt,torch.inf))
        cyl=torch.where((origin[...,:2].square().sum(-1)<=h[...,0].square())&(origin[...,2].abs()<=h[...,2]),0.,cyl)
        hits=torch.where(scene.kind[:,None]==0,box,torch.where(scene.kind[:,None]==1,cyl,sphere))
        dist,index=torch.where(scene.active[:,None],hits,torch.inf).min(-1)
        # Self wins ties, including coincident collision surfaces.
        self_dist=torch.where(scene.active[:,None]&self_mask[:,None],hits,torch.inf).amin(-1)
        self_out.append(torch.isfinite(self_dist)&(self_dist<=dist+1e-7))
        out.append(dist)
    distance=torch.cat(out,1);self_hit=torch.cat(self_out,1)
    valid=r.coverage_valid&complete[:,None]&(distance>=r.near_m)
    distance=torch.where(distance<=r.far_m,distance,torch.inf)
    return OracleReturns176(r,distance,valid,self_hit&torch.isfinite(distance))


def perturb_returns176(returns,cfg,*,hook=None):
    """Tuning-only hook for range/dropout/mount/odom schedules outside module.

    Baseline rejects a hook. Perturbed mount/pose MUST recast real geometry;
    callbacks must not relabel GT occupancy or carve behind a removed return.
    """
    if cfg.noise_stage=='oracle':
        if hook is not None:raise ValueError('oracle baseline must have no error hook')
        return returns
    if hook is None:raise ValueError('explicit tuning hook required')
    result=hook(returns)
    if not isinstance(result,OracleReturns176):raise ValueError('hook result')
    return result


def observed_support176(height,observed,normals,seeds,*,seen_ms,segment_type,
                        measured_current_support,config=None):
    """Existing C2.1: FLAT +/- .10m, STAIR step<=.25m, fitted slope<=35deg.
    Inputs are SENSOR hit fits/contact seeds only, never a GT support mesh.
    """
    from .support import grow_support
    z,_,accepted=grow_support(height,observed,normals,seeds,segment_type=segment_type,
        current_support_height=measured_current_support,config=config)
    return Support176(z,accepted,seen_ms)


def rasterize176(returns,support,cfg,*,capture_ms,epoch,obstacle_min_height_m=None):
    """Exact XY DDA of valid ray segments, first hit only, occupied wins.

    No return: free to configured far only when coverage_valid explicitly
    denotes a valid range-limited oracle measurement. Masked/invalid rays carve
    nothing. All actual hits stop rays even when above body_height or self.
    Unknown height reference at a hit prevents that endpoint becoming free.
    Surface hits within floor_min of observed support are not obstacles; stair
    treads therefore remain support through the existing C2 route-hint rules.
    """
    r=_validate_rays(returns.rays,front_depth183=cfg.front_depth183);n,b=r.near_m.shape;shape=(n,cfg.ny,cfg.nx);m=cfg.nx*cfg.ny
    for v in (support.reference_z,support.observed,support.seen_ms):
        if v.shape!=shape or v.device!=r.origins.device:raise ValueError('support grid')
    if support.observed.dtype!=torch.bool or support.seen_ms.dtype!=torch.int64:raise ValueError('support metadata')
    if capture_ms.shape!=(n,) or epoch.shape!=(n,) or capture_ms.dtype!=torch.int64 or epoch.dtype!=torch.int64 or capture_ms.device!=r.origins.device or epoch.device!=r.origins.device:raise ValueError('frame metadata')
    if bool((capture_ms<0).any()|(epoch<0).any()):raise ValueError('negative frame time/epoch')
    if support.reference_z.dtype!=r.origins.dtype or bool((support.observed&(~torch.isfinite(support.reference_z)|(support.seen_ms<0)|(support.seen_ms>capture_ms[:,None,None]))).any()):raise ValueError('observed support provenance')
    if returns.distance_m.shape!=(n,b) or returns.valid.shape!=(n,b) or returns.self_hit.shape!=(n,b) or returns.valid.dtype!=torch.bool or returns.self_hit.dtype!=torch.bool:raise ValueError('return shapes')
    if bool(torch.isnan(returns.distance_m).any()|(returns.distance_m<0).any()):raise ValueError('return distance')
    free=torch.zeros((n,m),device=r.origins.device,dtype=torch.int32);occ=torch.zeros_like(free);ambiguous=torch.zeros_like(free)
    dist=returns.distance_m;finite=torch.isfinite(dist)&(dist<=r.far_m)
    stop=torch.minimum(dist,r.far_m)
    good=returns.valid&r.coverage_valid&(stop>=r.near_m)
    o=r.origins[...,:2];d=r.directions[...,:2]
    low=o.new_tensor([cfg.x_min_m,cfg.y_min_m]);high=o.new_tensor([cfg.x_max_m,cfg.y_max_m])
    parallel=d.abs()<1e-12;safe=torch.where(parallel,1.,d)
    aa=(low-o)/safe;bb=(high-o)/safe
    entry=torch.where(parallel,-torch.inf,torch.minimum(aa,bb)).amax(-1)
    exit=torch.where(parallel,torch.inf,torch.maximum(aa,bb)).amin(-1)
    begin=torch.maximum(entry,r.near_m);end=torch.minimum(exit,stop)
    live=good&~(parallel&((o<low)|(o>=high))).any(-1)&(end>begin+1e-7)
    # Nudge inside at an entry edge, never carve the excluded prefix.
    pos=o+d*(begin+1e-6)[...,None]
    cell=torch.floor((pos-low)/cfg.cell_m).long()
    sx=torch.sign(d[...,0]).long();sy=torch.sign(d[...,1]).long()
    bx=low[0]+(cell[...,0]+(sx>0).long())*cfg.cell_m
    by=low[1]+(cell[...,1]+(sy>0).long())*cfg.cell_m
    tx=torch.where(d[...,0].abs()>1e-12,(bx-o[...,0])/safe[...,0],torch.inf)
    ty=torch.where(d[...,1].abs()>1e-12,(by-o[...,1])/safe[...,1],torch.inf)
    dx=torch.where(d[...,0].abs()>1e-12,cfg.cell_m/d[...,0].abs(),torch.inf)
    dy=torch.where(d[...,1].abs()>1e-12,cfg.cell_m/d[...,1].abs(),torch.inf)
    for _ in range(cfg.nx+cfg.ny+2):
        inside=live&(cell[...,0]>=0)&(cell[...,0]<cfg.nx)&(cell[...,1]>=0)&(cell[...,1]<cfg.ny)
        index=(cell[...,1].clamp(0,cfg.ny-1)*cfg.nx+cell[...,0].clamp(0,cfg.nx-1))
        free.scatter_reduce_(1,index,inside.to(torch.int32),reduce='amax',include_self=True)
        crossing=torch.minimum(tx,ty);active=inside&(crossing<end-1e-7)
        crossx=tx<=ty+1e-7;crossy=ty<=tx+1e-7
        cell=torch.stack((cell[...,0]+(active&crossx).long()*sx,cell[...,1]+(active&crossy).long()*sy),-1)
        tx=torch.where(active&crossx,tx+dx,tx);ty=torch.where(active&crossy,ty+dy,ty)
        live=active
        if not bool(live.any()):break
    point=r.origins+r.directions*dist.nan_to_num(posinf=0)[...,None]
    qxy=(point[...,:2]-low)/cfg.cell_m
    qxy=torch.where((qxy-qxy.round()).abs()<2e-5,qxy.round(),qxy)
    xy=torch.floor(qxy).long()
    hit=good&finite&~returns.self_hit&(xy[...,0]>=0)&(xy[...,0]<cfg.nx)&(xy[...,1]>=0)&(xy[...,1]<cfg.ny)
    idx=xy[...,1].clamp(0,cfg.ny-1)*cfg.nx+xy[...,0].clamp(0,cfg.nx-1)
    reference=support.reference_z.flatten(1).gather(1,idx)
    height=point[...,2]-reference
    floor_min=cfg.floor_min_m
    if obstacle_min_height_m is not None:
        limit=torch.as_tensor(obstacle_min_height_m,device=height.device,dtype=height.dtype)
        if limit.shape!=(height.shape[0],) or not bool(torch.isfinite(limit).all()&(limit>0).all()&(limit<=cfg.body_height_m).all()):raise ValueError('observed hit obstacle threshold [N]')
        floor_min=limit[:,None]+2e-6 # inclusive walkable limit, preserve native epsilon
    obstacle=hit&torch.isfinite(reference)&(height>=floor_min-1e-6)&(height<=cfg.body_height_m+1e-6)
    occ.scatter_reduce_(1,idx,obstacle.to(torch.int32),reduce='amax',include_self=True)
    ambiguous.scatter_reduce_(1,idx,(hit&~torch.isfinite(reference)).to(torch.int32),reduce='amax',include_self=True)
    # Self endpoint cannot be certified free even if it shares a visited cell.
    self_endpoint=good&finite&returns.self_hit&(xy[...,0]>=0)&(xy[...,0]<cfg.nx)&(xy[...,1]>=0)&(xy[...,1]<cfg.ny)
    ambiguous.scatter_reduce_(1,idx,self_endpoint.to(torch.int32),reduce='amax',include_self=True)
    state=torch.where(occ.bool(),OCCUPIED,torch.where(free.bool()&~ambiguous.bool(),FREE,UNKNOWN)).to(torch.uint8).reshape(shape)
    seen=torch.where(state!=UNKNOWN,capture_ms[:,None,None],-1)
    frozen_support=Support176(support.reference_z.clone(),support.observed.clone(),support.seen_ms.clone())
    return OccFrame176(state,seen,frozen_support,epoch.clone(),capture_ms.clone())


def expire176(frame,now_ms,cfg):
    age=(now_ms[:,None,None]-frame.seen_ms).to(torch.float64)/1000
    valid=(frame.seen_ms>=0)&(age>=0)&(age<=cfg.ttl_s+1e-9)&(now_ms>=frame.capture_ms)[:,None,None]
    support_age=(now_ms[:,None,None]-frame.support.seen_ms).to(torch.float64)/1000
    support_valid=frame.support.observed&torch.isfinite(frame.support.reference_z)&(frame.support.seen_ms>=0)&(support_age>=0)&(support_age<=(cfg.support_ttl_s if cfg.support_ttl_s is not None else cfg.ttl_s)+1e-9)
    # No observed support => UNKNOWN for supervisor use; no flat GT prior.
    state=torch.where(valid&((frame.state==OCCUPIED)|support_valid),frame.state,UNKNOWN).to(torch.uint8)
    quant=torch.where(state!=UNKNOWN,torch.ceil(age.clamp_min(0)*10).clamp(0,15),15).to(torch.uint8)
    return state,quant,support_valid


def merge_aligned176(current,previous,cfg,*,alignment_valid):
    """Caller must categorical-warp previous into current grid/frame first.

    Default retains occupied until TTL, NOT free behind new occlusion. Preserve
    that occupied cell's old support/classification/seen times with it. No pose
    is silently reassigned here. Epoch reset forbids any cross-episode reuse.
    """
    if current.state.shape!=previous.state.shape:raise ValueError('aligned grids required')
    if alignment_valid.shape!=current.epoch.shape or alignment_valid.dtype!=torch.bool or alignment_valid.device!=current.state.device:raise ValueError('explicit relative pose/warp validity required')
    age=(current.capture_ms[:,None,None]-previous.seen_ms).to(torch.float64)/1000
    keep=alignment_valid[:,None,None]&(current.epoch==previous.epoch)[:,None,None]&(age>=0)&(age<=cfg.ttl_s)&(previous.seen_ms>=0)
    occupied=keep&(previous.state==OCCUPIED)&(current.state!=OCCUPIED)
    carry_free=keep&(previous.state==FREE)&(current.state==UNKNOWN)&cfg.retain_free_across_captures
    take=occupied|carry_free
    state=torch.where(occupied,OCCUPIED,torch.where(carry_free,FREE,current.state))
    seen=torch.where(take,previous.seen_ms,current.seen_ms)
    support=Support176(*(torch.where(take,a,b) for a,b in zip((previous.support.reference_z,previous.support.observed,previous.support.seen_ms),(current.support.reference_z,current.support.observed,current.support.seen_ms))))
    return replace(current,state=state,seen_ms=seen,support=support)


def certify176(frame,required_mask,now_ms,cfg,*,required_outside):
    """5090 supplies stopping-swept + margin raster, no volume proof here."""
    state,age,support_valid=expire176(frame,now_ms,cfg)
    if required_mask.shape!=state.shape or required_mask.dtype!=torch.bool:raise ValueError('required mask')
    if required_outside.shape!=frame.epoch.shape or required_outside.dtype!=torch.bool or required_outside.device!=state.device:raise ValueError('outside-grid swept flag required')
    bad=required_mask&((state!=FREE)|~support_valid)
    # Empty requirements are not an affirmative motion certificate.
    ok=required_mask.flatten(1).any(1)&~bad.flatten(1).any(1)&~required_outside
    return ok,state,age
