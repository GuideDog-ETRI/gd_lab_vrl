"""UPDATE167: sensor-certified convex free frusta, complete-volume containment.

A centre ray is NOT a volume. Finite depth/intrinsics alone must never assert
volume_valid: the producer warrants opaque >=5 cm obstacles within the pixel
footprint are detected and every pixel in a block shares a first-hit bound.
Python/Torch reference, no implicit CUDA backend, no GT/robot mounts assumed.
"""
from dataclasses import dataclass, fields
from pathlib import Path
import math
import torch
import yaml
from . import schema as S
from .map_types import WIDE_GRID
from .packets import age_codes, age_upper_s
from .warp import warp_categorical
from .support import encode_support

@dataclass(frozen=True)
class FreeEvidenceConfig:
    minimum_obstacle_diameter_m: float = .05
    minimum_obstacle_height_m: float = .03
    range_error_floor_m: float = .02
    range_error_fraction: float = .02
    sensor_angle_error_deg: float = .5
    support_error_m: float = .02
    pose_rw_xy_m_sqrt_s: float = .015
    pose_rw_yaw_deg_sqrt_s: float = .3
    pose_slip_fraction: float = .03
    pose_white_xy_m: float = .005
    pose_white_yaw_deg: float = .2
    pose_delay_s: float = .04
    pose_sigma: float = 3.
    same_sample_pose_cancellation: bool = False # opt-in, audited synchronous sensor/pose source only
    historical_obstacle_speed_bound_mps: float | None = None
    max_captures: int = 6
    xy_parts: int = 2
    z_max_piece_m: float = .05
    cell_chunk: int = 32
    beam_chunk: int = 32
    def __post_init__(self):
        for item in fields(self):
            value=getattr(self,item.name)
            if value is not None and (not math.isfinite(value) or value<0):raise ValueError(item.name)
        for name in ('minimum_obstacle_diameter_m','minimum_obstacle_height_m','z_max_piece_m','max_captures','xy_parts','cell_chunk','beam_chunk'):
            if getattr(self,name)<=0:raise ValueError(name)
        for name in ('max_captures','xy_parts','cell_chunk','beam_chunk'):
            if not isinstance(getattr(self,name),int):raise ValueError(name)
    @classmethod
    def load(cls,path=None):
        path=Path(path) if path else Path(__file__).resolve().parents[3]/'configs/gast_nav/free_evidence167.yaml'
        return cls(**yaml.safe_load(path.read_text()))

@dataclass(frozen=True)
class FreeFrusta:
    origins: torch.Tensor       # [N,R,3] sample yaw frame
    normals: torch.Tensor       # [N,R,6,3] inward UNIT plane normals
    offsets: torch.Tensor       # [N,R,6], n dot (p-origin) >= offset
    far_m: torch.Tensor         # [N,R], conservative full-footprint first-hit depth
    volume_valid: torch.Tensor  # [N,R] bool, explicit sensor guarantee, NOT finite-depth alone

@dataclass(frozen=True)
class FreeCapture:
    frusta: FreeFrusta
    pose: torch.Tensor          # [N,4], odom x,y,z,yaw at capture
    support: torch.Tensor       # [N,Y,X], sample-relative observed reference height
    time_ms: torch.Tensor       # [N] int64, monotonic source time
    epoch: torch.Tensor         # [N] int64

@dataclass(frozen=True)
class FreeMeasurement:
    captures: tuple[FreeCapture,...]
    pose: torch.Tensor          # [N,4], current yaw frame
    now_ms: torch.Tensor        # [N] int64
    epoch: torch.Tensor         # [N] int64
    travel_m: torch.Tensor      # [N,K], conservative odom path length since capture (NOT displacement)
    linear_speed_bound_mps: torch.Tensor # [N], delay bound
    yaw_rate_bound_radps: torch.Tensor   # [N], delay bound
    settings: FreeEvidenceConfig

@dataclass(frozen=True)
class VolumeProof:
    free: torch.Tensor          # [N,3,Y,X] bool; no occupied semantics here
    age_s: torch.Tensor         # same, oldest required evidence, ceil 0.1s; inf if absent
    fraction: torch.Tensor      # same, fraction of fully covered micro-boxes, DIAGNOSTIC ONLY


def frusta_from_d4xx(depth, intrinsics, rotation, translation, *, blocks, volume_valid,
                    near_m=0., settings=None):
    """Depth [N,C,H,W] OPTICAL z in metres + K [C,3,3]/[N,C,3,3].

    R [N,C,3,3] maps camera (+z forward,+x right,+y down) to sample yaw;
    t [N,C,3]. No mount is hard coded. blocks [R,5] = camera,u0,v0,u1,v1,
    pixel-index half-open rectangles. Uses MIN depth, ALL pixels valid; never
    averages/bridges holes. volume_valid [N,R] is supplied by calibrated sensor.
    Pixel angular size at far range must also resolve the 5cm contract.
    Each block's full frustum ends before its nearest pixel return (self included).
    """
    cfg=settings or FreeEvidenceConfig.load()
    if depth.ndim!=4 or not depth.is_floating_point():raise ValueError('depth')
    n,c,h,w=depth.shape
    if intrinsics.shape==(c,3,3):intrinsics=intrinsics[None].expand(n,-1,-1,-1)
    if intrinsics.shape!=(n,c,3,3) or rotation.shape!=(n,c,3,3) or translation.shape!=(n,c,3):raise ValueError('camera shapes')
    if blocks.ndim!=2 or blocks.shape[1]!=5 or blocks.dtype not in (torch.int32,torch.int64):raise ValueError('blocks')
    r=len(blocks)
    if r==0:raise ValueError('at least one pixel block')
    if volume_valid.shape!=(n,r) or volume_valid.dtype!=torch.bool:raise ValueError('volume_valid')
    if not math.isfinite(near_m) or near_m<0:raise ValueError('near_m')
    if any(t.device!=depth.device for t in (intrinsics,rotation,translation,blocks,volume_valid)):raise ValueError('camera device')
    eye=torch.eye(3,device=depth.device,dtype=depth.dtype)
    if not bool(torch.isfinite(intrinsics).all() & torch.isfinite(rotation).all() & torch.isfinite(translation).all()):raise ValueError('camera finite')
    if not torch.allclose(rotation.transpose(-1,-2)@rotation,eye.expand_as(rotation),atol=1e-5) or not bool((torch.linalg.det(rotation)>0).all()):raise ValueError('camera rotation')
    if not bool((intrinsics[:,:,0,0]>0).all() & (intrinsics[:,:,1,1]>0).all()):raise ValueError('focal length')
    if not bool((intrinsics[:,:,0,1].abs()<1e-7).all() & (intrinsics[:,:,1,0].abs()<1e-7).all()) or not torch.allclose(intrinsics[:,:,2],depth.new_tensor([0.,0.,1.]).expand(n,c,3)):raise ValueError('rectified pinhole K required')
    os=[];ns=[];bs=[];ds=[];oks=[]
    for j,values in enumerate(blocks.tolist()):
        cam,u0,v0,u1,v1=values
        if not (0<=cam<c and 0<=u0<u1<=w and 0<=v0<v1<=h):raise ValueError('block extent')
        values=depth[:,cam,v0:v1,u0:u1]
        good=(torch.isfinite(values)&(values>near_m)).all((-2,-1))
        far=values.nan_to_num(nan=0.,posinf=0.,neginf=0.).flatten(1).amin(1)
        k=intrinsics[:,cam];fx,fy,cx,cy=k[:,0,0],k[:,1,1],k[:,0,2],k[:,1,2]
        xmin,xmax=(u0-.5-cx)/fx,(u1-.5-cx)/fx
        ymin,ymax=(v0-.5-cy)/fy,(v1-.5-cy)/fy
        z=torch.zeros_like(fx);one=torch.ones_like(fx)
        planes=torch.stack((torch.stack((one,z,-xmin),1),torch.stack((-one,z,xmax),1),torch.stack((z,one,-ymin),1),torch.stack((z,-one,ymax),1),torch.stack((z,z,one),1),torch.stack((z,z,-one),1)),1)
        length=planes.norm(dim=-1);planes=planes/length[...,None]
        b=torch.zeros((n,6),device=depth.device,dtype=depth.dtype);b[:,4]=near_m;b[:,5]=-far
        ns.append(torch.einsum('nij,nkj->nki',rotation[:,cam],planes));os.append(translation[:,cam]);bs.append(b);ds.append(far)
        # This is an extra sampling guard, not a derivation of volume_valid.
        pixel_diagonal=far*torch.sqrt(fx.reciprocal().square()+fy.reciprocal().square())
        oks.append(good & volume_valid[:,j] & (pixel_diagonal<=cfg.minimum_obstacle_diameter_m))
    return FreeFrusta(torch.stack(os,1),torch.stack(ns,1),torch.stack(bs,1),torch.stack(ds,1),torch.stack(oks,1))


def pose_budget(age_s,travel_m,speed_bound,yaw_rate_bound,settings):
    """P2 statistical working envelope. 3sigma is configurable, never a hard bound."""
    s=settings
    p=math.sqrt(2)*s.pose_sigma*torch.sqrt(s.pose_rw_xy_m_sqrt_s**2*age_s+2*s.pose_white_xy_m**2)
    p=p+s.pose_slip_fraction*travel_m+s.pose_delay_s*speed_bound
    angle=s.pose_sigma*torch.sqrt(s.pose_rw_yaw_deg_sqrt_s**2*age_s+2*s.pose_white_yaw_deg**2)*math.pi/180
    return p,angle+s.pose_delay_s*yaw_rate_bound


def _rotation(yaw):
    c,s=yaw.cos(),yaw.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
    return torch.stack((c,-s,z,s,c,z,z,z,one),-1).reshape(-1,3,3)


def _validate(m,support,grid):
    n=support.shape[0]
    if support.shape!=(n,grid.ny,grid.nx) or not support.is_floating_point():raise ValueError('support shape')
    if m.pose.shape!=(n,4) or m.now_ms.shape!=(n,) or m.epoch.shape!=(n,) or m.travel_m.shape!=(n,len(m.captures)):raise ValueError('measurement shape')
    if len(m.captures)>m.settings.max_captures:raise ValueError('capture limit')
    for v in (m.pose,m.now_ms,m.epoch,m.travel_m,m.linear_speed_bound_mps,m.yaw_rate_bound_radps):
        if v.device!=support.device or not bool(torch.isfinite(v).all()):raise ValueError('measurement device/finite')
    if m.now_ms.dtype!=torch.int64 or m.epoch.dtype!=torch.int64 or bool(((m.now_ms<0)|(m.epoch<0)).any()):raise ValueError('time/epoch')
    for v in (m.linear_speed_bound_mps,m.yaw_rate_bound_radps):
        if v.shape!=(n,) or bool((v<0).any()):raise ValueError('motion bound')
    if bool((m.travel_m<0).any()):raise ValueError('travel')
    previous=None
    for cap in m.captures:
        f=cap.frusta;r=f.origins.shape[1]
        if f.origins.shape!=(n,r,3) or f.normals.shape!=(n,r,6,3) or f.offsets.shape!=(n,r,6) or f.far_m.shape!=(n,r) or f.volume_valid.shape!=(n,r) or f.volume_valid.dtype!=torch.bool:raise ValueError('frusta shape')
        if cap.pose.shape!=(n,4) or cap.support.shape!=support.shape or cap.time_ms.shape!=(n,) or cap.epoch.shape!=(n,):raise ValueError('capture shape')
        for v in (f.origins,f.normals,f.offsets,f.far_m,f.volume_valid,cap.pose,cap.support,cap.time_ms,cap.epoch):
            if v.device!=support.device:raise ValueError('capture device')
        for v in (f.origins,f.normals,f.offsets,f.far_m):
            if v.dtype!=support.dtype:raise ValueError('frusta/support dtype mismatch')
        for v in (f.origins,f.normals,f.offsets,f.far_m,cap.pose):
            if not bool(torch.isfinite(v).all()):raise ValueError('capture finite')
        if not torch.allclose(f.normals.norm(dim=-1),torch.ones_like(f.offsets),atol=1e-5) or bool((f.far_m<0).any()):raise ValueError('frustum unit normal/range')
        forward=f.normals[:,:,4]
        if not torch.allclose(forward,-f.normals[:,:,5],atol=1e-5):raise ValueError('frustum near/far planes')
        transverse=f.normals[:,:,:4]-(f.normals[:,:,:4]*forward[:,:,None]).sum(-1)[...,None]*forward[:,:,None]
        lengths=transverse.norm(dim=-1)
        if bool((lengths<1e-5).any()):raise ValueError('frustum side planes')
        transverse=transverse/lengths[...,None]
        if not (torch.allclose(transverse[:,:,0],-transverse[:,:,1],atol=1e-5) and torch.allclose(transverse[:,:,2],-transverse[:,:,3],atol=1e-5)) or bool(((transverse[:,:,0]*transverse[:,:,2]).sum(-1).abs()>1e-5).any()):raise ValueError('frustum bounded transverse planes')
        if cap.time_ms.dtype!=torch.int64 or cap.epoch.dtype!=torch.int64 or bool((cap.time_ms<0).any()):raise ValueError('capture time')
        same=cap.epoch==m.epoch
        if bool((same & (cap.time_ms>m.now_ms)).any()):raise ValueError('future capture')
        if previous is not None and bool((same & (previous.epoch==m.epoch) & (cap.time_ms>=previous.time_ms)).any()):raise ValueError('newest-first unique capture times')
        previous=cap


def volume_free_proof(support,measurement,*,grid=WIDE_GRID,config=None,backend=None):
    """Full voxel coverage by union of eroded guaranteed-free convex frusta.

    Partition each cell/band into small boxes. A box must be wholly inside at
    least ONE frustum. Union is across such complete boxes (never just sampled
    points/subbin centre hits). More partitions reduce false unknown; never
    turn a gap into free by count/area. Whole band = all boxes, age=max necessary
    youngest proofs. Old footprints erode by bounded object reachability; if
    no dynamic speed bound is configured, past captures cannot prove free.
    """
    from .map_backend import use_triton
    if backend not in (None,'torch','triton'):raise ValueError('proof backend')
    selected=backend=='triton' or (backend is None and support.dtype==torch.float32 and use_triton(support))
    if selected:
        from .free_evidence169_cuda import volume_free_proof_cuda
        return volume_free_proof_cuda(support,measurement,grid=grid,config=config)
    from .map_config import load_config
    cfg=config or load_config();m=measurement;s=m.settings;_validate(m,support,grid)
    n=support.shape[0];xy=grid.xy(device=support.device,dtype=support.dtype).flatten(0,1);cells=len(xy)
    valid_support=torch.isfinite(support)&(support>=cfg.min_z_m)&(support<=cfg.max_z_m)
    prepared=[]
    for ci,cap in enumerate(m.captures):
        age=((m.now_ms-cap.time_ms).double()/1000).clamp_min(0)
        displacement=(m.pose[:,:2]-cap.pose[:,:2]).norm(dim=-1)
        if bool(((cap.epoch==m.epoch)&(m.travel_m[:,ci]+cfg.intersection_epsilon<displacement)).any()):raise ValueError('travel must bound displacement')
        upper=age_upper_s(age_codes(age));eligible=(cap.epoch==m.epoch)&(upper<=cfg.ttl_s)
        if s.historical_obstacle_speed_bound_mps is None:eligible &= age==0
        if not bool(eligible.any()):continue
        oldstate=torch.where(torch.isfinite(cap.support)[:,None],S.FREE,S.UNKNOWN).to(torch.uint8)
        age0=torch.zeros_like(oldstate,dtype=support.dtype)
        _,_,oldh=warp_categorical(oldstate,age0,cap.pose,m.pose,grid=grid,config=cfg,support=encode_support(cap.support,config=cfg),support_world_z=cap.support+cap.pose[:,2,None,None],return_support_height=True)
        # No old support is reclassified onto a changed/new stair reference.
        support_ok=valid_support & torch.isfinite(oldh) & ((oldh-support).abs()<=cfg.intersection_epsilon)
        rot=_rotation(cap.pose[:,3]-m.pose[:,3]).to(support.dtype)
        shift=(cap.pose[:,:3]-m.pose[:,:3]).to(support.dtype);shift=torch.einsum('nij,nj->ni',_rotation(-m.pose[:,3]).to(support.dtype),shift)
        f=cap.frusta;origins=torch.einsum('nij,nrj->nri',rot,f.origins)+shift[:,None]
        normals=torch.einsum('nij,nrkj->nrki',rot,f.normals)
        p,yaw=pose_budget(upper.to(support.dtype),m.travel_m[:,ci].to(support.dtype),m.linear_speed_bound_mps.to(support.dtype),m.yaw_rate_bound_radps.to(support.dtype),s)
        if s.same_sample_pose_cancellation:
            same=(cap.time_ms==m.now_ms)&(cap.pose==m.pose).all(-1)&(cap.epoch==m.epoch)&(m.travel_m[:,ci]==0)
            p=torch.where(same,0.,p);yaw=torch.where(same,0.,yaw)
        motion=upper.to(support.dtype)*(s.historical_obstacle_speed_bound_mps or 0.)
        prepared.append((cap,f,origins,normals,p,yaw,motion,upper.to(support.dtype),eligible,support_ok.flatten(1)))
    states=[];ages=[];fractions=[]
    for _,low,high in S.BANDS:
        nz=math.ceil((high-low)/s.z_max_piece_m);nx=s.xy_parts
        dz=(high-low)/nz
        offsets=support.new_tensor([( (i+.5)/nx*grid.res-grid.res/2,(j+.5)/nx*grid.res-grid.res/2,low+(k+.5)*dz) for i in range(nx) for j in range(nx) for k in range(nz)])
        half=support.new_tensor([grid.res/(2*nx),grid.res/(2*nx),dz/2]);k=len(offsets)
        band_free=torch.zeros((n,cells),device=support.device,dtype=torch.bool);band_age=torch.full((n,cells),torch.inf,device=support.device,dtype=support.dtype);fraction=torch.zeros_like(band_age)
        for start in range(0,cells,s.cell_chunk):
            end=min(start+s.cell_chunk,cells);h=support.flatten(1)[:,start:end]
            centers=torch.cat((xy[None,start:end].expand(n,-1,-1),h[...,None]),-1)[:,:,None]+offsets[None,None]
            box_age=torch.full((n,end-start,k),torch.inf,device=support.device,dtype=support.dtype)
            for cap,f,o,norm,p,yaw,motion,upper,eligible,support_ok in prepared:
                ok=eligible[:,None] & support_ok[:,start:end]
                for r0 in range(0,o.shape[1],s.beam_chunk):
                    origin=o[:,r0:r0+s.beam_chunk];normal=norm[:,r0:r0+s.beam_chunk]
                    q=centers[:,None]-origin[:,:,None,None]
                    # Angular displacement bound uses farthest corner distance.
                    radius=(q.norm(dim=-1)+half.norm()+f.origins[:,r0:r0+s.beam_chunk].norm(dim=-1)[:,:,None,None]).clamp_min(0)
                    # XY odom errors do not become vertical uncertainty. Support
                    # is vertical; yaw rotates about base, calibration about sensor.
                    sensor_error=2*math.sin(min(math.radians(s.sensor_angle_error_deg),math.pi)/2)*radius
                    yaw_error=2*torch.sin(yaw[:,None,None,None].clamp(max=math.pi)/2)*radius
                    contained=torch.ones(q.shape[:-1],device=support.device,dtype=torch.bool)
                    for plane in range(6):
                        normal_p=normal[:,:,plane]
                        distance=(q*normal_p[:,:,None,None]).sum(-1)-(normal_p.abs()*half).sum(-1)[:,:,None,None]
                        offset=f.offsets[:,r0:r0+s.beam_chunk,plane]
                        if plane==5:offset=torch.maximum(torch.maximum(offset,torch.full_like(offset,-cfg.ray_max_range_m)),-f.far_m[:,r0:r0+s.beam_chunk])+torch.maximum(torch.full_like(offset,s.range_error_floor_m),s.range_error_fraction*f.far_m[:,r0:r0+s.beam_chunk])
                        horizontal=normal_p[:,:,:2].norm(dim=-1)[:,:,None,None]
                        vertical=normal_p[:,:,2].abs()[:,:,None,None]
                        error=p[:,None,None,None]*horizontal+s.support_error_m*vertical
                        error=error+yaw_error*horizontal+sensor_error+motion[:,None,None,None]
                        contained &= distance>=offset[:,:,None,None]+error+cfg.intersection_epsilon
                    contained &= f.volume_valid[:,r0:r0+s.beam_chunk,None,None] & ok[:,None,:,None]
                    box_age=torch.minimum(box_age,torch.where(contained.any(1),upper[:,None,None],torch.inf))
            covered=torch.isfinite(box_age)
            band_free[:,start:end]=covered.all(-1) & valid_support.flatten(1)[:,start:end]
            band_age[:,start:end]=torch.where(band_free[:,start:end],box_age.amax(-1),torch.inf)
            fraction[:,start:end]=covered.float().mean(-1)
        states.append(band_free.reshape(n,grid.ny,grid.nx));ages.append(band_age.reshape(n,grid.ny,grid.nx));fractions.append(fraction.reshape(n,grid.ny,grid.nx))
    return VolumeProof(torch.stack(states,1),torch.stack(ages,1),torch.stack(fractions,1))

class FreeEvidenceHistory:
    """Internal immutable capture cache; no partial free bits are refreshed.

    Caller supplies measured cumulative PATH bounds per capture at query time.
    Do not derive slip travel from displacement. Reset rows invalidate old epoch.
    Actor/OBS1 history remains independent and exactly three frames.
    """
    def __init__(self,settings=None):
        self.settings=settings or FreeEvidenceConfig.load();self.captures=[]
    def push(self,frusta,pose,support,time_ms,epoch):
        if self.captures:
            old=self.captures[0];same=old.epoch==epoch
            if bool((same & (time_ms<=old.time_ms)).any()):raise ValueError('non-increasing capture time')
        f=FreeFrusta(*(getattr(frusta,name).clone() for name in ('origins','normals','offsets','far_m','volume_valid')))
        cap=FreeCapture(f,pose.clone(),support.clone(),time_ms.clone(),epoch.clone())
        self.captures=[cap]+self.captures[:self.settings.max_captures-1]
    def reset(self,env_ids=None):
        if env_ids is None:self.captures=[];return
        from dataclasses import replace
        out=[]
        for cap in self.captures:
            epoch=cap.epoch.clone();epoch[env_ids]=-1;out.append(replace(cap,epoch=epoch))
        self.captures=out
    def measurement(self,pose,now_ms,epoch,travel_m,linear_speed_bound_mps,yaw_rate_bound_radps):
        return FreeMeasurement(tuple(self.captures),pose,now_ms,epoch,travel_m,linear_speed_bound_mps,yaw_rate_bound_radps,self.settings)


def limit_by_current_returns(measurement,resolved,*,backend=None):
    """A current real/self first return may only SHRINK even an old footprint.

    Contradictory producer free volumes never erase an observed barrier. Clip
    the whole affected frustum at its earliest observed axial barrier. This is
    a conservative Torch reference O(captures*frusta*current_returns), not a
    claim of realtime sensor-fusion throughput. No hidden dynamic GT is used.
    """
    from .map_backend import use_triton
    if backend not in (None,'torch','triton'):raise ValueError('clip backend')
    selected=backend=='triton' or (backend is None and resolved.origins.dtype==torch.float32 and all(c.frusta.origins.dtype==torch.float32 for c in measurement.captures) and use_triton(resolved.origins))
    if selected:
        from .free_evidence169_cuda import limit_by_current_returns_cuda
        return limit_by_current_returns_cuda(measurement,resolved)
    from dataclasses import replace
    m=measurement;s=m.settings
    blockers=resolved.occupied_hit|resolved.self_hit
    if not bool(blockers.any()):return m
    current=resolved.origins+resolved.directions*resolved.distances[...,None]
    out=[]
    for cap in m.captures:
        f=cap.frusta
        rot=_rotation(m.pose[:,3]-cap.pose[:,3]).to(f.origins.dtype)
        shift=(m.pose[:,:3]-cap.pose[:,:3]).to(f.origins.dtype)
        shift=torch.einsum('nij,nj->ni',_rotation(-cap.pose[:,3]).to(f.origins.dtype),shift)
        points=torch.einsum('nij,nqj->nqi',rot,current.to(f.origins.dtype))+shift[:,None]
        far=f.far_m.clone();valid=f.volume_valid.clone()
        for b in range(0,far.shape[1],s.beam_chunk):
            end=min(b+s.beam_chunk,far.shape[1]);nearest=far[:,b:end].clone()
            for q in range(0,points.shape[1],s.beam_chunk):
                delta=points[:,None,q:q+s.beam_chunk]-f.origins[:,b:end,None]
                distance=torch.einsum('nbqj,nbpj->nbqp',delta,f.normals[:,b:end])
                # UPDATE169 user-approved strengthening: a blocker before the
                # near cut-off still occludes the cone. Exclude only the near
                # plane from membership; NEVER carve past such a first hit.
                satisfies=distance>=f.offsets[:,b:end,None]-1e-6
                inside=satisfies[:,:,:,:4].all(-1)&satisfies[:,:,:,5]
                inside &= blockers[:,None,q:q+s.beam_chunk] & (cap.epoch==m.epoch)[:,None,None]
                axial=distance[:,:,:,4]
                negative=(inside & (axial<0)).any(-1)
                before_near=(inside & (axial<f.offsets[:,b:end,None,4]-1e-6)).any(-1)
                valid[:,b:end] &= ~(negative|before_near)
                nearest=torch.minimum(nearest,torch.where(inside&~negative[:,:,None],axial,torch.inf).amin(-1))
            far[:,b:end]=nearest.clamp_min(0)
        offsets=f.offsets.clone();offsets[:,:,5]=torch.maximum(offsets[:,:,5],-far)
        out.append(replace(cap,frusta=replace(f,offsets=offsets,far_m=far,volume_valid=valid)))
    return replace(m,captures=tuple(out))
