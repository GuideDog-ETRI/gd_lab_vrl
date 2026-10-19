"""GAST-Nav v0.3.3 assumed coverage measurements (UPDATE174B).

NO hardware FoV/specification, D4xx, robot mesh, trunk aperture or GT map.
Six cube charts merely partition abstract coverage into continuous volumes.
Scene-only first-hit queries clip free before actual static/dynamic occluders.
"""
from dataclasses import dataclass,replace
import math
import torch
from .producer168 import SensorPrimitives,BlockMeasurement,produce_blocks168
from .free_evidence167 import FreeFrusta

@dataclass(frozen=True)
class CoverageConfig174:
    hip_x_m: float = -.31
    # Virtual measurement origins, NOT a proposed sensor kind/mount/spec.
    front_origins: tuple = ((.28,0.,.12),(.10,0.,.18)) # Ouster + Mid-360 coverage
    rear_origins: tuple = ((-.39,.24,.10),(-.39,-.24,.10)) # unspecified distance sensor
    mount_jitter_m: tuple = (.03,.03,.03)
    angle_jitter_deg: tuple = (5.,5.,10.)
    chart_divisions: int = 10 # numeric volume partition, not ray density/specification
    query_radius_m: float = 6.0 # map-local query bound, not physical detection range
    numerical_near_m: float = .001
    noise_floor_m: float = .02
    noise_fraction: float = .02
    missing_probability: float = .05
    occluded_block_probability: float = .02 # optional unmodelled sensor obstruction, invalid

    @classmethod
    def load(cls,path):
        import yaml
        with open(path) as stream: data=yaml.safe_load(stream)
        if data.pop('model')!='v033_assumed_coverage':raise ValueError('explicit coverage assumption required')
        for key in ('front_origins','rear_origins'):
            data[key]=tuple(tuple(v) for v in data[key])
        for key in ('mount_jitter_m','angle_jitter_deg'):data[key]=tuple(data[key])
        cfg=cls(**data);validate_config174(cfg);return cfg


def validate_config174(c):
    if not isinstance(c.chart_divisions,int) or not 2<=c.chart_divisions<=64:raise ValueError('numeric partition')
    if len(c.front_origins)!=2 or not c.rear_origins:raise ValueError('front pair + unspecified rear coverage required')
    if len(c.mount_jitter_m)!=3 or len(c.angle_jitter_deg)!=3:raise ValueError('jitter xyz/rpy')
    vals=(c.hip_x_m,c.query_radius_m,c.numerical_near_m,c.noise_floor_m,c.noise_fraction,c.missing_probability,c.occluded_block_probability,*c.mount_jitter_m,*c.angle_jitter_deg)
    if not all(math.isfinite(v) for v in vals):raise ValueError('finite config')
    if not 0<c.numerical_near_m<c.query_radius_m/math.sqrt(3):raise ValueError('query limits')
    if min(*c.mount_jitter_m,*c.angle_jitter_deg,c.noise_floor_m,c.noise_fraction)<0:raise ValueError('noise/jitter')
    if not 0<=c.missing_probability<=1 or not 0<=c.occluded_block_probability<=1:raise ValueError('missing probability')
    for front,origins in ((True,c.front_origins),(False,c.rear_origins)):
        for p in origins:
            if len(p)!=3 or not all(math.isfinite(v) for v in p):raise ValueError('virtual origin')
            # Keep origins inside their promised halfspace under ALL xyz jitters.
            if (front and p[0]-c.mount_jitter_m[0]<=c.hip_x_m) or (not front and p[0]+c.mount_jitter_m[0]>=c.hip_x_m):raise ValueError('coverage origin must stay inside region')


def sample_mounts174(n,cfg,*,device='cpu',dtype=torch.float32,generator=None):
    """Sample once/reset; caller stores [N,M,6] xyz(m), rpy(degrees) + RNG."""
    validate_config174(cfg);m=len(cfg.front_origins)+len(cfg.rear_origins)
    scale=torch.tensor((*cfg.mount_jitter_m,*cfg.angle_jitter_deg),device=device,dtype=dtype)
    return (2*torch.rand((n,m,6),device=device,dtype=dtype,generator=generator)-1)*scale


def _rpy(a):
    r,p,y=torch.deg2rad(a).unbind(-1);cr,sr=r.cos(),r.sin();cp,sp=p.cos(),p.sin();cy,sy=y.cos(),y.sin()
    return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr),-1).reshape(*r.shape,3,3)


def layout174(cfg,*,device='cpu',dtype=torch.float32):
    """Cube charts cover every direction, no invented rear hardware FoV."""
    directions=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    matrices=[]
    for d in directions:
        z=torch.tensor(d,device=device,dtype=dtype)
        up=torch.tensor((0,1,0) if d[2] else (0,0,1),device=device,dtype=dtype)
        x=torch.linalg.cross(up,z);y=torch.linalg.cross(z,x)
        matrices.append(torch.stack((x,y,z),1))
    r=torch.stack(matrices);v=cfg.chart_divisions
    k=torch.tensor([[v/2,0,(v-1)/2],[0,v/2,(v-1)/2],[0,0,1]],device=device,dtype=dtype).expand(6,-1,-1).clone()
    blocks=torch.tensor([(face,x,y,x+1,y+1) for face in range(6) for y in range(v) for x in range(v)],device=device,dtype=torch.long)
    return k,r,blocks


def produce_coverage174(scene,cfg,*,scene_only,scene_capture_id,pose_capture_id,
                        mount_deltas=None,generator=None,available=None,
                        corrupt=True,backend='torch'):
    """Return unchanged BlockMeasurement [N,B] / FreeFrusta [N,B,6,3].

    B=(2+len(rear_origins))*6*chart_divisions**2, nominal 2400. Order:
    front Ouster, front Mid-360, rear L/R; chart, y, x. Input scene is private
    WORLD geometry already transformed into the true capture body-yaw frame,
    with static+dynamic actual obstacles/floor, complete inventory, NO self
    snapshot/mesh. Capture IDs [N] int64 include reset identity and must match.
    Attach separate odometry/time/epoch to immutable FreeCapture afterwards.
    No D4xx/terrain return can be passed into this API.

    Front free prefixes are confined x>=hip_x; rear x<=hip_x, including both
    leg sides. A crossing chart is shortened by its worst corner, not centre.
    Position/angle changes are virtual coverage perturbations, not hardware
    FoV. Missing/blocked measurement invalid. Real obstacles use first hit;
    no witness and incomplete scene invalid. Radius/partition/first-hit bound
    approximation may reject evidence; never fills unknown by assumption.
    """
    validate_config174(cfg)
    if scene_only is not True:raise ValueError('scene-only actual obstacle inventory required')
    if scene.is_trunk is not None and bool(scene.is_trunk.any()):raise ValueError('robot snapshot not a coverage scene')
    n=len(scene.centres);dev=scene.centres.device;dt=scene.centres.dtype
    for ids in (scene_capture_id,pose_capture_id):
        if ids.shape!=(n,) or ids.dtype!=torch.int64 or ids.device!=dev or bool((ids<0).any()):raise ValueError('capture ids')
    if not torch.equal(scene_capture_id,pose_capture_id):raise ValueError('unsynchronised capture')
    origins=cfg.front_origins+cfg.rear_origins;m=len(origins)
    if mount_deltas is None:mount_deltas=scene.centres.new_zeros((n,m,6))
    bounds=scene.centres.new_tensor((*cfg.mount_jitter_m,*cfg.angle_jitter_deg))
    if mount_deltas.shape!=(n,m,6) or mount_deltas.device!=dev or mount_deltas.dtype!=dt or not bool(torch.isfinite(mount_deltas).all()) or bool((mount_deltas.abs()>bounds+1e-6).any()):raise ValueError('mount deltas')
    k,charts,blocks=layout174(cfg,device=dev,dtype=dt);b=len(blocks)
    if available is not None and (available.shape!=(n,m*b) or available.dtype!=torch.bool or available.device!=dev):raise ValueError('availability')
    values=[]
    for i,nominal in enumerate(origins):
        t=scene.centres.new_tensor(nominal)[None]+mount_deltas[:,i,:3]
        r=_rpy(mount_deltas[:,i,3:])[:,None]@charts[None]
        q=produce_blocks168(scene,k,r,t[:,None].expand(-1,6,-1),blocks,
            near_m=cfg.numerical_near_m,max_m=cfg.query_radius_m/math.sqrt(3),backend=backend,image_size=(cfg.chart_divisions,cfg.chart_divisions))
        # Whole cone stays in promised front/rear coverage. Compute each
        # corner's sample x displacement per unit optical z.
        v=cfg.chart_divisions;u0=2*blocks[:,1]/v-1;u1=2*blocks[:,3]/v-1
        v0=2*blocks[:,2]/v-1;v1=2*blocks[:,4]/v-1
        rx=r[:,blocks[:,0],0,:]
        corners=torch.stack([rx[:,:,0]*u+rx[:,:,1]*vv+rx[:,:,2] for u in (u0,u1) for vv in (v0,v1)],-1)
        front=i<len(cfg.front_origins)
        toward=-corners.amin(-1) if front else corners.amax(-1)
        gap=t[:,0]-cfg.hip_x_m if front else cfg.hip_x_m-t[:,0]
        cap=torch.where(toward>1e-8,gap[:,None]/toward.clamp_min(1e-8),torch.inf)
        depth=torch.minimum(q.depth_lower_m,cap)
        valid=q.volume_valid.clone()
        if corrupt:
            err=torch.maximum(torch.full_like(depth,cfg.noise_floor_m),depth*cfg.noise_fraction)
            noise=(2*torch.rand(depth.shape,device=dev,dtype=dt,generator=generator)-1)*err
            depth=(depth+noise-err).clamp_min(0)
            keep=(torch.rand(depth.shape,device=dev,dtype=dt,generator=generator)>=cfg.missing_probability)&(torch.rand(depth.shape,device=dev,dtype=dt,generator=generator)>=cfg.occluded_block_probability)
            valid &= keep
        if available is not None:valid &= available[:,i*b:(i+1)*b]
        valid &= depth>cfg.numerical_near_m+1e-5
        depth=torch.where(valid,depth,0.)
        offsets=q.frusta.offsets.clone();offsets[:,:,5]=-depth
        f=FreeFrusta(q.frusta.origins,q.frusta.normals,offsets,depth,valid)
        values.append(BlockMeasurement(depth,valid,q.return_witness&valid,f,q.cutoff_m))
    cat=lambda name:torch.cat([getattr(q,name) for q in values],1)
    f=FreeFrusta(*(torch.cat([getattr(q.frusta,key) for q in values],1) for key in ('origins','normals','offsets','far_m','volume_valid')))
    return BlockMeasurement(cat('depth_lower_m'),cat('volume_valid'),cat('return_witness'),f,cat('cutoff_m'))


def scene_in_sample174(world,sample_pose):
    """Private WORLD -> true sample yaw transform; creates NO robot geometry."""
    n=len(world.centres)
    if sample_pose.shape!=(n,4) or sample_pose.device!=world.centres.device or sample_pose.dtype!=world.centres.dtype or not bool(torch.isfinite(sample_pose).all()):raise ValueError('true sensor sample pose')
    c,s=sample_pose[:,3].cos(),sample_pose[:,3].sin();z=torch.zeros_like(c);one=torch.ones_like(c)
    inv=torch.stack((c,s,z,-s,c,z,z,z,one),-1).reshape(n,3,3)
    return replace(world,centres=torch.einsum('nij,npj->npi',inv,world.centres-sample_pose[:,None,:3]),rotations=inv[:,None]@world.rotations)


@dataclass(frozen=True)
class CoverageMeasurement174:
    free: BlockMeasurement
    rays: object # map_types.RayBatch, actual central-ray FIRST returns, not depth lower bounds
    valid_hit: torch.Tensor # [N,B], missing/no return invalid
    range_error_m: torch.Tensor # [N,B], bounded synthetic noise (not a hardware claim)
    height_error_m: torch.Tensor # [N,B], range-only vertical component; add angular/pose error separately


def measure_coverage174(scene,cfg,**kwargs):
    """Free evidence + actual range observations for occupied/support estimation.

    No GT occupancy/support is exported. A raw first-return point is an ordinary
    sensor observation; derive support using C2.1 from observed neighbours.
    NEVER use free.depth_lower_m as a hit. Dropped blocks cannot supply points.
    This optional companion preserves the original produce_coverage174 return
    contract. Caller supplies the same scene-only/IDs/mounts/generator options.
    """
    from .map_types import RayBatch
    q=produce_coverage174(scene,cfg,**kwargs)
    n=len(scene.centres);origins=cfg.front_origins+cfg.rear_origins
    delta=kwargs.get('mount_deltas')
    if delta is None:delta=scene.centres.new_zeros((n,len(origins),6))
    _,charts,blocks=layout174(cfg,device=scene.centres.device,dtype=scene.centres.dtype)
    v=cfg.chart_divisions
    local=torch.stack((2*(blocks[:,1]+.5)/v-1,2*(blocks[:,2]+.5)/v-1,torch.ones_like(blocks[:,1])),1).to(scene.centres.dtype)
    dirs=[]
    for i in range(len(origins)):
        rr=_rpy(delta[:,i,3:])[:,None]@charts[None,:, :, :]
        d=torch.einsum('nbij,bj->nbi',rr[:,blocks[:,0]],local)
        dirs.append(d/d.norm(dim=-1,keepdim=True))
    direction=torch.cat(dirs,1);o=q.frusta.origins
    # Exact local intersections for the actual supported primitive shapes.
    # Chunk rays to keep N*B*P temporaries bounded; no actor geometry escapes.
    first=[]
    for start in range(0,direction.shape[1],128):
        dd=direction[:,start:start+128];oo=o[:,start:start+128]
        inv=scene.rotations.transpose(-1,-2)
        origin=torch.einsum('npij,nbpj->nbpi',inv,oo[:,:,None]-scene.centres[:,None])
        d=torch.einsum('npij,nbj->nbpi',inv,dd)
        h=scene.half[:,None];parallel=d.abs()<1e-10;safe=torch.where(parallel,1.,d)
        aa=(-h-origin)/safe;bb=(h-origin)/safe
        entry=torch.where(parallel,-torch.inf,torch.minimum(aa,bb)).amax(-1).clamp_min(0)
        exit=torch.where(parallel,torch.inf,torch.maximum(aa,bb)).amin(-1)
        box=torch.where(~(parallel&(origin.abs()>h)).any(-1)&(exit>=entry),entry,torch.inf)
        def roots(a,b,c):
            disc=b*b-a*c;den=torch.where(a.abs()>1e-12,a,1.)
            return (-b-disc.clamp_min(0).sqrt())/den,(-b+disc.clamp_min(0).sqrt())/den,(disc>=0)&(a>1e-12)
        t0,t1,ok=roots(d.square().sum(-1),(origin*d).sum(-1),origin.square().sum(-1)-h[...,0].square())
        sphere=torch.where(origin.square().sum(-1)<=h[...,0].square(),0.,torch.where(ok&(t0>=0),t0,torch.inf))
        t0,t1,ok=roots(d[...,:2].square().sum(-1),(origin[...,:2]*d[...,:2]).sum(-1),origin[...,:2].square().sum(-1)-h[...,0].square())
        cyl=torch.full_like(box,torch.inf)
        for tt in (t0,t1):
            cyl=torch.minimum(cyl,torch.where(ok&(tt>=0)&((origin[...,2]+tt*d[...,2]).abs()<=h[...,2]),tt,torch.inf))
        for sign in (-1.,1.):
            dz=d[...,2];tt=(sign*h[...,2]-origin[...,2])/torch.where(dz.abs()>1e-12,dz,1.)
            xy=origin[...,:2]+tt[...,None]*d[...,:2]
            cyl=torch.minimum(cyl,torch.where((dz.abs()>1e-12)&(tt>=0)&(xy.square().sum(-1)<=h[...,0].square()),tt,torch.inf))
        cyl=torch.where((origin[...,:2].square().sum(-1)<=h[...,0].square())&(origin[...,2].abs()<=h[...,2]),0.,cyl)
        hit=torch.where(scene.kind[:,None]==0,box,torch.where(scene.kind[:,None]==1,cyl,sphere))
        first.append(torch.where(scene.active[:,None],hit,torch.inf).amin(-1))
    distance=torch.cat(first,1)
    valid=q.volume_valid & torch.isfinite(distance)&(distance>cfg.numerical_near_m)&(distance<=cfg.query_radius_m)
    # Only return hit points inside that sensor family's coverage region.
    x=o[:,:,0]+distance.nan_to_num(posinf=0)*direction[:,:,0];per=len(blocks)
    front=torch.arange(distance.shape[1],device=distance.device)<len(cfg.front_origins)*per
    valid &= torch.where(front[None],x>=cfg.hip_x_m,x<=cfg.hip_x_m)
    err=torch.maximum(torch.full_like(distance,cfg.noise_floor_m),distance.nan_to_num(posinf=0)*cfg.noise_fraction)
    if kwargs.get('corrupt',True):
        noise=(2*torch.rand(distance.shape,device=distance.device,dtype=distance.dtype,generator=kwargs.get('generator'))-1)*err
        distance=(distance+noise).clamp_min(cfg.numerical_near_m)
    else:err=torch.zeros_like(err)
    distance=torch.where(valid,distance,torch.inf)
    err=torch.where(valid,err,0.)
    rays=RayBatch(o,direction,distance,torch.zeros_like(valid))
    return CoverageMeasurement174(q,rays,valid,err,err*direction[:,:,2].abs())
