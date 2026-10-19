"""Conservative continuous block first-surface sensor, Isaac primitives only.

Geometry is private to this sensor. Public measurement exposes no occupancy,
object id, support height, map, feasibility, or privileged obstacle geometry.
The queried prefix is a convex frustum from optical origin (not just rays).
OBB enclosing proxies cover boxes/cylinders/spheres; projection separation
proves disjointness. Incomplete SAT can reject free, never miss an obstacle.
"""
from dataclasses import dataclass
import math
import torch
from .free_evidence167 import FreeFrusta

BOX,CYLINDER,SPHERE=0,1,2
@dataclass(frozen=True)
class SensorPrimitives:
    centres: torch.Tensor # [N,P,3], sample yaw frame
    rotations: torch.Tensor # [N,P,3,3], primitive-local -> sample frame
    half: torch.Tensor # [N,P,3], cylinder r,r,h/2; sphere r,r,r
    kind: torch.Tensor # [N,P] int64
    active: torch.Tensor # [N,P] bool
    complete: torch.Tensor # [N] bool; false for unknown/unsupported scene geometry
    is_trunk: torch.Tensor | None = None # [N,P] exact single simplified trunk box

@dataclass(frozen=True)
class BlockMeasurement:
    depth_lower_m: torch.Tensor # [N,B] conservative OPTICAL-z, no centre-ray semantics
    volume_valid: torch.Tensor # [N,B]
    return_witness: torch.Tensor # [N,B] a real supported surface ray witness
    frusta: FreeFrusta
    cutoff_m: torch.Tensor # [N,B], before this is NEVER certified (includes aperture exit)


def validate(primitives,k,rotation,translation,blocks):
    n,p,_=primitives.centres.shape;g=primitives
    if n<1 or p<1:raise ValueError('nonempty geometry batch required')
    if k.ndim==3:k=k[None].expand(n,-1,-1,-1)
    c=k.shape[1]
    if k.shape!=(n,c,3,3) or rotation.shape!=(n,c,3,3) or translation.shape!=(n,c,3):raise ValueError('camera shapes')
    if g.rotations.shape!=(n,p,3,3) or g.half.shape!=(n,p,3) or g.kind.shape!=(n,p) or g.active.shape!=(n,p) or g.complete.shape!=(n,):raise ValueError('primitive shapes')
    if g.kind.dtype!=torch.int64 or g.active.dtype!=torch.bool or g.complete.dtype!=torch.bool:raise ValueError('primitive dtype')
    if blocks.ndim!=2 or blocks.shape[1]!=5 or blocks.dtype!=torch.int64 or len(blocks)==0:raise ValueError('blocks')
    for t in (k,rotation,translation,g.centres,g.rotations,g.half):
        if t.dtype!=g.centres.dtype or t.device!=g.centres.device or not bool(torch.isfinite(t).all()):raise ValueError('geometry/camera dtype/device/finite')
    if g.centres.dtype not in (torch.float32,torch.float64):raise ValueError('float geometry')
    for t in (g.kind,g.active,g.complete,blocks):
        if t.device!=g.centres.device:raise ValueError('device')
    eye=torch.eye(3,dtype=g.centres.dtype,device=g.centres.device)
    for r in (rotation,g.rotations):
        if not torch.allclose(r.transpose(-1,-2)@r,eye.expand_as(r),atol=1e-5) or not bool((torch.linalg.det(r)>0).all()):raise ValueError('rotation')
    if bool((g.active & (g.half<=0).any(-1)).any()):raise ValueError('nonpositive primitive')
    if not bool((k[:,:,0,0]>0).all() & (k[:,:,1,1]>0).all()):raise ValueError('focal')
    if not torch.allclose(k[:,:,2],k.new_tensor([0.,0.,1.]).expand(n,c,3)) or bool((k[:,:,0,1].abs()>1e-7).any()|(k[:,:,1,0].abs()>1e-7).any()):raise ValueError('rectified K')
    b=blocks
    if bool(((b[:,0]<0)|(b[:,0]>=c)|(b[:,1]<0)|(b[:,2]<0)|(b[:,3]<=b[:,1])|(b[:,4]<=b[:,2])).any()):raise ValueError('block extent')
    # Unsupported active shape invalidates whole scene; never silently skip it.
    supported=((g.kind>=BOX)&(g.kind<=SPHERE))|~g.active
    cylinder=g.kind==CYLINDER;sphere=g.kind==SPHERE
    correct=(~cylinder|((g.half[:,:,0]-g.half[:,:,1]).abs()<1e-6)) & (~sphere|((g.half-g.half[:,:,:1]).abs().amax(-1)<1e-6))
    complete=g.complete & (supported&correct).all(1)
    return k,complete


def camera_space(g,rotation,translation):
    inv=rotation.transpose(-1,-2)
    centres=torch.einsum('ncij,ncpj->ncpi',inv,g.centres[:,None]-translation[:,:,None])
    axes=torch.einsum('ncij,npjk->ncpik',inv,g.rotations)
    return centres,axes


def slopes_from_k(k,blocks):
    i=blocks[:,0];kk=k[:,i];fx,fy,cx,cy=kk[:,:,0,0],kk[:,:,1,1],kk[:,:,0,2],kk[:,:,1,2]
    return torch.stack(((blocks[None,:,1]-.5-cx)/fx,(blocks[None,:,3]-.5-cx)/fx,(blocks[None,:,2]-.5-cy)/fy,(blocks[None,:,4]-.5-cy)/fy),-1)


def _overlap_interval(centres,axes,half,active,slopes,max_m,epsilon):
    """Necessary cross-section/OBB projection inequalities, solved in optical z.

    The interval contains EVERY true intersection (may include false positives).
    Consequently its lower edge bounds first hit; its upper edge bounds last exit.
    No pixel/ray sampling or root search is used to establish this volume claim.
    """
    x0,x1,y0,y1=slopes.unbind(-1)
    h=half[:,None]
    zr=(axes[:,:,:,2].abs()*h).sum(-1)
    lo=(centres[...,2]-zr-epsilon).clamp_min(0)
    hi=(centres[...,2]+zr+epsilon).clamp_max(max_m)
    possible=active.clone()
    for j in range(3):
        ax=axes[...,j];centre=(centres*ax).sum(-1)
        a,b,c=ax.unbind(-1)
        mn=c+torch.minimum(a*x0[...,None],a*x1[...,None])+torch.minimum(b*y0[...,None],b*y1[...,None])
        mx=c+torch.maximum(a*x0[...,None],a*x1[...,None])+torch.maximum(b*y0[...,None],b*y1[...,None])
        # mx*z >= centre-h; -mn*z >= -(centre+h)
        for coef,rhs in ((mx,centre-h[...,j]-epsilon),(-mn,-centre-h[...,j]-epsilon)):
            zero=coef==0
            possible &= ~zero|(rhs<=0)
            bound=rhs/torch.where(zero,torch.ones_like(coef),coef)
            lo=torch.where(coef>0,torch.maximum(lo,bound),lo)
            hi=torch.where(coef<0,torch.minimum(hi,bound),hi)
    # Fixed cone face normals separate some OBBs that pass axis projections.
    for normal in ((1.,0.,-x0),(-1.,0.,x1),(0.,1.,-y0),(0.,-1.,y1)):
        nx,ny,nz=normal
        value=nx*centres[...,0]+ny*centres[...,1]+nz[...,None]*centres[...,2]
        radius=((nx*axes[...,0,:]+ny*axes[...,1,:]+nz[...,None,None]*axes[...,2,:]).abs()*h).sum(-1)
        possible &= value+radius>=-epsilon*torch.sqrt(1+nz[...,None].square())
    return lo,hi,possible&(hi>=lo)


def _witness(centres,axes,half,kind,active,slopes,near,far):
    """Exact rays (four corners+centre) against actual supported primitives.

    Witness establishes a returned block, NOT the continuous-volume guarantee.
    The prefix SAT above alone supplies that guarantee, including subpixel hits.
    """
    if torch.is_tensor(near):near=near[:,:,None]
    if torch.is_tensor(far):far=far[:,:,None]
    x0,x1,y0,y1=slopes.unbind(-1);result=torch.zeros_like(x0,dtype=torch.bool)
    for u,v in [(x0,y0),(x0,y1),(x1,y0),(x1,y1),((x0+x1)/2,(y0+y1)/2)]:
        d=torch.stack((u,v,torch.ones_like(u)),-1)
        o=-torch.einsum('nbpij,nbpi->nbpj',axes,centres)
        dl=torch.einsum('nbpij,nbi->nbpj',axes,d)
        parallel=dl.abs()<1e-10;safe=torch.where(parallel,1.,dl)
        aa=(-half[:,None]-o)/safe;bb=(half[:,None]-o)/safe
        entry=torch.where(parallel,-torch.inf,torch.minimum(aa,bb)).amax(-1)
        exit=torch.where(parallel,torch.inf,torch.maximum(aa,bb)).amin(-1)
        box=~(parallel&(o.abs()>half[:,None])).any(-1)&(exit>=entry.clamp_min(0))&(((entry>=near)&(entry<=far))|((exit>=near)&(exit<=far)))
        # General ray/sphere quadratic, including camera inside.
        def roots(a,b,c):
            disc=b*b-a*c;safe=torch.where(a.abs()>1e-12,a,torch.ones_like(a))
            return (-b-disc.clamp_min(0).sqrt())/safe,(-b+disc.clamp_min(0).sqrt())/safe,(disc>=0)&(a>1e-12)
        t0,t1,ok=roots(dl.square().sum(-1),(o*dl).sum(-1),o.square().sum(-1)-half[:,None,:,0].square())
        sphere=ok&(((t0>=near)&(t0<=far))|((t1>=near)&(t1<=far)))
        # Cylinder side plus both caps, exact in cylinder-local coordinates.
        t0,t1,ok=roots(dl[:,:,:,:2].square().sum(-1),(o[:,:,:,:2]*dl[:,:,:,:2]).sum(-1),o[:,:,:,:2].square().sum(-1)-half[:,None,:,0].square())
        cyl=torch.zeros_like(ok)
        for t in (t0,t1):cyl |= ok&(t>=near)&(t<=far)&((o[:,:,:,2]+t*dl[:,:,:,2]).abs()<=half[:,None,:,2])
        for sign in (-1.,1.):
            dz=dl[:,:,:,2];t=(sign*half[:,None,:,2]-o[:,:,:,2])/torch.where(dz.abs()>1e-12,dz,1.)
            xy=o[:,:,:,:2]+t[...,None]*dl[:,:,:,:2]
            cyl |= (dz.abs()>1e-12)&(t>=near)&(t<=far)&(xy.square().sum(-1)<=half[:,None,:,0].square())
        hit=torch.where(kind[:,None]==BOX,box,torch.where(kind[:,None]==CYLINDER,cyl,sphere))
        result |= (hit&active).any(-1)
    return result


def produce_blocks168(g,k,rotation,translation,blocks,*,near_m=.15,max_m=5.,epsilon_m=1e-5,backend='torch',self_trunk_aperture=False,context='deployment',image_size=(80,45)):
    """Only this sensor function sees geometry. Returned arrays contain none.

    Enclosing OBBs for cylinders/spheres cause early bounds, never late ones.
    Reject unsupported/incomplete scenes, optical-origin self occlusion, blocks
    without a real surface witness, and bounds before near. The explicit training-only trunk aperture is off by default.
    """
    if not (all(math.isfinite(v) for v in (near_m,max_m,epsilon_m)) and 0<=near_m<max_m and epsilon_m>0):raise ValueError('query limits')
    if self_trunk_aperture and context!='training_sim':raise ValueError('trunk aperture is training_sim only')
    k,complete=validate(g,k,rotation,translation,blocks)
    width,height=image_size
    if width<1 or height<1 or bool(((blocks[:,3]>width)|(blocks[:,4]>height)).any()):raise ValueError('outside calibrated image')
    slopes=slopes_from_k(k,blocks);cc,rr=camera_space(g,rotation,translation)
    cams=blocks[:,0]
    n,b=slopes.shape[:2]
    cutoff=torch.full((n,b),near_m,device=g.centres.device,dtype=g.centres.dtype)
    camera_active=g.active[:,None].expand(-1,k.shape[1],-1).clone()
    if self_trunk_aperture:
        if g.is_trunk is None or g.is_trunk.shape!=g.active.shape or g.is_trunk.dtype!=torch.bool or g.is_trunk.device!=g.active.device:raise ValueError('explicit trunk mask required')
        if bool((g.is_trunk.sum(1)>1).any()) or bool((g.is_trunk & (g.kind!=BOX)).any()):raise ValueError('one exact trunk box per env')
        index=g.is_trunk.long().argmax(1);row=torch.arange(n,device=index.device)
        tc=cc[row,:,index];tr=rr[row,:,index];th=g.half[row,index]
        origin_local=-torch.einsum('ncij,nci->ncj',tr,tc)
        inside=(origin_local.abs()<=th[:,None]).all(-1)&(g.is_trunk&g.active).any(1)[:,None]
        camera_active &= ~(inside[:,:,None]&g.is_trunk[:,None])
        # Upper bound on LAST whole-block trunk overlap; convexity forbids
        # later re-entry. Origins outside the trunk retain strict first-hit.
        cc1=tc[:,cams,None];rr1=tr[:,cams,None]
        act=inside[:,cams,None]
        _,exit_upper,overlap=_overlap_interval(cc1,rr1,th[:,None],act,slopes,max_m,epsilon_m)
        cutoff=torch.where(inside[:,cams],torch.maximum(cutoff,torch.where(overlap[...,0],exit_upper[...,0]+epsilon_m,cutoff)),cutoff)
    if backend=='torch':
        centres=cc[:,cams];axes=rr[:,cams];block_active=camera_active[:,cams]
        lower,_,overlap=_overlap_interval(centres,axes,g.half,block_active,slopes,max_m,epsilon_m)
        lo=torch.where(overlap,lower,max_m).amin(-1)
        witness=_witness(centres,axes,g.half,g.kind,block_active,slopes,cutoff,max_m)
    elif backend=='triton':
        from .producer168_cuda import query_cuda
        lo,witness=query_cuda(cc,rr,g.half,g.kind,camera_active,slopes,cams,cutoff,max_m,epsilon_m)
    else:raise ValueError('backend')
    valid=complete[:,None]&witness&(lo>cutoff+epsilon_m)
    depth=torch.where(valid,(lo-epsilon_m).clamp_min(0),0.)
    # Sensor outputs a whole-block certified lower depth, not a sampled pixel.
    x0,x1,y0,y1=slopes.unbind(-1);zero=torch.zeros_like(x0);one=torch.ones_like(x0)
    normal=torch.stack((torch.stack((one,zero,-x0),-1),torch.stack((-one,zero,x1),-1),torch.stack((zero,one,-y0),-1),torch.stack((zero,-one,y1),-1),torch.stack((zero,zero,one),-1),torch.stack((zero,zero,-one),-1)),2)
    normal=normal/normal.norm(dim=-1,keepdim=True)
    normal=torch.einsum('nbij,nbfj->nbfi',rotation[:,cams],normal)
    offsets=torch.zeros((n,b,6),device=depth.device,dtype=depth.dtype);offsets[:,:,4]=cutoff;offsets[:,:,5]=-depth
    frusta=FreeFrusta(translation[:,cams].clone(),normal,offsets,depth,valid)
    return BlockMeasurement(depth,valid,witness,frusta,cutoff)
