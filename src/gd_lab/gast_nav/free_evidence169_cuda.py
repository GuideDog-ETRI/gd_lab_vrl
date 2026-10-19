"""Full-box free proof and current-return clipping. No sampling relaxation.

Coarse tile/frustum tests ONLY eliminate disjoint candidates. Every positive
still checks every micro-box against all six eroded planes. Compaction is a
lossless bitset, with no per-tile candidate truncation.
"""
import math
from dataclasses import replace
import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from . import schema as S
from .map_types import WIDE_GRID

@tr.jit
def _support(SOURCE_GOOD,WORLD,CURRENT,X,Y,CA,SA,BOUND,INSIDE,NEWZ,OUT,
             NX:tl.constexpr,NY:tl.constexpr,CELLS:tl.constexpr,RES:tl.constexpr,X0:tl.constexpr,Y0:tl.constexpr,
             OVERLAP_EPS:tl.constexpr,EPS:tl.constexpr,MINZ:tl.constexpr,MAXZ:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0);idx=tl.program_id(1)*B+tl.arange(0,B);mask=idx<CELLS
    x=tl.load(X+e*CELLS+idx,mask,0);y=tl.load(Y+e*CELLS+idx,mask,0)
    ca=tl.load(CA+e);sa=tl.load(SA+e);bound=tl.load(BOUND+e)
    ix=tl.floor((x-X0)/RES+.5).to(tl.int32);iy=tl.floor((y-Y0)/RES+.5).to(tl.int32)
    same=tl.load(INSIDE+e*CELLS+idx,mask,0);count=tl.full((B,),0,tl.int32)
    first=tl.full((B,),0,x.dtype)
    for dy in tl.static_range(-1,2):
        for dx in tl.static_range(-1,2):
            cx=ix+dx;cy=iy+dy
            # Torch int64*Python float creates float32 even for double poses.
            sx=(cx.to(tl.float32)*RES+X0).to(x.dtype);sy=(cy.to(tl.float32)*RES+Y0).to(x.dtype)
            vx=sx-x;vy=sy-y
            overlap=(tl.abs(vx)<bound)&(tl.abs(vy)<bound)&(tl.abs(ca*vx+sa*vy)<bound)&(tl.abs(-sa*vx+ca*vy)<bound)
            real=overlap&(cx>=0)&(cx<NX)&(cy>=0)&(cy<NY)&mask
            j=tl.minimum(tl.maximum(cy,0),NY-1)*NX+tl.minimum(tl.maximum(cx,0),NX-1)
            h=tl.load(WORLD+e*CELLS+j,mask,0)
            good=tl.load(SOURCE_GOOD+e*CELLS+j,mask,0)
            first=tl.where(real&(count==0),h,first)
            same=same&~(real&((tl.abs(h-first)>OVERLAP_EPS)|~good|(h!=h)))
            count+=real.to(tl.int32)
    oldh=first-tl.load(NEWZ+e);current=tl.load(CURRENT+e*CELLS+idx,mask,0)
    ok=same&(count>0)&(current==current)&(current>=MINZ)&(current<=MAXZ)&(tl.abs(oldh-current)<=EPS)
    tl.store(OUT+e*CELLS+idx,ok,mask)

@tr.jit
def _candidates(PLANES,VALID,SUPPORT,SOK,ELIG,BUDGET,HALF,MASKS,
                N:tl.constexpr,K:tl.constexpr,R:tl.constexpr,RR:tl.constexpr,WORDS:tl.constexpr,
                NX:tl.constexpr,NY:tl.constexpr,TX:tl.constexpr,TY:tl.constexpr,T:tl.constexpr,
                RES:tl.constexpr,X0:tl.constexpr,Y0:tl.constexpr,LO:tl.constexpr,HI:tl.constexpr,BAND:tl.constexpr,SUPPORT_ERR:tl.constexpr):
    et=tl.program_id(0);e=et//T;tile=et%T;ci=tl.program_id(1)
    cx=(tile%TX)*8+tl.arange(0,64)%8;cy=(tile//TX)*8+tl.arange(0,64)//8
    real=(cx<NX)&(cy<NY);cell=cy*NX+cx
    h=tl.load(SUPPORT+e*NX*NY+cell,real,0)
    ok=tl.load(SOK+(ci*N+e)*NX*NY+cell,real,0)&real
    any_ok=tl.sum(ok.to(tl.int32),0)>0
    hmin=tl.min(tl.where(ok,h,float('inf')),0);hmax=tl.max(tl.where(ok,h,-float('inf')),0)
    x=X0+((tile%TX)*8+3.5)*RES;y=Y0+((tile//TX)*8+3.5)*RES
    z=(hmin+LO+hmax+HI)*.5;hz=(hmax+HI-hmin-LO)*.5
    r=tl.arange(0,RR);base=((ci*N+e)*R+r)*42
    possible=tl.load(VALID+(ci*N+e)*R+r,r<R,0)&any_ok&tl.load(ELIG+ci*N+e)
    for plane in tl.static_range(6):
        nx=tl.load(PLANES+base+plane*7,r<R,0);ny=tl.load(PLANES+base+plane*7+1,r<R,0);nz=tl.load(PLANES+base+plane*7+2,r<R,0)
        # Packed offsets are relative to global sensor-frame point, not origin.
        off=tl.load(PLANES+base+plane*7+3,r<R,0)
        maximum=nx*x+ny*y+nz*z+tl.abs(nx)*(4*RES)+tl.abs(ny)*(4*RES)+tl.abs(nz)*hz
        possible=possible&(maximum>=off-1.e-4)
    # Farkas necessary feasibility test for a whole micro-box. Nonnegative
    # combinations of left+right+far constraints must also hold. Bound their
    # combined normal over the ENTIRE tile AABB; residuals are NOT assumed zero.
    # This can prove an eroded narrow frustum too small for any micro-box, without
    # running every micro-box or relaxing the final six-plane predicate.
    mhx=tl.load(HALF);mhy=tl.load(HALF+1);mhz=tl.load(HALF+2)
    p=tl.load(BUDGET+(ci*N+e)*3);motion=tl.load(BUDGET+(ci*N+e)*3+2)
    fx=tl.load(PLANES+base+4*7,r<R,0);fy=tl.load(PLANES+base+4*7+1,r<R,0);fz=tl.load(PLANES+base+4*7+2,r<R,1)
    for pair in tl.static_range(2):
        sx=tl.full((RR,),0.,tl.float32);sy=tl.full((RR,),0.,tl.float32);sz=tl.full((RR,),0.,tl.float32);rhs=tl.full((RR,),0.,tl.float32);depthcoef=tl.full((RR,),0.,tl.float32)
        for side in tl.static_range(2):
            plane=pair*2+side
            nx=tl.load(PLANES+base+plane*7,r<R,0);ny=tl.load(PLANES+base+plane*7+1,r<R,0);nz=tl.load(PLANES+base+plane*7+2,r<R,0)
            axial=nx*fx+ny*fy+nz*fz
            transverse=lib.sqrt_rn((nx-axial*fx)*(nx-axial*fx)+(ny-axial*fy)*(ny-axial*fy)+(nz-axial*fz)*(nz-axial*fz))
            weight=1./tl.maximum(transverse,1.e-5)
            off=tl.load(PLANES+base+plane*7+3,r<R,0)
            horizontal=tl.load(PLANES+base+plane*7+4,r<R,0);vertical=tl.load(PLANES+base+plane*7+5,r<R,0)
            required=off+tl.abs(nx)*mhx+tl.abs(ny)*mhy+tl.abs(nz)*mhz+p*horizontal+SUPPORT_ERR*vertical+motion
            sx+=weight*nx;sy+=weight*ny;sz+=weight*nz;rhs+=weight*required;depthcoef+=weight*axial
        weight=tl.maximum(depthcoef,0.)
        nx=tl.load(PLANES+base+5*7,r<R,0);ny=tl.load(PLANES+base+5*7+1,r<R,0);nz=tl.load(PLANES+base+5*7+2,r<R,0)
        off=tl.load(PLANES+base+5*7+3,r<R,0);horizontal=tl.load(PLANES+base+5*7+4,r<R,0);vertical=tl.load(PLANES+base+5*7+5,r<R,0)
        required=off+tl.abs(nx)*mhx+tl.abs(ny)*mhy+tl.abs(nz)*mhz+p*horizontal+SUPPORT_ERR*vertical+motion
        sx+=weight*nx;sy+=weight*ny;sz+=weight*nz;rhs+=weight*required
        maximum=sx*x+sy*y+sz*z+tl.abs(sx)*(4*RES)+tl.abs(sy)*(4*RES)+tl.abs(sz)*hz
        possible=possible&(maximum>=rhs-1.e-4*(1+tl.abs(rhs)))
    bits=tl.where(possible,1 << (r%32),0).to(tl.uint32).reshape(WORDS,32)
    words=tl.sum(bits,1).to(tl.uint32)
    wi=tl.arange(0,WORDS)
    tl.store(MASKS+(((ci*N+e)*3+BAND)*T+tile)*WORDS+wi,words)

@tr.jit
def _proof(PLANES,ORIGINS,ORIGNORM,SOK,ELIG,BUDGET,UPPER,MASKS,SUPPORT,XY,OFFSETS,HALF,
           FREE,AGE,FRACTION,
           N:tl.constexpr,K:tl.constexpr,R:tl.constexpr,WORDS:tl.constexpr,NX:tl.constexpr,NY:tl.constexpr,
           TX:tl.constexpr,T:tl.constexpr,MTX:tl.constexpr,MT:tl.constexpr,CELLS:tl.constexpr,M:tl.constexpr,MM:tl.constexpr,BAND:tl.constexpr,
           EPS:tl.constexpr,SENSOR_FACTOR:tl.constexpr,SUPPORT_ERR:tl.constexpr):
    et=tl.program_id(0);e=et//T;tile=et%T
    mtile=((tile//TX)//2)*MTX+(tile%TX)//2
    ci0=tl.arange(0,16);cx=(tile%TX)*4+ci0%4;cy=(tile//TX)*4+ci0//4;cell=cy*NX+cx;real=(cx<NX)&(cy<NY)
    mx=tl.arange(0,MM);micro=mx<M
    x=tl.load(XY+cell*2,real,0)[:,None]+tl.load(OFFSETS+mx*3,micro,0)[None,:]
    y=tl.load(XY+cell*2+1,real,0)[:,None]+tl.load(OFFSETS+mx*3+1,micro,0)[None,:]
    h=tl.load(SUPPORT+e*CELLS+cell,real,0)
    z=h[:,None]+tl.load(OFFSETS+mx*3+2,micro,0)[None,:]
    hx=tl.load(HALF);hy=tl.load(HALF+1);hz=tl.load(HALF+2);hn=tl.load(HALF+3)
    best=tl.full((16,MM),float('inf'),tl.float32)
    for ci in range(K):
        upper=tl.load(UPPER+ci*N+e)
        valid_cell=tl.load(SOK+(ci*N+e)*CELLS+cell,real,0)&real
        need=(best>upper)&micro[None,:]&valid_cell[:,None]
        if tl.load(ELIG+ci*N+e)&(tl.sum(need.to(tl.int32))>0):
            p=tl.load(BUDGET+(ci*N+e)*3);yawfactor=tl.load(BUDGET+(ci*N+e)*3+1);motion=tl.load(BUDGET+(ci*N+e)*3+2)
            for wi in range(WORDS):
                word=tl.load(MASKS+(((ci*N+e)*3+BAND)*MT+mtile)*WORDS+wi)
                pending=tl.sum(((best>upper)&micro[None,:]&valid_cell[:,None]).to(tl.int32))
                while (word!=0)&(pending>0):
                    bit=lib.ffs(word.to(tl.int32))-1;r=wi*32+bit;word=word&(word-1)
                    base=((ci*N+e)*R+r)
                    ox=tl.load(ORIGINS+base*3);oy=tl.load(ORIGINS+base*3+1);oz=tl.load(ORIGINS+base*3+2)
                    qx=x-ox;qy=y-oy;qz=z-oz
                    radius=lib.sqrt_rn(qx*qx+qy*qy+qz*qz)+hn+tl.load(ORIGNORM+base)
                    sensor=SENSOR_FACTOR*radius;yaw=yawfactor*radius
                    contained=micro[None,:]&valid_cell[:,None]
                    for plane in tl.static_range(6):
                        nx=tl.load(PLANES+base*42+plane*7);ny=tl.load(PLANES+base*42+plane*7+1);nz=tl.load(PLANES+base*42+plane*7+2)
                        # Original relative offset retained exactly; do not subtract absolute offsets.
                        off=tl.load(PLANES+base*42+plane*7+6)
                        horizontal=tl.load(PLANES+base*42+plane*7+4);vertical=tl.load(PLANES+base*42+plane*7+5)
                        distance=(qx*nx+qy*ny+qz*nz)-(tl.abs(nx)*hx+tl.abs(ny)*hy+tl.abs(nz)*hz)
                        error=p*horizontal+SUPPORT_ERR*vertical
                        error=error+yaw*horizontal+sensor+motion
                        contained=contained&(distance>=off+error+EPS)
                    best=tl.minimum(best,tl.where(contained,upper,float('inf')))
                    pending=tl.sum(((best>upper)&micro[None,:]&valid_cell[:,None]).to(tl.int32))
    covered=(best<float('inf'))&micro[None,:]
    count=tl.sum(covered.to(tl.int32),1);full=count==M
    age=tl.max(tl.where(micro[None,:],best,0.),1)
    out=(e*3+BAND)*CELLS+cell
    tl.store(FREE+out,full,real);tl.store(AGE+out,tl.where(full,age,float('inf')),real)
    tl.store(FRACTION+out,count.to(tl.float32)/M,real)

@tr.jit
def _clip(POINTS,BLOCKERS,O,NORMAL,OFF,FAR,VALID,EPOCH,OUTF,OUTV,
          N:tl.constexpr,R:tl.constexpr,Q:tl.constexpr,QQ:tl.constexpr,CHUNK:tl.constexpr):
    pid=tl.program_id(0);e=pid//R;r=pid%R;q=tl.arange(0,QQ)
    base=e*R+r
    dx=tl.load(POINTS+(e*Q+q)*3,q<Q,0)-tl.load(O+base*3)
    dy=tl.load(POINTS+(e*Q+q)*3+1,q<Q,0)-tl.load(O+base*3+1)
    dz=tl.load(POINTS+(e*Q+q)*3+2,q<Q,0)-tl.load(O+base*3+2)
    inside=tl.load(BLOCKERS+e*Q+q,q<Q,0)&tl.load(EPOCH+e)
    axial=tl.full((QQ,),0.,tl.float32)
    for plane in tl.static_range(6):
        nx=tl.load(NORMAL+base*18+plane*3);ny=tl.load(NORMAL+base*18+plane*3+1);nz=tl.load(NORMAL+base*18+plane*3+2)
        distance=dx*nx+dy*ny+dz*nz
        if plane!=4:inside=inside&(distance>=tl.load(OFF+base*6+plane)-1.e-6)
        if plane==4:axial=distance
    # Match reference chunk-local negative semantics, including returned far/valid.
    negative=inside&(axial<0)
    neg_chunk=tl.sum(negative.reshape(QQ//CHUNK,CHUNK).to(tl.int32),1)>0
    excluded=tl.broadcast_to(neg_chunk[:,None],(QQ//CHUNK,CHUNK)).reshape(QQ)
    nearest=tl.minimum(tl.load(FAR+base),tl.min(tl.where(inside&~excluded,axial,float('inf')),0))
    before_near=inside&(axial<tl.load(OFF+base*6+4)-1.e-6)
    valid=tl.load(VALID+base)&(tl.sum((negative|before_near).to(tl.int32),0)==0)
    tl.store(OUTF+base,tl.maximum(nearest,0.));tl.store(OUTV+base,valid)


def _warp_support(cap,m,support,grid,cfg):
    from .free_evidence167 import _rotation
    n=len(support);xy=grid.xy(device=support.device,dtype=cap.pose.dtype)
    angle=m.pose[:,3]-cap.pose[:,3];ca=angle.cos();sa=angle.sin()
    delta=m.pose[:,:2]-cap.pose[:,:2];co=cap.pose[:,3].cos();si=cap.pose[:,3].sin()
    tx=co*delta[:,0]+si*delta[:,1];ty=-si*delta[:,0]+co*delta[:,1]
    x=ca[:,None,None]*xy[...,0]-sa[:,None,None]*xy[...,1]+tx[:,None,None]
    y=sa[:,None,None]*xy[...,0]+ca[:,None,None]*xy[...,1]+ty[:,None,None]
    half=grid.res/2;extent=half*(ca.abs()+sa.abs());eps=cfg.overlap_epsilon
    inside=(x-extent[:,None,None]>=grid.x0-half-eps)&(x+extent[:,None,None]<=grid.x0+(grid.nx-1)*grid.res+half+eps)&(y-extent[:,None,None]>=grid.y0-half-eps)&(y+extent[:,None,None]<=grid.y0+(grid.ny-1)*grid.res+half+eps)
    good=torch.isfinite(cap.support)&(cap.support>=cfg.min_z_m)&(cap.support<=cfg.max_z_m)
    world=(cap.support+cap.pose[:,2,None,None]).contiguous();out=torch.empty_like(support,dtype=torch.bool)
    _support[(n,tr.cdiv(grid.nx*grid.ny,128))](good.contiguous(),world,support,x.contiguous(),y.contiguous(),ca,sa,(half+extent-eps).contiguous(),inside.contiguous(),m.pose[:,2].contiguous(),out,grid.nx,grid.ny,grid.nx*grid.ny,grid.res,grid.x0,grid.y0,eps,cfg.intersection_epsilon,cfg.min_z_m,cfg.max_z_m,128,enable_fp_fusion=False)
    return out


_LAYOUT_CACHE169={}
def _layouts169(support,grid,settings):
    key=(support.device,support.dtype,grid,settings.xy_parts,settings.z_max_piece_m)
    if key not in _LAYOUT_CACHE169:
        layouts=[]
        for _,low,high in S.BANDS:
            nz=math.ceil((high-low)/settings.z_max_piece_m);nx=settings.xy_parts;dz=(high-low)/nz
            offsets=support.new_tensor([((i+.5)/nx*grid.res-grid.res/2,(j+.5)/nx*grid.res-grid.res/2,low+(z+.5)*dz) for i in range(nx) for j in range(nx) for z in range(nz)])
            h=support.new_tensor([grid.res/(2*nx),grid.res/(2*nx),dz/2]);layouts.append((offsets,torch.cat((h,h.norm()[None]))))
        # Bounded immutable cache; no unbounded capture/config retention.
        if len(_LAYOUT_CACHE169)>=16:_LAYOUT_CACHE169.clear()
        _LAYOUT_CACHE169[key]=(grid.xy(device=support.device,dtype=support.dtype).flatten(0,1).contiguous(),layouts)
    return _LAYOUT_CACHE169[key]


def volume_free_proof_cuda(support,measurement,*,grid=WIDE_GRID,config=None):
    from .map_config import load_config
    from .free_evidence167 import _validate,_rotation,pose_budget,VolumeProof
    from .packets import age_codes,age_upper_s
    cfg=config or load_config();m=measurement;s=m.settings;support=support.contiguous();_validate(m,support,grid)
    if support.dtype!=torch.float32 or not support.is_cuda:raise ValueError('CUDA proof requires float32 CUDA support')
    n=len(support);cells=grid.nx*grid.ny;k=len(m.captures)
    shape=(n,3,grid.ny,grid.nx)
    free=torch.zeros(shape,device=support.device,dtype=torch.bool);age=torch.full(shape,torch.inf,device=support.device);fraction=torch.zeros(shape,device=support.device)
    if k==0:return VolumeProof(free,age,fraction)
    r=max(cap.frusta.origins.shape[1] for cap in m.captures)
    if r==0:return VolumeProof(free,age,fraction)
    planes=[];origins=[];origin_norm=[];valids=[];soks=[];eligs=[];budgets=[];uppers=[]
    for ci,cap in enumerate(m.captures):
        raw_age=((m.now_ms-cap.time_ms).double()/1000).clamp_min(0)
        displacement=(m.pose[:,:2]-cap.pose[:,:2]).norm(dim=-1)
        if bool(((cap.epoch==m.epoch)&(m.travel_m[:,ci]+cfg.intersection_epsilon<displacement)).any()):raise ValueError('travel must bound displacement')
        upper=age_upper_s(age_codes(raw_age));eligible=(cap.epoch==m.epoch)&(upper<=cfg.ttl_s)
        if s.historical_obstacle_speed_bound_mps is None:eligible &= raw_age==0
        if not bool(eligible.any()):continue
        f=cap.frusta;nr=f.origins.shape[1]
        rot=_rotation(cap.pose[:,3]-m.pose[:,3]).to(support.dtype)
        shift=(cap.pose[:,:3]-m.pose[:,:3]).to(support.dtype);shift=torch.einsum('nij,nj->ni',_rotation(-m.pose[:,3]).to(support.dtype),shift)
        o=torch.einsum('nij,nrj->nri',rot,f.origins)+shift[:,None]
        normal=torch.einsum('nij,nrkj->nrki',rot,f.normals)
        off=f.offsets.clone();off[:,:,5]=torch.maximum(torch.maximum(off[:,:,5],torch.full_like(off[:,:,5],-cfg.ray_max_range_m)),-f.far_m)+torch.maximum(torch.full_like(f.far_m,s.range_error_floor_m),s.range_error_fraction*f.far_m)
        # Keep both absolute coarse offset and the exact relative proof offset.
        absolute=off+(normal*o[:,:,None]).sum(-1)
        plane=torch.cat((normal,absolute[...,None],normal[...,:2].norm(dim=-1)[...,None],normal[...,2].abs()[...,None],off[...,None]),-1)
        p,yaw=pose_budget(upper.to(support.dtype),m.travel_m[:,ci].to(support.dtype),m.linear_speed_bound_mps.to(support.dtype),m.yaw_rate_bound_radps.to(support.dtype),s)
        if s.same_sample_pose_cancellation:
            same=(cap.time_ms==m.now_ms)&(cap.pose==m.pose).all(-1)&(cap.epoch==m.epoch)&(m.travel_m[:,ci]==0)
            p=torch.where(same,0.,p);yaw=torch.where(same,0.,yaw)
        motion=upper.to(support.dtype)*(s.historical_obstacle_speed_bound_mps or 0.)
        budgets.append(torch.stack((p,2*torch.sin(yaw.clamp(max=math.pi)/2),motion),-1));uppers.append(upper.to(support.dtype));eligs.append(eligible)
        planes.append(F.pad(plane,(0,0,0,0,0,r-nr)));origins.append(F.pad(o,(0,0,0,r-nr)));origin_norm.append(F.pad(f.origins.norm(dim=-1),(0,r-nr)));valids.append(F.pad(f.volume_valid,(0,r-nr)));soks.append(_warp_support(cap,m,support,grid,cfg))
    k=len(planes)
    if k==0:return VolumeProof(free,age,fraction)
    plane=torch.stack(planes).contiguous();o=torch.stack(origins).contiguous();on=torch.stack(origin_norm).contiguous();valid=torch.stack(valids).contiguous();sok=torch.stack(soks).contiguous();elig=torch.stack(eligs).contiguous();budget=torch.stack(budgets).contiguous();upper=torch.stack(uppers).contiguous()
    tx,ty=tr.cdiv(grid.nx,4),tr.cdiv(grid.ny,4);tiles=tx*ty;rr=max(32,tr.next_power_of_2(r));words=rr//32
    mtx,mty=tr.cdiv(grid.nx,8),tr.cdiv(grid.ny,8);mtiles=mtx*mty
    masks=torch.empty((k,n,3,mtiles,words),device=support.device,dtype=torch.uint32)
    xy,layouts=_layouts169(support,grid,s)
    for band,(_,low,high) in enumerate(S.BANDS):
        offsets,half=layouts[band];micros=len(offsets)
        _candidates[(n*mtiles,k)](plane,valid,support.contiguous(),sok,elig,budget,half,masks,n,k,r,rr,words,grid.nx,grid.ny,mtx,mty,mtiles,grid.res,grid.x0,grid.y0,low,high,band,s.support_error_m,num_warps=4,enable_fp_fusion=False)
        _proof[(n*tiles,)](plane,o,on,sok,elig,budget,upper,masks,support.contiguous(),xy,offsets,half,free,age,fraction,n,k,r,words,grid.nx,grid.ny,tx,tiles,mtx,mtiles,cells,micros,tr.next_power_of_2(micros),band,cfg.intersection_epsilon,2*math.sin(min(math.radians(s.sensor_angle_error_deg),math.pi)/2),s.support_error_m,num_warps=8,enable_fp_fusion=False)
    return VolumeProof(free,age,fraction)


def limit_by_current_returns_cuda(measurement,resolved):
    from .free_evidence167 import _rotation
    m=measurement
    if not resolved.origins.is_cuda or resolved.origins.dtype!=torch.float32 or any(c.frusta.origins.dtype!=torch.float32 for c in m.captures):raise ValueError('CUDA clipping requires float32 CUDA measurements')
    blockers=resolved.occupied_hit|resolved.self_hit
    if not bool(blockers.any()):return m
    q=resolved.origins.shape[1]
    if q==0:return m
    # Reference negative handling is per settings.beam_chunk; non-power chunks
    # are unusual custom settings and use reference rather than change semantics.
    chunk=m.settings.beam_chunk
    if chunk&(chunk-1):
        from .free_evidence167 import limit_by_current_returns
        return limit_by_current_returns(m,resolved,backend='torch')
    current=resolved.origins+resolved.directions*resolved.distances[...,None];out=[]
    for cap in m.captures:
        f=cap.frusta;n,r=f.far_m.shape
        if r==0:out.append(cap);continue
        rot=_rotation(m.pose[:,3]-cap.pose[:,3]).to(f.origins.dtype)
        shift=(m.pose[:,:3]-cap.pose[:,:3]).to(f.origins.dtype);shift=torch.einsum('nij,nj->ni',_rotation(-cap.pose[:,3]).to(f.origins.dtype),shift)
        points=torch.einsum('nij,nqj->nqi',rot,current.to(f.origins.dtype))+shift[:,None]
        far=torch.empty_like(f.far_m);valid=torch.empty_like(f.volume_valid)
        _clip[(n*r,)](points.contiguous(),blockers.contiguous(),f.origins.contiguous(),f.normals.contiguous(),f.offsets.contiguous(),f.far_m.contiguous(),f.volume_valid.contiguous(),(cap.epoch==m.epoch).contiguous(),far,valid,n,r,q,max(chunk,tr.next_power_of_2(q)),chunk,num_warps=4,enable_fp_fusion=False)
        offsets=f.offsets.clone();offsets[:,:,5]=torch.maximum(offsets[:,:,5],-far)
        out.append(replace(cap,frusta=replace(f,offsets=offsets,far_m=far,volume_valid=valid)))
    return replace(m,captures=tuple(out))
