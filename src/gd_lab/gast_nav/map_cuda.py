"""Exact CUDA fast paths; CPU implementations remain the golden references.

Ray candidates enumerate dominant-axis slabs and three minor-axis cells.
A ray traverses at most two minor cells within one dominant slab; including
both neighbours also covers floating point boundary rounding. Final evidence
uses the original strict 3D AABB intersection and sub-bin boundary semantics.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _evidence(O,D,T,H,BITS,LOW,HIGH,R:tl.constexpr,NX:tl.constexpr,NY:tl.constexpr,RES:tl.constexpr,X0:tl.constexpr,Y0:tl.constexpr,
              EPS:tl.constexpr,ZMIN:tl.constexpr,ZMAX:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    ray=tl.program_id(0);env=ray//R
    ox=tl.load(O+ray*3);oy=tl.load(O+ray*3+1);oz=tl.load(O+ray*3+2)
    dx=tl.load(D+ray*3);dy=tl.load(D+ray*3+1);dz=tl.load(D+ray*3+2);stop=tl.load(T+ray)-EPS
    xmajor=tl.abs(dx)>=tl.abs(dy)
    lane=tl.arange(0,B);col=lane//3;neighbor=lane%3-1
    major_o=tl.where(xmajor,ox,oy);major_d=tl.where(xmajor,dx,dy)
    minor_o=tl.where(xmajor,oy,ox);minor_d=tl.where(xmajor,dy,dx)
    major0=tl.where(xmajor,X0,Y0);minor0=tl.where(xmajor,Y0,X0)
    major_parallel=tl.abs(major_d)<=EPS
    denom=tl.where(major_parallel,1.,major_d)
    # Same float32 centre +/- half construction as Grid.xy and reference.
    centre=major0+col*RES
    a=(centre-RES/2-major_o)/denom;b=(centre+RES/2-major_o)/denom
    start=tl.maximum(tl.minimum(a,b),0.);finish=tl.minimum(tl.maximum(a,b),stop)
    mid=(start+finish)*.5
    # Vertical rays use their fixed xy cell; only that dominant column is live.
    minor=tl.floor((minor_o+minor_d*mid-minor0)/RES+.5).to(tl.int32)+neighbor
    minor=tl.where(major_parallel,tl.floor((minor_o-minor0)/RES+.5).to(tl.int32)+neighbor,minor)
    ix=tl.where(xmajor,col,minor);iy=tl.where(xmajor,minor,col)
    valid=(ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)&(stop>0.)
    valid &= tl.where(major_parallel,tl.abs(major_o-centre)<RES/2-EPS,finish>=start)
    index=iy*NX+ix
    h=tl.load(H+env*NX*NY+index,valid,other=float('nan'))
    valid &= (h>=ZMIN)&(h<=ZMAX)
    cx=X0+ix*RES;cy=Y0+iy*RES
    lx=cx-RES/2;hx=cx+RES/2;ly=cy-RES/2;hy=cy+RES/2
    px=tl.abs(dx)<=EPS;py=tl.abs(dy)<=EPS;pz=tl.abs(dz)<=EPS
    ax=(lx-ox)/tl.where(px,1.,dx);bx=(hx-ox)/tl.where(px,1.,dx)
    ay=(ly-oy)/tl.where(py,1.,dy);by=(hy-oy)/tl.where(py,1.,dy)
    enter=tl.maximum(tl.where(px,-float('inf'),tl.minimum(ax,bx)),tl.where(py,-float('inf'),tl.minimum(ay,by)))
    leave=tl.minimum(tl.where(px,float('inf'),tl.maximum(ax,bx)),tl.where(py,float('inf'),tl.maximum(ay,by)))
    boundary=(px&((ox<=lx+EPS)|(ox>=hx-EPS)))|(py&((oy<=ly+EPS)|(oy>=hy-EPS)))
    valid &= ~boundary
    bits=tl.full((B,),0,tl.int32)
    for k in tl.static_range(K):
        lo=h+tl.load(LOW+k);hi=h+tl.load(HIGH+k)
        az=(lo-oz)/tl.where(pz,1.,dz);bz=(hi-oz)/tl.where(pz,1.,dz)
        entry=tl.maximum(enter,tl.where(pz,-float('inf'),tl.minimum(az,bz)))
        exit=tl.minimum(leave,tl.where(pz,float('inf'),tl.maximum(az,bz)))
        edge=pz&((oz<=lo+EPS)|(oz>=hi-EPS))
        crossed=valid&~edge&(tl.minimum(exit,stop)-tl.maximum(entry,0.)>EPS)
        bits |= tl.where(crossed,1<<k,0)
    tl.atomic_or(BITS+env*NX*NY+index,bits,valid&(bits!=0),sem='relaxed')


def ray_evidence_cuda(resolved,support_height,grid,cfg):
    from .obstacle_map import subbin_edges
    from . import schema as S
    n,r=resolved.distances.shape;c=grid.nx*grid.ny
    if support_height.shape!=(n,grid.ny,grid.nx):raise ValueError('support/grid shape')
    edges=subbin_edges();flat=[p for group in edges for p in group]
    bits=torch.zeros((n,c),dtype=torch.int32,device=support_height.device)
    _evidence[(n*r,)](resolved.origins.contiguous(),resolved.directions.contiguous(),resolved.distances.contiguous(),support_height.contiguous(),bits,
        support_height.new_tensor([p[0] for p in flat]),support_height.new_tensor([p[1] for p in flat]),r,grid.nx,grid.ny,grid.res,grid.x0,grid.y0,cfg.intersection_epsilon,cfg.min_z_m,cfg.max_z_m,
        len(flat),triton.next_power_of_2(max(grid.nx,grid.ny)*3),enable_fp_fusion=False)
    support=support_height.flatten(1);known=torch.isfinite(support)&(support>=cfg.min_z_m)&(support<=cfg.max_z_m)
    end=resolved.origins+resolved.directions*resolved.distances[...,None]
    ix=torch.floor((end[...,0]-grid.x0)/grid.res+.5).long();iy=torch.floor((end[...,1]-grid.y0)/grid.res+.5).long()
    hit_in=resolved.occupied_hit&(ix>=0)&(ix<grid.nx)&(iy>=0)&(iy<grid.ny)
    index=iy.clamp(0,grid.ny-1)*grid.nx+ix.clamp(0,grid.nx-1);hit_z=end[...,2]-support.gather(1,index)
    result=[];diagnostics=[];offset=0
    for band,group in enumerate(edges):
        masks=1<<torch.arange(offset,offset+len(group),device=bits.device,dtype=torch.int32)
        passed=(bits[...,None]&masks)!=0;offset+=len(group)
        state=torch.where(passed.all(-1)&known,S.FREE,S.UNKNOWN).to(torch.uint8)
        _,lo,hi=S.BANDS[band]
        hits=hit_in&known.gather(1,index)&(hit_z>=lo)&((hit_z<hi) if band<2 else (hit_z<=hi))
        occupied=torch.zeros_like(bits);occupied.scatter_add_(1,index,hits.int())
        result.append(torch.where(occupied>0,S.OCCUPIED,state))
        diagnostics.append(passed.reshape(n,grid.ny,grid.nx,len(group)))
    return torch.stack(result,1).reshape(n,3,grid.ny,grid.nx),tuple(diagnostics)


@triton.jit
def _clearance(STATE,OFFX,OFFY,DIST,OUT,NX:tl.constexpr,NY:tl.constexpr,CELLS:tl.constexpr,NOFF:tl.constexpr,B:tl.constexpr):
    item=tl.program_id(0);env=item//(2*CELLS);band=(item//CELLS)%2;cell=item%CELLS
    k=tl.arange(0,B);dx=tl.load(OFFX+k,k<NOFF,other=0);dy=tl.load(OFFY+k,k<NOFF,other=0)
    x=cell%NX+dx;y=cell//NX+dy;mask=(k<NOFF)&(x>=0)&(x<NX)&(y>=0)&(y<NY)
    index=y*NX+x
    a=tl.load(STATE+env*3*CELLS+tl.where(band==0,0,2)*CELLS+index,mask,other=0)
    b=tl.load(STATE+env*3*CELLS+CELLS+index,mask&(band==0),other=0)
    distance=tl.load(DIST+k,k<NOFF,other=.5)
    value=tl.min(tl.where(mask&((a==2)|(b==2)),distance,.5),0)
    tl.store(OUT+item,value)

_OFFSETS={}
def clearance_cuda(state,grid):
    import math
    from . import schema as S
    key=(grid.res,str(state.device));offsets=_OFFSETS.get(key)
    if offsets is None:
        radius=math.ceil(S.NEAR_CLEAR_MAX/grid.res+.5);values=[]
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                d=math.hypot(max(abs(dx)*grid.res-grid.res/2,0),max(abs(dy)*grid.res-grid.res/2,0))
                if d<=S.NEAR_CLEAR_MAX:values.append((dx,dy,d))
        offsets=tuple(torch.tensor([v[i] for v in values],device=state.device,dtype=torch.int32 if i<2 else torch.float32) for i in range(3));_OFFSETS[key]=offsets
    result=torch.empty((state.shape[0],2,grid.ny,grid.nx),device=state.device)
    _clearance[(result.numel(),)](state.contiguous(),*offsets,result,grid.nx,grid.ny,grid.nx*grid.ny,len(offsets[0]),triton.next_power_of_2(len(offsets[0])))
    return result

@triton.jit
def _moments(P,V,H,COV,COUNT,R:tl.constexpr,NX:tl.constexpr,NY:tl.constexpr,RES:tl.constexpr,X0:tl.constexpr,Y0:tl.constexpr,RADIUS2:tl.constexpr,B:tl.constexpr):
    item=tl.program_id(0);env=item//(NX*NY);cell=item%(NX*NY)
    z=tl.load(H+item);x=X0+(cell%NX)*RES;y=Y0+(cell//NX)*RES
    # Absent observed cells can never be support: avoid point scans for them.
    if (z!=float('inf')) & (z==z):
        k=tl.arange(0,B);valid=tl.load(V+env*R+k,k<R,other=0)
        px=tl.load(P+(env*R+k)*3,k<R,other=0.);py=tl.load(P+(env*R+k)*3+1,k<R,other=0.);pz=tl.load(P+(env*R+k)*3+2,k<R,other=0.)
        nearby=valid&((px-x)*(px-x)+(py-y)*(py-y)+(pz-z)*(pz-z)<=RADIUS2)
        count=tl.sum(nearby.to(tl.int32),0);denom=tl.maximum(count,1)
        mx=tl.sum(tl.where(nearby,px,0.),0)/denom;my=tl.sum(tl.where(nearby,py,0.),0)/denom;mz=tl.sum(tl.where(nearby,pz,0.),0)/denom
        ax=px-mx;ay=py-my;az=pz-mz
        xx=tl.sum(tl.where(nearby,ax*ax,0.),0)/denom;yy=tl.sum(tl.where(nearby,ay*ay,0.),0)/denom;zz=tl.sum(tl.where(nearby,az*az,0.),0)/denom
        xy=tl.sum(tl.where(nearby,ax*ay,0.),0)/denom;xz=tl.sum(tl.where(nearby,ax*az,0.),0)/denom;yz=tl.sum(tl.where(nearby,ay*az,0.),0)/denom
        tl.store(COV+item*9,xx);tl.store(COV+item*9+1,xy);tl.store(COV+item*9+2,xz)
        tl.store(COV+item*9+3,xy);tl.store(COV+item*9+4,yy);tl.store(COV+item*9+5,yz)
        tl.store(COV+item*9+6,xz);tl.store(COV+item*9+7,yz);tl.store(COV+item*9+8,zz)
        tl.store(COUNT+item,count)


def fit_surface_cuda(points,valid,grid,settings):
    n,r,_=points.shape
    finite=valid.bool()&torch.isfinite(points).all(-1);safe=torch.where(finite[...,None],points,0.).contiguous()
    ix=torch.floor((safe[...,0]-grid.x0)/grid.res+.5).long();iy=torch.floor((safe[...,1]-grid.y0)/grid.res+.5).long()
    inside=finite&(ix>=0)&(ix<grid.nx)&(iy>=0)&(iy<grid.ny)
    index=iy.clamp(0,grid.ny-1)*grid.nx+ix.clamp(0,grid.nx-1)
    heights=points.new_full((n,grid.nx*grid.ny),torch.inf)
    heights.scatter_reduce_(1,index,torch.where(inside,safe[...,2],torch.inf),reduce='amin',include_self=True)
    observed=torch.isfinite(heights)
    covariance=points.new_zeros(n,grid.nx*grid.ny,3,3);count=torch.zeros_like(heights,dtype=torch.int32)
    _moments[(heights.numel(),)](safe,finite.contiguous(),heights,covariance,count,r,grid.nx,grid.ny,grid.res,grid.x0,grid.y0,
        settings['support_normal_radius_m']**2,triton.next_power_of_2(r),enable_fp_fusion=False)
    # cuSOLVER batched Jacobi has a batch-size limit. Solve only observed
    # candidates, in bounded chunks; rejected/unobserved cells stay NaN.
    active=(observed&(count>=settings['normal_min_points'])).flatten().nonzero().flatten()
    normals=points.new_full((heights.numel(),3),torch.nan)
    matrices=covariance.reshape(-1,3,3)
    for start in range(0,len(active),32768):
        index=active[start:start+32768]
        eigval,eigvec=torch.linalg.eigh(matrices[index]);normal=eigvec[...,0]
        normal=torch.where(normal[...,2:3]<0,-normal,normal)
        good=eigval[...,1]>settings['support_min_scatter']
        normals[index]=torch.where(good[...,None],normal,torch.nan)
    shape=(n,grid.ny,grid.nx)
    return torch.where(observed,heights,torch.nan).reshape(shape),observed.reshape(shape),normals.reshape(*shape,3)
