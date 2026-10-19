"""Fused independent block queries. Geometry never leaves the sensor stage."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _query(C,A,H,K,ACTIVE,S,CAMS,NEAR,OUT,WIT,
           B:tl.constexpr,NC:tl.constexpr,P:tl.constexpr,L:tl.constexpr,
           FAR:tl.constexpr,EPS:tl.constexpr):
    bid=tl.program_id(0);env=bid//B;block=bid%B
    p=tl.arange(0,L);mask=p<P
    cam=tl.load(CAMS+block);cp=(env*NC+cam)*P+p;ep=env*P+p
    active=tl.load(ACTIVE+cp,mask,other=0).to(tl.int1)
    x=tl.load(C+cp*3,mask,0);y=tl.load(C+cp*3+1,mask,0);z=tl.load(C+cp*3+2,mask,0)
    hx=tl.load(H+ep*3,mask,1);hy=tl.load(H+ep*3+1,mask,1);hz=tl.load(H+ep*3+2,mask,1)
    kind=tl.load(K+ep,mask,0)
    x0=tl.load(S+bid*4);x1=tl.load(S+bid*4+1);y0=tl.load(S+bid*4+2);y1=tl.load(S+bid*4+3)
    a00=tl.load(A+cp*9,mask,0);a01=tl.load(A+cp*9+1,mask,0);a02=tl.load(A+cp*9+2,mask,0)
    a10=tl.load(A+cp*9+3,mask,0);a11=tl.load(A+cp*9+4,mask,0);a12=tl.load(A+cp*9+5,mask,0)
    a20=tl.load(A+cp*9+6,mask,0);a21=tl.load(A+cp*9+7,mask,0);a22=tl.load(A+cp*9+8,mask,0)
    zr=tl.abs(a20)*hx+tl.abs(a21)*hy+tl.abs(a22)*hz
    lo=tl.maximum(0.,z-zr-EPS);hi=tl.minimum(FAR,z+zr+EPS);ok=active
    for j in tl.static_range(3):
        ax=tl.load(A+cp*9+j,mask,0);ay=tl.load(A+cp*9+3+j,mask,0);az=tl.load(A+cp*9+6+j,mask,0)
        half=tl.load(H+ep*3+j,mask,1);centre=ax*x+ay*y+az*z
        mn=az+tl.minimum(ax*x0,ax*x1)+tl.minimum(ay*y0,ay*y1)
        mx=az+tl.maximum(ax*x0,ax*x1)+tl.maximum(ay*y0,ay*y1)
        for side in tl.static_range(2):
            coef=mx if side==0 else -mn
            rhs=centre-half-EPS if side==0 else -centre-half-EPS
            zero=coef==0;ok=ok&(~zero|(rhs<=0))
            bound=rhs/tl.where(zero,1.,coef)
            lo=tl.where(coef>0,tl.maximum(lo,bound),lo)
            hi=tl.where(coef<0,tl.minimum(hi,bound),hi)
    for face in tl.static_range(4):
        nx=1. if face==0 else (-1. if face==1 else 0.)
        ny=1. if face==2 else (-1. if face==3 else 0.)
        nz=-x0 if face==0 else (x1 if face==1 else (-y0 if face==2 else y1))
        radius=tl.abs(nx*a00+ny*a10+nz*a20)*hx+tl.abs(nx*a01+ny*a11+nz*a21)*hy+tl.abs(nx*a02+ny*a12+nz*a22)*hz
        ok=ok&(nx*x+ny*y+nz*z+radius>=-EPS*tl.sqrt(1+nz*nz))
    lower=tl.min(tl.where(ok&(hi>=lo),lo,FAR),0)
    near=tl.load(NEAR+bid);witness=tl.full((L,),False,tl.int1)
    ox=-(a00*x+a10*y+a20*z);oy=-(a01*x+a11*y+a21*z);oz=-(a02*x+a12*y+a22*z)
    for ray in tl.static_range(5):
        u=x0 if ray<2 else (x1 if ray<4 else (x0+x1)*.5)
        v=y0 if ray==0 or ray==2 else (y1 if ray<4 else (y0+y1)*.5)
        dx=a00*u+a10*v+a20;dy=a01*u+a11*v+a21;dz=a02*u+a12*v+a22
        entry=tl.full((L,),-float('inf'),tl.float32);exit=tl.full((L,),float('inf'),tl.float32);boxok=active
        for j in tl.static_range(3):
            o=ox if j==0 else (oy if j==1 else oz)
            d=dx if j==0 else (dy if j==1 else dz)
            h=hx if j==0 else (hy if j==1 else hz)
            parallel=tl.abs(d)<1.e-10;safe=tl.where(parallel,1.,d)
            t0=(-h-o)/safe;t1=(h-o)/safe
            entry=tl.maximum(entry,tl.where(parallel,-float('inf'),tl.minimum(t0,t1)))
            exit=tl.minimum(exit,tl.where(parallel,float('inf'),tl.maximum(t0,t1)))
            boxok=boxok&~(parallel&(tl.abs(o)>h))
        box=boxok&(exit>=tl.maximum(entry,0.))&(((entry>=near)&(entry<=FAR))|((exit>=near)&(exit<=FAR)))
        qa=dx*dx+dy*dy+dz*dz;qb=ox*dx+oy*dy+oz*dz;qc=ox*ox+oy*oy+oz*oz-hx*hx
        disc=qb*qb-qa*qc;root=tl.sqrt(tl.maximum(disc,0.));t0=(-qb-root)/qa;t1=(-qb+root)/qa
        sphere=(disc>=0)&(((t0>=near)&(t0<=FAR))|((t1>=near)&(t1<=FAR)))
        qa=dx*dx+dy*dy;qb=ox*dx+oy*dy;qc=ox*ox+oy*oy-hx*hx
        disc=qb*qb-qa*qc;root=tl.sqrt(tl.maximum(disc,0.));safe=tl.where(qa>1.e-12,qa,1.)
        t0=(-qb-root)/safe;t1=(-qb+root)/safe
        cyl=(qa>1.e-12)&(disc>=0)&(((t0>=near)&(t0<=FAR)&(tl.abs(oz+t0*dz)<=hz))|((t1>=near)&(t1<=FAR)&(tl.abs(oz+t1*dz)<=hz)))
        for sign in tl.static_range(-1,2,2):
            t=(sign*hz-oz)/tl.where(tl.abs(dz)>1.e-12,dz,1.)
            cyl=cyl|((tl.abs(dz)>1.e-12)&(t>=near)&(t<=FAR)&((ox+t*dx)*(ox+t*dx)+(oy+t*dy)*(oy+t*dy)<=hx*hx))
        witness=witness|(active&tl.where(kind==0,box,tl.where(kind==1,cyl,sphere)))
    tl.store(OUT+bid,lower);tl.store(WIT+bid,tl.sum(witness.to(tl.int32),0)>0)


def query_cuda(centres,axes,half,kind,active,slopes,cams,near,max_m,epsilon):
    if centres.dtype!=torch.float32 or not centres.is_cuda:raise ValueError('Triton requires CUDA float32')
    n,c,p,_=centres.shape;b=len(cams)
    args=[t.contiguous() for t in (centres,axes,half,kind,active,slopes,cams,near)]
    lower=torch.empty((n,b),device=centres.device);witness=torch.empty((n,b),device=centres.device,dtype=torch.bool)
    _query[(n*b,)](*args,lower,witness,b,c,p,tr.next_power_of_2(p),max_m,epsilon,num_warps=4,enable_fp_fusion=False)
    return lower,witness
