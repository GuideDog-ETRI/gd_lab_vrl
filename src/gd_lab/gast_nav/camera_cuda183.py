"""Bounded CUDA first-hit and XY-DDA for full-pixel UPDATE183 cameras.

Reference is oracle_occgrid176. Every load/store is masked to ray/grid extent;
no allocator aliases or persistent CUDA state. Source selection stays in Python.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _cast(O,D,C,A,H,K,ACTIVE,SELF,OUT,SH,R:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
 e=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);ok=j<R;ix=e*R+j
 ox=tl.load(O+ix*3,ok,0);oy=tl.load(O+ix*3+1,ok,0);oz=tl.load(O+ix*3+2,ok,0)
 dx=tl.load(D+ix*3,ok,0);dy=tl.load(D+ix*3+1,ok,0);dz=tl.load(D+ix*3+2,ok,0)
 best=tl.full((B,),float('inf'),tl.float32);selfbest=best
 for p in range(P):
  base=e*P+p;cx=tl.load(C+base*3);cy=tl.load(C+base*3+1);cz=tl.load(C+base*3+2)
  x=ox-cx;y=oy-cy;z=oz-cz
  ax=tl.load(A+base*9);ay=tl.load(A+base*9+3);az=tl.load(A+base*9+6)
  bx=tl.load(A+base*9+1);by=tl.load(A+base*9+4);bz=tl.load(A+base*9+7)
  cxr=tl.load(A+base*9+2);cyr=tl.load(A+base*9+5);czr=tl.load(A+base*9+8)
  x0=ax*x+ay*y+az*z;y0=bx*x+by*y+bz*z;z0=cxr*x+cyr*y+czr*z
  vx=ax*dx+ay*dy+az*dz;vy=bx*dx+by*dy+bz*dz;vz=cxr*dx+cyr*dy+czr*dz
  hx=tl.load(H+base*3);hy=tl.load(H+base*3+1);hz=tl.load(H+base*3+2)
  px=tl.abs(vx)<1e-10;py=tl.abs(vy)<1e-10;pz=tl.abs(vz)<1e-10
  x1=(-hx-x0)/tl.where(px,1.,vx);x2=(hx-x0)/tl.where(px,1.,vx)
  y1=(-hy-y0)/tl.where(py,1.,vy);y2=(hy-y0)/tl.where(py,1.,vy)
  z1=(-hz-z0)/tl.where(pz,1.,vz);z2=(hz-z0)/tl.where(pz,1.,vz)
  lo=tl.maximum(0.,tl.maximum(tl.where(px,-float('inf'),tl.minimum(x1,x2)),tl.maximum(tl.where(py,-float('inf'),tl.minimum(y1,y2)),tl.where(pz,-float('inf'),tl.minimum(z1,z2)))))
  hi=tl.minimum(tl.where(px,float('inf'),tl.maximum(x1,x2)),tl.minimum(tl.where(py,float('inf'),tl.maximum(y1,y2)),tl.where(pz,float('inf'),tl.maximum(z1,z2))))
  hit=tl.where((hi>=lo)&~((px&(tl.abs(x0)>hx))|(py&(tl.abs(y0)>hy))|(pz&(tl.abs(z0)>hz))),lo,float('inf'))
  kind=tl.load(K+base)
  aa=vx*vx+vy*vy+vz*vz;bb=x0*vx+y0*vy+z0*vz;cc=x0*x0+y0*y0+z0*z0-hx*hx
  disc=bb*bb-aa*cc;t=(-bb-tl.sqrt(tl.maximum(disc,0.)))/tl.maximum(aa,1e-12)
  sphere=tl.where(cc<=0.,0.,tl.where((disc>=0.)&(aa>1e-12)&(t>=0.),t,float('inf')))
  hit=tl.where(kind==2,sphere,hit)
  # This adapter accepts boxes/spheres only; unsupported kinds fail in wrapper.
  hit=tl.where(tl.load(ACTIVE+base),hit,float('inf'))
  best=tl.minimum(best,hit);selfbest=tl.minimum(selfbest,tl.where(tl.load(SELF+base),hit,float('inf')))
 tl.store(OUT+ix,best,ok);tl.store(SH+ix,(selfbest<float('inf'))&(selfbest<=best+1e-7),ok)


def cast183(scene,rays,self_mask):
 from .oracle_occgrid176 import OracleReturns176
 if bool((scene.active&((scene.kind!=0)&(scene.kind!=2))).any()):raise ValueError('183 C camera inventory boxes/spheres only')
 n,r=rays.near_m.shape
 out=torch.empty((n,r),device=rays.origins.device);sh=torch.empty_like(out,dtype=torch.bool)
 _cast[(n,triton.cdiv(r,128))](rays.origins.contiguous(),rays.directions.contiguous(),scene.centres.contiguous(),scene.rotations.contiguous(),scene.half.contiguous(),scene.kind.contiguous(),scene.active.contiguous(),self_mask.contiguous(),out,sh,r,scene.kind.shape[1],128,enable_fp_fusion=False)
 valid=rays.coverage_valid&scene.complete[:,None]&(out>=rays.near_m)&~sh
 return OracleReturns176(rays,torch.where(out<=rays.far_m,out,torch.inf),valid,sh&(out<=rays.far_m))

@triton.jit
def _dda(O,D,T,NEAR,FAR,V,S,H,BITS,R:tl.constexpr,NX:tl.constexpr,NY:tl.constexpr,X0:tl.constexpr,Y0:tl.constexpr,RES:tl.constexpr,B:tl.constexpr):
 e=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);ok=j<R;idx=e*R+j
 ox=tl.load(O+idx*3,ok,0);oy=tl.load(O+idx*3+1,ok,0);oz=tl.load(O+idx*3+2,ok,0)
 dx=tl.load(D+idx*3,ok,0);dy=tl.load(D+idx*3+1,ok,0);dz=tl.load(D+idx*3+2,ok,0)
 dist=tl.load(T+idx,ok,float('inf'));near=tl.load(NEAR+idx,ok,0);far=tl.load(FAR+idx,ok,1.)
 stop=tl.minimum(dist,far);valid=ok&tl.load(V+idx,ok,False)&(stop>=near)
 px=tl.abs(dx)<1e-12;py=tl.abs(dy)<1e-12
 ax=(X0-ox)/tl.where(px,1.,dx);bx=(X0+NX*RES-ox)/tl.where(px,1.,dx)
 ay=(Y0-oy)/tl.where(py,1.,dy);by=(Y0+NY*RES-oy)/tl.where(py,1.,dy)
 begin=tl.maximum(near,tl.maximum(tl.where(px,-float('inf'),tl.minimum(ax,bx)),tl.where(py,-float('inf'),tl.minimum(ay,by))))
 end=tl.minimum(stop,tl.minimum(tl.where(px,float('inf'),tl.maximum(ax,bx)),tl.where(py,float('inf'),tl.maximum(ay,by))))
 live=valid&~((px&((ox<X0)|(ox>=X0+NX*RES)))|(py&((oy<Y0)|(oy>=Y0+NY*RES))))&(end>begin+1e-7)
 x=tl.floor((ox+dx*(begin+1e-6)-X0)/RES).to(tl.int32);y=tl.floor((oy+dy*(begin+1e-6)-Y0)/RES).to(tl.int32)
 sx=tl.where(dx>0,1,tl.where(dx<0,-1,0));sy=tl.where(dy>0,1,tl.where(dy<0,-1,0))
 tx=tl.where(px,float('inf'),(X0+(x+tl.cast(sx>0,tl.int32))*RES-ox)/tl.where(px,1.,dx))
 ty=tl.where(py,float('inf'),(Y0+(y+tl.cast(sy>0,tl.int32))*RES-oy)/tl.where(py,1.,dy))
 stepx=tl.where(px,float('inf'),RES/tl.abs(dx));stepy=tl.where(py,float('inf'),RES/tl.abs(dy))
 k=0
 while (k<NX+NY+2)&(tl.sum(live.to(tl.int32),0)>0):
  inside=live&(x>=0)&(x<NX)&(y>=0)&(y<NY)
  tl.atomic_or(BITS+e*NX*NY+y*NX+x,1,inside,sem='relaxed')
  active=inside&(tl.minimum(tx,ty)<end-1e-7);crossx=tx<=ty+1e-7;crossy=ty<=tx+1e-7
  x+=tl.cast(active&crossx,tl.int32)*sx;y+=tl.cast(active&crossy,tl.int32)*sy
  tx=tl.where(active&crossx,tx+stepx,tx);ty=tl.where(active&crossy,ty+stepy,ty);live=active
  k+=1
 finite=dist<float('inf');td=tl.where(finite,dist,0.)
 qx=(ox+dx*td-X0)/RES;qy=(oy+dy*td-Y0)/RES
 qx=tl.where(tl.abs(qx-tl.floor(qx+.5))<2e-5,tl.floor(qx+.5),qx)
 qy=tl.where(tl.abs(qy-tl.floor(qy+.5))<2e-5,tl.floor(qy+.5),qy)
 x=tl.floor(qx).to(tl.int32);y=tl.floor(qy).to(tl.int32)
 hit=valid&finite&(x>=0)&(x<NX)&(y>=0)&(y<NY)
 height=tl.load(H+e*NX*NY+y*NX+x,hit,float('nan'));z=oz+dz*td-height
 sh=tl.load(S+idx,ok,False);known=height==height
 occ=hit&~sh&known&(z>=.160001)&(z<=.660001)
 amb=hit&(~known|sh)
 tl.atomic_or(BITS+e*NX*NY+y*NX+x,tl.where(occ,2,tl.where(amb,4,0)),hit&(occ|amb),sem='relaxed')


def raster183(ret,support,cfg,*,capture_ms,epoch):
 from .oracle_occgrid176 import OccFrame176,Support176
 r=ret.rays;n,count=ret.distance_m.shape
 bits=torch.zeros_like(support.reference_z,dtype=torch.int32)
 _dda[(n,triton.cdiv(count,128))](r.origins.contiguous(),r.directions.contiguous(),ret.distance_m.contiguous(),r.near_m.contiguous(),r.far_m.contiguous(),(ret.valid&r.coverage_valid).contiguous(),ret.self_hit.contiguous(),support.reference_z.contiguous(),bits,count,cfg.nx,cfg.ny,cfg.x_min_m,cfg.y_min_m,cfg.cell_m,128,enable_fp_fusion=False)
 st=torch.where((bits&2)!=0,2,torch.where(((bits&1)!=0)&((bits&4)==0),1,0)).to(torch.uint8)
 return OccFrame176(st,torch.where(st!=0,capture_ms[:,None,None],-1),Support176(support.reference_z.clone(),support.observed.clone(),support.seen_ms.clone()),epoch.clone(),capture_ms.clone())
