"""Analysis-only inner union proof. Never returns positive from samples alone.

Independent local angular/range residuals stay per block. A camera-common
budget may be applied after its blocks' union; root error after camera union.
In this audit the unspecified 0.5deg remains entirely local (camera-common=0).
Common rigid root error is reduced to a conservative axis-aligned outer box.
Every accepted box is a finite partition whose leaves are wholly in one local
robust frustum. Unresolved leaves are false. No hull, hole filling or ratio gate.
"""
import math,itertools
import torch

class UnionProof:
 def __init__(self,origins,normals,offsets,valid,lever,*,angle_rad=0.,support_error=0.,root_xy=0.,root_yaw=0.,epsilon=1e-6,max_depth=9,chunk=256):
  self.o=origins;self.n=normals;self.off=offsets;self.valid=valid;self.lever=lever
  self.support_error=support_error;self.root_xy=root_xy;self.root_yaw=2*math.sin(min(root_yaw,math.pi)/2);self.angle=2*math.sin(min(angle_rad,math.pi)/2);self.eps=epsilon;self.depth=max_depth;self.chunk=chunk
  self.sign=origins.new_tensor(list(itertools.product((-1.,1.),repeat=3)))
  self.stats=dict(nodes=0,splits=0,depth_reject=0,point_reject=0,whole_box_accept=0)
 def contained(self,c,h):
  q=c[:,None]-self.o[None];radius=q.norm(dim=-1)+h.norm(dim=-1)[:,None]+self.lever[None]
  distance=torch.einsum('qrd,rkd->qrk',q,self.n)-torch.einsum('rkd,qd->qrk',self.n.abs(),h)
  error=self.angle*radius[:,:,None]+self.support_error*self.n[:,:,2].abs()[None]
  error=error+self.root_xy*self.n[:,:,:2].norm(dim=-1)[None]+self.root_yaw*radius[:,:,None]*self.n[:,:,:2].norm(dim=-1)[None]
  return (((distance>=self.off[None]+error+self.eps).all(-1))&self.valid[None]).any(-1)
 def points_inside(self,c):
  return self.contained(c,torch.zeros_like(c))
 def cover(self,c,h):
  h=torch.broadcast_to(h,c.shape).clone();result=torch.zeros(len(c),device=c.device,dtype=torch.bool)
  for first in range(0,len(c),self.chunk):
   cc=c[first:first+self.chunk];hh=h[first:first+self.chunk];size=len(cc)
   roots=torch.arange(size,device=c.device);ok=torch.ones(size,device=c.device,dtype=torch.bool)
   for level in range(self.depth+1):
    self.stats['nodes']+=len(cc)
    accepted=self.contained(cc,hh);self.stats['whole_box_accept']+=int(accepted.sum())
    pending=~accepted
    cc=cc[pending];hh=hh[pending];roots=roots[pending]
    if not len(cc):break
    if level==self.depth:
     ok[roots]=False;self.stats['depth_reject']+=len(cc);break
    # Point tests only REJECT; a positive proof always comes from full-volume
    # leaves. Include center to expose a thin interior slit missed by corners.
    pts=torch.cat((cc[:,None],cc[:,None]+hh[:,None]*self.sign[None]),1)
    hits=[]
    for p0 in range(0,pts.numel()//3,self.chunk):hits.append(self.points_inside(pts.reshape(-1,3)[p0:p0+self.chunk]))
    possible=torch.cat(hits).reshape(len(cc),9).all(-1)
    ok[roots[~possible]]=False;self.stats['point_reject']+=int((~possible).sum())
    keep=possible&ok[roots];cc=cc[keep];hh=hh[keep];roots=roots[keep]
    if not len(cc):break
    # Exact binary partition on longest axis, no enlargement/fill at seams.
    axis=hh.argmax(-1);rows=torch.arange(len(cc),device=c.device);delta=torch.zeros_like(cc);delta[rows,axis]=hh[rows,axis]/2
    hh=hh.clone();hh[rows,axis]/=2
    self.stats['splits']+=len(cc)
    cc=torch.cat((cc-delta,cc+delta));hh=torch.cat((hh,hh));roots=torch.cat((roots,roots))
   result[first:first+size]=ok
  return result

def root_outer_half(c,h,p,yaw,root_shift,lever_extra):
 """Inverse rigid-transform reachability outer box. Translation XY <=p;
 yaw about capture-root, with radius >= every B corner to root, includes p
 cross-term. 3D distance and extra sensor lever are conservative overestimates.
 """
 h=torch.broadcast_to(h,c.shape).clone()
 radius=(c-root_shift).norm(dim=-1)+h.norm(dim=-1)+float(lever_extra)+float(p)
 reach=float(p)+2*math.sin(min(float(yaw),math.pi)/2)*radius
 h[:,:2]+=reach[:,None]
 return h
