"""UPDATE165 opt-in certificates. Never writes policy maps or collision evidence."""
import math
import torch
import torch.nn.functional as F
from . import schema as S
from .warp import warp_categorical
from .support import encode_support


def _tetrominoes():
    shapes={((0,0),)}
    for _ in range(3):
        grown=set()
        for shape in shapes:
            cells=set(shape)
            for y,x in cells:
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    if (y+dy,x+dx) in cells:continue
                    new=cells|{(y+dy,x+dx)};y0=min(y for y,x in new);x0=min(x for y,x in new)
                    grown.add(tuple(sorted((y-y0,x-x0) for y,x in new)))
        shapes=grown
    return sorted(shapes)


def unknown_risk(known, required, *, resolution, relaxed=False, fraction_limit=.1, cluster_area=.01):
    missing=required&~known
    count=missing.flatten(1).sum(-1);denom=required.flatten(1).sum(-1).clamp_min(1)
    fraction=count/denom
    if not relaxed:return count>0,fraction
    cells=math.ceil(cluster_area/(resolution*resolution)-1e-8)
    if cells==1:cluster=count>0
    elif cells==4:
        # Every 4-connected component >=4 contains a connected tetromino.
        # Also rejects a 1x4 strip; checking only full 2x2 squares is weaker.
        kernels=known.new_zeros((19,1,4,4),dtype=torch.float32)
        for i,shape in enumerate(_tetrominoes()):
            for y,x in shape:kernels[i,0,y,x]=1
        cluster=(F.conv2d(missing[:,None].float(),kernels,padding=3)>=3.999).flatten(1).any(-1)
    else:raise ValueError('UPDATE165 supports 5cm or 10cm grids only')
    return cluster|(fraction>fraction_limit),fraction


class CertificateHistory:
    """Complete static per-cell certificates; no partial sub-bin fusion.

    Each source cell must already have all required bands + support observed.
    Area-overlap warp requires ALL overlapping cells certified. Support must
    agree in world height, and exact source age expires after ttl_s.
    """
    def __init__(self,grid,ttl_s=2.):
        self.grid=grid;self.ttl_s=ttl_s;self.frames=[]

    def reset(self,ids):
        for frame in self.frames:frame['known'][ids]=False

    def update(self,known,support_height,pose,now,static_eligible,*,fresh=None):
        self.frames=[f for f in self.frames if now-f['time']<=self.ttl_s+1e-8]
        result=known.clone()
        if self.frames:
            n=len(known);frames=self.frames
            state=torch.cat([f['known'][:,None].to(torch.uint8) for f in frames])
            support=torch.cat([encode_support(f['height']) for f in frames])
            world=torch.cat([f['height']+f['pose'][:,2,None,None] for f in frames])
            poses=torch.cat([f['pose'] for f in frames]);current=pose.repeat(len(frames),1)
            warped,_,height=warp_categorical(state,torch.zeros_like(state,dtype=torch.float32),poses,current,
                grid=self.grid,support=support,support_world_z=world,return_support_height=True)
            current_height=support_height.repeat(len(frames),1,1)
            consistent=~torch.isfinite(current_height)|((height-current_height).abs()<=.02)
            old=((warped[:,0]==S.FREE)&torch.isfinite(height)&consistent).reshape(len(frames),n,*known.shape[1:]).any(0)
            result|=old&static_eligible[:,None,None]
        # Dynamic scenes cannot cache current observations as static evidence.
        self.frames.append(dict(known=((known if fresh is None else fresh)&static_eligible[:,None,None]).clone(),height=support_height.clone(),pose=pose.clone(),time=now))
        return result
