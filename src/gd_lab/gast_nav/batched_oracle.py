"""GPU-resident critic/reward geometry. Never used for actor obstacle maps.

Float32 route/OBB arithmetic has absolute tolerance 1e-5m vs scalar float64
oracle away from decision thresholds; threshold-adjacent cases are tested.
"""
import torch
from . import schema as S

class BatchedOracle:
    def __init__(self,tiles,settings,*,device):
        self.settings=settings;self.device=device
        n=len(tiles);p=max(len(t.route_polyline)-1 for t in tiles);w=max(len(t.width_profile) for t in tiles);b=max(1,max(len(t.obstacles) for t in tiles));q=max(len(t.segments) for t in tiles)
        # Build on CPU once, transfer each complete table once.
        a=torch.zeros(n,p,2);v=a.clone();route_valid=torch.zeros(n,p,dtype=torch.bool)
        profiles=torch.zeros(n,w,2);profile_valid=torch.zeros(n,w,dtype=torch.bool)
        boxes=torch.zeros(n,b,7);box_valid=torch.zeros(n,b,dtype=torch.bool);step=box_valid.clone()
        segments=torch.zeros(n,q,3);segment_valid=torch.zeros(n,q,dtype=torch.bool)
        for i,t in enumerate(tiles):
            for j,(x,y) in enumerate(zip(t.route_polyline,t.route_polyline[1:])):a[i,j]=torch.tensor(x);v[i,j]=torch.tensor(y)-a[i,j];route_valid[i,j]=True
            profiles[i,:len(t.width_profile)]=torch.tensor(t.width_profile);profile_valid[i,:len(t.width_profile)]=True
            for j,o in enumerate(t.obstacles):boxes[i,j]=torch.tensor((*o.center,*o.half,o.yaw));box_valid[i,j]=True;step[i,j]=o.kind=='step'
            for j,(kind,start,end) in enumerate(t.segments):segments[i,j]=torch.tensor((S.SEGMENT_TYPES.index(kind),start,end));segment_valid[i,j]=True
        for key,value in dict(a=a,v=v,route_valid=route_valid,profiles=profiles,profile_valid=profile_valid,boxes=boxes,box_valid=box_valid,step=step,segments=segments,segment_valid=segment_valid,
            legacy=torch.tensor([t.kind=='legacy' for t in tiles]),stairs=torch.tensor([t.floor_z_at=='stairs' for t in tiles]),turn=torch.tensor([t.turn_spot is not None for t in tiles])).items():setattr(self,key,value.to(device))
        self.length=self.v.norm(dim=-1);self.prefix=self.length.cumsum(-1)-self.length
        self.cos=self.boxes[...,6].cos();self.sin=self.boxes[...,6].sin()
    def evaluate(self,ids,xy,turn_mask,margin,*,swept_width,turn_clear_radius,sdf_clip):
        a=self.a[ids];v=self.v[ids];length=self.length[ids];delta=xy[:,None]-a
        unit=v/length.clamp_min(1e-12)[...,None]
        along=(delta*unit).sum(-1).clamp_min(0).minimum(length)
        distance=(xy[:,None]-(a+along[...,None]*unit)).square().sum(-1)
        nearest=distance.masked_fill(~self.route_valid[ids],torch.inf).argmin(-1)
        pick=lambda x:x.gather(1,nearest[:,None]).squeeze(1)
        s=pick(self.prefix[ids]+along);offset=pick(-delta[...,0]*unit[...,1]+delta[...,1]*unit[...,0]);axis=pick(torch.atan2(v[...,1],v[...,0]))
        profile=self.profiles[ids];valid=self.profile_valid[ids]
        future=torch.where(valid&(profile[...,0]>=s[:,None]),profile[...,1],torch.inf).amin(-1)
        indices=torch.arange(profile.shape[1],device=xy.device)[None].expand(len(xy),-1)
        prev=torch.where(valid&(profile[...,0]<=s[:,None]),indices,-1).amax(-1)
        previous=profile[...,1].gather(1,prev.clamp_min(0)[:,None]).squeeze(1)
        available=torch.minimum(future,torch.where(prev>=0,previous,torch.inf))
        box=self.boxes[ids];co=self.cos[ids];si=self.sin[ids];bd=xy[:,None]-box[...,:2]
        localx=co*bd[...,0]+si*bd[...,1];localy=-si*bd[...,0]+co*bd[...,1]
        floor=(self.step[ids]&self.box_valid[ids]&(localx.abs()<=box[...,3])&(localy.abs()<=box[...,4])&self.stairs[ids,None])
        z=torch.where(floor,box[...,2]+box[...,5],0.).amax(-1).clamp_min(0)
        dx=localx.abs()-box[...,3];dy=localy.abs()-box[...,4]
        distance=torch.hypot(dx.clamp_min(0),dy.clamp_min(0))+torch.maximum(dx,dy).clamp_max(0)
        solid=self.box_valid[ids]&~self.step[ids]
        cfg=self.settings['oracle']
        active=solid&(box[...,2]+box[...,5]>z[:,None]+cfg['epsilon_m'])&(box[...,2]-box[...,5]<z[:,None]+cfg['collision_height_m'])
        rotation=distance.masked_fill(~active,torch.inf).amin(-1).clamp_min(0)
        feasible=torch.where(turn_mask,self.turn[ids]&(rotation>=cfg['turn_swept_radius_m']+margin),available>=cfg['swept_width_m']+2*margin)
        feasible |= self.legacy[ids]
        active_clear=solid&(box[...,2]+box[...,5]>z[:,None])
        radius=torch.where(turn_mask,turn_clear_radius,swept_width/2)
        clearance=(distance-radius[:,None]).masked_fill(~active_clear,torch.inf).amin(-1).clamp_max(sdf_clip)
        clearance=torch.where(self.legacy[ids],sdf_clip,clearance)
        segments=self.segments[ids];inside=self.segment_valid[ids]&(segments[...,1]<=s[:,None])&(segments[...,2]>=s[:,None])
        ix=inside.int().argmax(-1);ix=torch.where(inside.any(-1),ix,self.segment_valid[ids].sum(-1)-1)
        segment=segments[...,0].gather(1,ix[:,None]).squeeze(1).long()
        return dict(s=s,offset=offset,axis=axis,feasible=feasible,clearance=clearance,segment=segment,available=available,rotation=rotation)
