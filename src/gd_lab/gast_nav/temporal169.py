"""Bounded temporal sensor evidence, explicitly opt-in for static training C.

Path integrates an operating speed upper bound at every physics interval and
also bounds observed displacement. It never uses net capture displacement as
travel. Bounds are diagnostic assumptions; sample checks cannot prove a
continuous-time physical bound. No changes to the default evidence settings.
"""
from dataclasses import replace
import torch
from .free_evidence167 import FreeEvidenceConfig, FreeEvidenceHistory, _rotation
from .map_types import ResolvedRays
from .warp import warp_categorical
from .support import encode_support
from . import schema as S

class TemporalEvidence169:
    def __init__(self, n, device, cfg):
        if cfg['context'] != 'training_static_c' or cfg['historical_obstacle_speed_bound_mps'] != 0:
            raise ValueError('static C opt-in required')
        self.linear=float(cfg['linear_speed_bound_mps']);self.angular=float(cfg['yaw_rate_bound_radps'])
        if self.linear<=0 or self.angular<=0:raise ValueError('positive operating bounds required')
        self.settings=FreeEvidenceConfig(max_captures=int(cfg['max_captures']),historical_obstacle_speed_bound_mps=0.)
        self.histories=[FreeEvidenceHistory(self.settings),FreeEvidenceHistory(self.settings)]
        self.distance=torch.zeros(n,device=device,dtype=torch.float64)
        self.previous=None;self.anchors=[]
        self.invalid=torch.zeros(n,device=device,dtype=torch.bool)
        self.violations=torch.zeros(n,device=device,dtype=torch.long)

    def reset(self, ids):
        for h in self.histories:h.reset(ids)
        self.invalid[ids]=False
        if self.previous is not None:self.previous[ids]=torch.nan

    def observe(self, pose, speed, yaw_rate, dt):
        if dt<0:raise ValueError('negative physics interval')
        bad=(~torch.isfinite(pose).all(-1))|(~torch.isfinite(speed))|(~torch.isfinite(yaw_rate))|(speed>self.linear)|(yaw_rate.abs()>self.angular)
        if self.previous is not None:
            old_ok=torch.isfinite(self.previous).all(-1)
            delta=(pose[:,:3]-self.previous[:,:3]).norm(dim=-1)
            angle=torch.atan2(torch.sin(pose[:,3]-self.previous[:,3]),torch.cos(pose[:,3]-self.previous[:,3])).abs()
            bad |= old_ok & ((delta>self.linear*dt+1e-5)|(angle>self.angular*dt+1e-5))
            increment=torch.maximum(torch.nan_to_num(delta,nan=0.,posinf=0.),torch.full_like(delta,self.linear*dt))
            self.distance += torch.where(old_ok,increment,torch.zeros_like(increment)).double()
        self.previous=pose.clone()
        self.invalid |= bad;self.violations += bad.long()
        # Any interval violation removes its entire previous evidence chain.
        for h in self.histories:
            for i,c in enumerate(h.captures):
                h.captures[i]=replace(c,epoch=torch.where(bad,-1,c.epoch))

    def push(self, frusta, pose, supports, time_ms, epoch, *, dynamic_enabled):
        if bool(dynamic_enabled.any()):raise ValueError('static-C history disallows enabled moving cylinders')
        valid=frusta.volume_valid & ~self.invalid[:,None]
        frusta=replace(frusta,volume_valid=valid)
        for h,s in zip(self.histories,supports):h.push(frusta,pose,s,time_ms,epoch)
        self.anchors=[self.distance.clone()]+self.anchors[:self.settings.max_captures-1]
        # A new sample can recover only after a full good interval, never by
        # resurrecting invalidated previous captures.
        self.invalid.zero_()

    def measurement(self,index,pose,now_ms,epoch):
        travel=torch.stack([(self.distance-a).clamp_min(0) for a in self.anchors],-1).to(pose.dtype)
        # Floating-point guard only; actual travel is the integrated bound.
        for k,c in enumerate(self.histories[index].captures):
            travel[:,k]=torch.maximum(travel[:,k],(pose[:,:2]-c.pose[:,:2]).norm(dim=-1))
        return self.histories[index].measurement(pose,now_ms,epoch,travel,
            pose.new_full((len(pose),),self.linear),pose.new_full((len(pose),),self.angular))

def current_support169(cap, pose, now_ms, epoch, grid, config):
    state=torch.where(torch.isfinite(cap.support)[:,None],S.FREE,S.UNKNOWN).to(torch.uint8)
    elapsed=((now_ms-cap.time_ms).double()/1000).clamp_min(0).to(cap.support.dtype)
    ages=elapsed[:,None,None,None].expand_as(state)
    _,_,height=warp_categorical(state,ages,cap.pose,pose,grid=grid,config=config,
        support=encode_support(cap.support,config=config),support_world_z=cap.support+cap.pose[:,2,None,None],return_support_height=True)
    eligible=(cap.epoch==epoch)&(elapsed<=config.ttl_s)
    return torch.where(eligible[:,None,None],height,torch.nan)

def transform_returns169(rays, old_pose, pose, valid):
    """Old real first returns remain conservative blockers, never free evidence."""
    rot=_rotation(old_pose[:,3]-pose[:,3]);shift=old_pose[:,:3]-pose[:,:3]
    shift=torch.einsum('nij,nj->ni',_rotation(-pose[:,3]),shift)
    origins=torch.einsum('nij,nrj->nri',rot,rays.origins)+shift[:,None]
    directions=torch.einsum('nij,nrj->nri',rot,rays.directions)
    return ResolvedRays(origins,directions,rays.distances,rays.occupied_hit&valid[:,None],rays.self_hit&valid[:,None])
