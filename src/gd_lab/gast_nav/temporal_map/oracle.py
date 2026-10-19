"""Native-pattern first-hit adapter; no GT occupancy/support lookup is exposed.

Ray streams use full live base rotation, then project into the capture's yaw
chart (the UPDATE176 grid chart). C2 measured support fitting stays with caller.
"""
from dataclasses import replace
import torch
from ..oracle_occgrid176 import (OracleRays176,OracleReturns176,oracle_first_hits176,
    rasterize176,OccFrame176,Support176,UNKNOWN,FREE,OCCUPIED)
from .sensors import HeadSensors


def yaw_rotation(yaw):
    c=yaw.cos();s=yaw.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
    return torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(-1,3,3)


class HeadOracle175B:
    def __init__(self,settings,*,device='cpu'):
        self.sensors=HeadSensors(settings,device=device)

    def returns(self,scene,base_rotation,pose,*,capture_id,scene_capture_id,self_mask,
                capture_index,screen_door=False,sensors=('ouster','mid360','d455')):
        """scene primitives are in current sample yaw chart, SAME capture_id.

        pose [N,4] is world x,y,z,yaw; rotation [N,3,3] maps base to world.
        scene must include floor, all obstacles AND all existing self geometry;
        complete=False makes every ray invalid. No synthetic support plane here.
        """
        n=pose.shape[0];yaw=yaw_rotation(pose[:,3].to(base_rotation.dtype))
        if base_rotation.shape!=(n,3,3) or not bool(torch.isfinite(base_rotation).all()):raise ValueError('live base rotation')
        eye=torch.eye(3,device=base_rotation.device,dtype=base_rotation.dtype)
        if not bool(torch.allclose(base_rotation.transpose(1,2)@base_rotation,eye[None].expand(n,-1,-1),atol=1e-5)) or bool((torch.linalg.det(base_rotation)<.999).any()):raise ValueError('proper live base rotation')
        rot=yaw.transpose(1,2)@base_rotation
        for sensor in sensors:
            source={'ouster':0,'mid360':1,'d455':4}[sensor]
            def callback(o,d,far):
                oo=torch.einsum('nij,nbj->nbi',rot,o);dd=torch.einsum('nij,nbj->nbi',rot,d)
                rays=OracleRays176(oo,dd,oo.new_zeros(oo.shape[:2]),oo.new_full(oo.shape[:2],far),torch.ones(oo.shape[:2],device=oo.device,dtype=torch.bool),torch.full(oo.shape[:2],source,device=oo.device,dtype=torch.long))
                ret=oracle_first_hits176(scene,rays,scene_capture_id=scene_capture_id,ray_capture_id=capture_id,self_mask=self_mask)
                return ret.distance_m,ret.self_hit,ret.valid
            for sample in self.sensors.capture(sensor,callback,n=n,capture_index=capture_index,screen_door=False):
                o=torch.einsum('nij,nbj->nbi',rot,sample.origins);d=torch.einsum('nij,nbj->nbi',rot,sample.directions)
                near=o.new_full(o.shape[:2],sample.near_m);far=o.new_full(o.shape[:2],sample.far_m)
                valid=sample.valid.clone()
                if screen_door and sensor=='mid360':
                    # Same UPDATE176 body-yaw x>0 clipping contract, not a
                    # sensor-local direction crop. No free behind x=0.
                    dx=d[...,0];limit=torch.where(dx<0,o[...,0]/(-dx).clamp_min(1e-12),torch.inf)
                    far=torch.minimum(far,limit);valid &= (o[...,0]>=0)&(far>near)
                interval=far>near;coverage=interval.clone()
                far=torch.where(interval,far,near+1.)
                rays=OracleRays176(o,d,near,far,coverage,torch.full(o.shape[:2],source,device=o.device,dtype=torch.long))
                yield OracleReturns176(rays,sample.distances,valid&interval,sample.self_hit),sample.source_id


def measured_shadow175b(returns,support,cfg,*,capture_ms,epoch):
    """XY shadow of a CURRENT actual first hit; invalidation only, NEVER evidence.

    Includes self/near first hits even if measurement is invalid for free. Invalid
    inventory must not call this with fabricated hit distances. Conservative XY
    shadow includes above-height hits too, because they terminate the same rays.
    """
    r=returns.rays
    hit=torch.isfinite(returns.distance_m)&(returns.distance_m>=0)&(returns.distance_m<r.far_m-1e-5)
    # Include the endpoint cell: do not leave historical free in a self endpoint.
    shadow_r=replace(r,near_m=torch.where(hit,returns.distance_m.clamp_min(0),r.near_m),coverage_valid=r.coverage_valid&hit)
    shadow_ret=OracleReturns176(shadow_r,torch.full_like(returns.distance_m,torch.inf),hit,torch.zeros_like(hit))
    return rasterize176(shadow_ret,support,cfg,capture_ms=capture_ms,epoch=epoch).state==FREE


def rasterize_head175b(stream,support,cfg,*,capture_ms,epoch):
    """Consume bounded head-ray chunks, occupied wins; source OR only winners.

    Returns (OccFrame176, uint8 source_bits[N,Y,X], bool shadow_mask[N,Y,X]).
    Support MUST be supplied from observed_support176/C2 hit fitting. The stream
    can be recast twice (first fits, then occupancy); it is not a GT floor prior.
    """
    shape=support.reference_z.shape;dev=support.reference_z.device
    state=torch.zeros(shape,device=dev,dtype=torch.uint8);bits=torch.zeros_like(state)
    shadow=torch.zeros(shape,device=dev,dtype=torch.bool)
    for returns,source_bit in stream:
        f=rasterize176(returns,support,cfg,capture_ms=capture_ms,epoch=epoch)
        winner=(f.state>state);same=(f.state==state)&(state!=UNKNOWN)
        bits=torch.where(winner,source_bit,torch.where(same,bits|source_bit,bits)).to(torch.uint8)
        state=torch.maximum(state,f.state)
        shadow |=measured_shadow175b(returns,support,cfg,capture_ms=capture_ms,epoch=epoch)
    return OccFrame176(state,torch.where(state!=UNKNOWN,capture_ms[:,None,None],-1),Support176(support.reference_z.clone(),support.observed.clone(),support.seen_ms.clone()),epoch.clone(),capture_ms.clone()),bits,shadow
