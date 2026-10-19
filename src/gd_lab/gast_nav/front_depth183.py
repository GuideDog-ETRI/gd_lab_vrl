"""Optical-centre first-hit contract for supervisor-only UPDATE183 cameras.

No mapper/actor observations are modified. Lower camera IDs 0/1 are the only
permitted lower sources; rear cameras cannot be relabelled as head evidence.
"""
from dataclasses import replace
import torch
from .oracle_occgrid176 import OracleRays176, OracleReturns176, OccFrame176, Support176


def optical_returns183(origins, directions, distance, *, pixel_length, near_z,
                       far_z, measured, self_hit, source):
    """All distances start at the optical centre, including a near-plane blocker.

    An explicitly valid infinite oracle distance is no-hit; an invalid/near/self
    return never becomes a no-hit. Pixel length converts optical clips to metric.
    Sources 5/6 distinguish front D430 camera0/1; 4 is head D455.
    """
    if source not in (4,5,6):raise ValueError('head D455 or front D430 0/1 only')
    shape=distance.shape
    if len(shape)!=2 or origins.shape!=(*shape,3) or directions.shape!=origins.shape:raise ValueError('camera ray shape')
    if measured.shape!=shape or self_hit.shape!=shape or measured.dtype!=torch.bool or self_hit.dtype!=torch.bool:raise ValueError('explicit measurement metadata')
    if not 0<near_z<far_z:raise ValueError('optical clipping planes')
    length=torch.as_tensor(pixel_length,device=distance.device,dtype=distance.dtype).expand(shape)
    if not bool(torch.isfinite(length).all()&(length>=1-1e-6).all()):raise ValueError('pinhole pixel length')
    if not bool(torch.isfinite(origins).all()&torch.isfinite(directions).all()):raise ValueError('finite rays')
    if not bool((directions.norm(dim=-1)-1).abs().le(1e-5).all()):raise ValueError('unit metric ray')
    near=length*near_z;far=length*far_z
    valid=measured&~torch.isnan(distance)&(distance>=near)&~self_hit
    clipped=torch.where(distance>far,torch.inf,distance)
    # Malformed measurements stay invalid without feeding NaN into a raster.
    clipped=torch.where(torch.isnan(clipped)|(clipped<0),torch.inf,clipped)
    rays=OracleRays176(origins,directions,near,far,measured.clone(),torch.full(shape,source,device=distance.device,dtype=torch.long))
    return OracleReturns176(rays,clipped,valid,self_hit&torch.isfinite(clipped))


def merge_camera_frame183(head,camera):
    """Same pose/time only: occupied always wins, including head vs lower free.

    Support disagreement fails closed instead of raising the floor reference to
    the obstacle hit. Independent support observations keep original timestamps.
    """
    if not torch.equal(head.epoch,camera.epoch) or not torch.equal(head.capture_ms,camera.capture_ms):raise ValueError('same capture required')
    a,b=head.support,camera.support
    ah=a.observed&torch.isfinite(a.reference_z);bh=b.observed&torch.isfinite(b.reference_z)
    conflict=ah&bh&((a.reference_z-b.reference_z).abs()>.020001)
    take=~ah&bh
    height=torch.where(take,b.reference_z,a.reference_z)
    seen=torch.where(take,b.seen_ms,a.seen_ms)
    valid=(ah|bh)&~conflict
    state=torch.maximum(head.state,camera.state)
    state=torch.where(conflict&(state!=2),0,state).to(torch.uint8)
    # Occupied keeps the reference with which that return was classified.
    takeocc=(camera.state==2)&(head.state!=2)
    height=torch.where(takeocc,b.reference_z,height)
    seen=torch.where(takeocc,b.seen_ms,seen)
    valid=torch.where(takeocc,bh,valid)
    take_state=(camera.state>head.state)|((camera.state==head.state)&(camera.seen_ms>head.seen_ms)&(head.state!=2))
    state_seen=torch.where(take_state,camera.seen_ms,head.seen_ms)
    return OccFrame176(state,torch.where(state!=0,state_seen,-1),
        Support176(height,valid,seen),head.epoch.clone(),head.capture_ms.clone())


def virtual_entry183(pose,axis_yaw,speed,now_ms,*,distance_m=2.,spacing_m=.2):
    """Oldest first; caller MUST skip collision poses and perform sensor raycasts.

    The time origin is arbitrary: caller supplies sufficient nonnegative clock
    offset, also used for all live evidence. This is NOT measured walking history.
    """
    if pose.ndim!=2 or pose.shape[1]!=4:raise ValueError('pose')
    if speed.shape!=pose.shape[:1] or not bool(torch.isfinite(speed).all()&(speed>0).all()):raise ValueError('actual configured approach speed')
    if axis_yaw.shape!=speed.shape or now_ms.shape!=speed.shape:raise ValueError('entry metadata')
    count=round(distance_m/spacing_m)
    if spacing_m<=0 or count<1 or abs(count*spacing_m-distance_m)>1e-6:raise ValueError('entry spacing')
    direction=torch.stack((axis_yaw.cos(),axis_yaw.sin()),-1)
    for i in range(count,0,-1):
        distance=i*spacing_m;p=pose.clone();p[:,:2]-=direction*distance
        t=now_ms-(distance/speed*1000).round().long()
        if bool((t<0).any()):raise ValueError('insufficient time origin offset')
        yield p,t


def packet_returns183(packet,pose,*,device=None):
    """Read raw UPDATE177 front tap; never mutate the mapper's packet.

    Source resolution is recorded in the packet and must match its full array.
    No pixel stride is applied. Frame pose is actual world optical-centre pose.
    """
    from .obs_terms import world_to_sample
    if tuple(packet['camera_indices'])!=(0,1):raise ValueError('front camera0/1 only')
    device=pose.device if device is None else device
    def tensor(key,dtype=torch.float32):return torch.as_tensor(packet[key],device=device,dtype=dtype)
    direction=tensor('dirs');distance=tensor('ranges');length=tensor('pixel_length')
    n,c,h,w,_=direction.shape
    if c!=2 or (h,w)!=tuple(packet['resolution']) or distance.shape!=(n,c,h,w):raise ValueError('raw full-resolution packet')
    origin=tensor('origins')[:,:,None,None].expand_as(direction)
    measured=tensor('measured',torch.bool);self_hit=tensor('self_hit',torch.bool)
    for camera in range(2):
        o=world_to_sample(origin[:,camera].reshape(n,-1,3),pose)
        d=world_to_sample(direction[:,camera].reshape(n,-1,3),pose,vectors=True)
        yield optical_returns183(o,d,distance[:,camera].reshape(n,-1),
            pixel_length=length[:,camera].reshape(n,-1),near_z=packet['near_z'],far_z=packet['far_z'],
            measured=measured[:,camera].reshape(n,-1),self_hit=self_hit[:,camera].reshape(n,-1),source=5+camera)


def supplement_memory183(current,memory,*,obstacle_ttl_s,support_ttl_s):
    """Keep capture-local evidence; world-cache round trips may only fill holes.

    A still-live occupied memory keeps its priority over current free. Memory
    free cannot overwrite a conflicting measured floor. All ages stay original.
    """
    cf=torch.isfinite(current.age_s)&(current.age_s>=0)&(current.age_s<=obstacle_ttl_s)
    mf=torch.isfinite(memory.age_s)&(memory.age_s>=0)&(memory.age_s<=obstacle_ttl_s)&(current.epoch==memory.epoch)[:,None,None]
    cs=torch.isfinite(current.support)&torch.isfinite(current.support_age_s)&(current.support_age_s>=0)&(current.support_age_s<=support_ttl_s)
    ms=torch.isfinite(memory.support)&torch.isfinite(memory.support_age_s)&(memory.support_age_s>=0)&(memory.support_age_s<=support_ttl_s)&(current.epoch==memory.epoch)[:,None,None]
    conflict=cs&ms&((current.support-memory.support).abs()>.020001)
    cur=torch.where(cf,current.state,0).to(torch.uint8)
    mem=torch.where(mf&~((memory.state==1)&conflict),memory.state,0).to(torch.uint8)
    take=((cur==0)&(mem!=0))|((mem==2)&(cur!=2))
    state=torch.where(take,mem,cur);age=torch.where(take,memory.age_s,current.age_s)
    height=torch.where(cs,current.support,torch.where(ms,memory.support,torch.nan))
    sa=torch.where(cs,current.support_age_s,memory.support_age_s)
    return replace(current,state=state,age_s=age,support=height,support_age_s=sa)
