"""Fuse a read-only terrain tap into UPDATE176 AFTER sensor-memory query.

D-6 opt-in: observed walkable terrain supplies near-ROI FREE; head occupied wins.
Terrain only supplies observed support or vetoes (gap, low object, disagreement).
"""
from dataclasses import dataclass,replace
from pathlib import Path
import math
import torch,yaml
from .oracle_occgrid176 import OccConfig176,OccFrame176,Support176,UNKNOWN,FREE,OCCUPIED
from .support import grow_support
from .map_config import load_config as support_config
from .terrain_warp177 import resample

@dataclass(frozen=True)
class FusionConfig177:
    terrain_free180:bool=False
    support_ttl_s:float=.5
    obstacle_ttl_s:float=.5
    flat_raised_m:float=.03
    flat_drop_m:float=.03
    support_agreement_m:float=.02
    normal_max_slope_deg:float=35.
    normal_fit_rms_m:float=.02
    seed_near_x_m:float=.35
    seed_near_y_m:float=.28
    flat_support_tolerance_m:float=.10
    max_step_m:float=.25
    chunk_envs:int=128
    terrain_limits177b:bool=False
    gap_max_width_m:float=.26
    flat_step_max_m:float=.16
    stair_step_max_m:float=.25
    rough_max_m:float=.10
    low_object_max_m:float=.16
    def __post_init__(self):
        if any(not math.isfinite(v) or v<=0 for v in (self.gap_max_width_m,self.flat_step_max_m,self.stair_step_max_m,self.rough_max_m,self.low_object_max_m)):raise ValueError('terrain limits')
        if self.support_ttl_s!=.5:raise ValueError('foot support TTL remains .5s')
        if self.obstacle_ttl_s not in (.5,1.,2.,3.):raise ValueError('obstacle TTL comparison')
        if any(not math.isfinite(v) or v<=0 for v in (self.flat_raised_m,self.flat_drop_m,self.support_agreement_m,self.normal_fit_rms_m,self.seed_near_x_m,self.seed_near_y_m,self.flat_support_tolerance_m,self.max_step_m)):raise ValueError('positive config')
        if not 0<self.normal_max_slope_deg<=35 or self.chunk_envs<1:raise ValueError('config bounds')
    @classmethod
    def load(cls,path=None):
        p=Path(path) if path else Path(__file__).resolve().parents[3]/'configs/gast_nav/terrain_fusion177.yaml'
        return cls(**yaml.safe_load(p.read_text()))

@dataclass(frozen=True)
class TerrainFusion177:
    frame:OccFrame176
    foot_support:Support176
    cert_cfg:OccConfig176
    terrain_roi:torch.Tensor
    gap:torch.Tensor
    terrain_occupied:torch.Tensor
    support_conflict:torch.Tensor
    terrain_seen_ms:torch.Tensor
    terrain_source_bits:torch.Tensor # 1<<native source code; diagnostic only
    head_capture_ms:torch.Tensor
    terrain_capture_ms:torch.Tensor
    walkable_gap:torch.Tensor
    walkable_gap_seen_ms:torch.Tensor


def fit_normals177(height,valid,cfg):
    """Local observed-height plane fit; no GT normals or default flat normal.

    Discontinuous neighbouring treads are excluded by a 35deg + fit-error gate.
    At least three non-collinear samples and RMS<=.02m are required. Missing
    neighbours and isolated cells are not inferred to be horizontal support.
    """
    n,ny,nx=height.shape;fields=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            h=torch.roll(height,(-dy,-dx),(1,2));v=torch.roll(valid,(-dy,-dx),(1,2)).clone()
            if dy<0:v[:,:-dy]=False
            if dy>0:v[:,-dy:]=False
            if dx<0:v[:,:,:-dx]=False
            if dx>0:v[:,:,-dx:]=False
            limit=math.tan(math.radians(cfg.normal_max_slope_deg))*math.hypot(dx*.1,dy*.1)+cfg.normal_fit_rms_m
            v &= valid&torch.isfinite(h)&((h-height).abs()<=limit+1e-6)
            fields.append((dx*.1,dy*.1,torch.where(v,h,0.),v.float()))
    count=sum(v for _,_,_,v in fields);den=count.clamp_min(1)
    mx=sum(x*v for x,_,_,v in fields)/den;my=sum(y*v for _,y,_,v in fields)/den
    mz=sum(z*v for _,_,z,v in fields)/den
    xx=sum(v*(x-mx)**2 for x,_,_,v in fields);yy=sum(v*(y-my)**2 for _,y,_,v in fields)
    xy=sum(v*(x-mx)*(y-my) for x,y,_,v in fields)
    xz=sum(v*(x-mx)*(z-mz) for x,_,z,v in fields);yz=sum(v*(y-my)*(z-mz) for _,y,z,v in fields)
    det=xx*yy-xy*xy;safe=det.clamp_min(1e-12)
    a=(xz*yy-yz*xy)/safe;b=(yz*xx-xz*xy)/safe
    err=sum(v*(z-(mz+a*(x-mx)+b*(y-my)))**2 for x,y,z,v in fields)/den
    good=valid&(count>=3)&(det>1e-12)&(err<=cfg.normal_fit_rms_m**2+1e-8)
    normal=torch.stack((-a,-b,torch.ones_like(a)),-1);normal/=normal.norm(dim=-1,keepdim=True)
    return torch.where(good[...,None],normal,torch.nan)


def _metadata(frame,pose,now,epoch):
    n=frame.surface_z.shape[0];dev=frame.surface_z.device
    if frame.surface_z.shape!=(n,11,17) or frame.surface_z.dtype!=torch.float32:raise ValueError('terrain float32 [N,11,17]')
    for v in (frame.valid,frame.gap,frame.pending):
        if v.shape!=frame.surface_z.shape or v.dtype!=torch.bool or v.device!=dev:raise ValueError('terrain masks')
    if bool((~((frame.source==0)|(frame.source==1)|(frame.source==4)|(frame.source==5))).any()):raise ValueError('terrain native source')
    if frame.seen_ms.shape!=frame.surface_z.shape or frame.seen_ms.dtype!=torch.int64 or frame.source.shape!=frame.surface_z.shape or frame.source.dtype!=torch.uint8:raise ValueError('terrain provenance')
    for v in (now,epoch,frame.epoch,frame.sample_ms,frame.last_capture_ms):
        if v.shape!=(n,) or v.dtype!=torch.int64 or v.device!=dev:raise ValueError('time/epoch [N] int64')
    if pose.shape!=(n,4) or frame.pose.shape!=(n,4) or pose.device!=dev or frame.pose.device!=dev or not bool(torch.isfinite(pose).all()&torch.isfinite(frame.pose).all()):raise ValueError('world pose [N,x,y,z,yaw]')
    if bool((now<0).any()|(epoch<0).any()|(frame.sample_ms<0).any()|(frame.last_capture_ms>frame.sample_ms).any()|(frame.sample_ms>now).any()):raise ValueError('future/invalid time')
    if bool((frame.valid&(~torch.isfinite(frame.surface_z)|(frame.seen_ms<0)|(frame.seen_ms>frame.sample_ms[:,None,None]))).any()):raise ValueError('terrain observation provenance')
    return n


def fuse_terrain177(head,terrain,target_pose,now_ms,epoch,*,grid=None,config=None,
                    measured_current_support,segment_type=None,contact_seed_mask=None,head_gap_mask=None, gap_axis_enabled=None):
    """head = raw obstacle OccFrame176, BEFORE support-expiry turns free unknown.

    The terrain chart is the ACTUAL pose used by actor maps.sample, not latest
    pose assigned to old samples. measured_current_support [N] is measured in
    terrain.pose's base-z frame. Route hint is gdm segment_type, never GT family.
    Contact mask selects cells but cannot fill missing hits. Never mutate either
    input, native mapper, 3906 history or actor OBS1.
    """
    cfg=config or FusionConfig177();grid=grid or OccConfig176()
    pose=torch.as_tensor(target_pose,device=terrain.surface_z.device,dtype=torch.float64)
    n=_metadata(terrain,pose,now_ms,epoch);dev=pose.device;shape=(n,grid.ny,grid.nx)
    if head.state.shape!=shape or head.seen_ms.shape!=shape or head.state.dtype!=torch.uint8 or head.seen_ms.dtype!=torch.int64 or bool((head.state>2).any()):raise ValueError('head state/provenance')
    if head.epoch.shape!=(n,) or head.capture_ms.shape!=(n,) or head.epoch.dtype!=torch.int64 or head.capture_ms.dtype!=torch.int64:raise ValueError('head frame metadata')
    if bool((head.capture_ms>now_ms).any()):raise ValueError('future head capture')
    for v in (head.support.reference_z,head.support.observed,head.support.seen_ms):
        if v.shape!=shape or v.device!=dev:raise ValueError('head support grid')
    if head_gap_mask is not None and (head_gap_mask.shape!=shape or head_gap_mask.dtype!=torch.bool or head_gap_mask.device!=dev):raise ValueError('head gap mask must be current TTL/epoch-filtered bool grid')
    reference=torch.as_tensor(measured_current_support,device=dev,dtype=torch.float32)
    if reference.shape!=(n,) or not bool(torch.isfinite(reference).all()):raise ValueError('measured floor [N], not GT/default base height')
    if contact_seed_mask is not None and (contact_seed_mask.shape!=(n,11,17) or contact_seed_mask.dtype!=torch.bool):raise ValueError('contact seed mask')
    source_grid=OccConfig176(x_min_m=-.85,x_max_m=.85,y_min_m=-.55,y_max_m=.55)
    c2=replace(support_config(),flat_support_tolerance_m=cfg.flat_support_tolerance_m,max_step_m=cfg.max_step_m,max_slope_deg=cfg.normal_max_slope_deg,horizontal_max_slope_deg=cfg.normal_max_slope_deg)
    from .schema import SEGMENT_TYPES
    segment=SEGMENT_TYPES.index('FLAT') if segment_type is None else (SEGMENT_TYPES.index(segment_type) if isinstance(segment_type,str) else segment_type)
    segment=torch.as_tensor(segment,device=dev,dtype=torch.long).expand(n)
    stair=(segment==SEGMENT_TYPES.index('STAIR_UP'))|(segment==SEGMENT_TYPES.index('STAIR_DOWN'))|(segment==SEGMENT_TYPES.index('LANDING'))
    yy,xx=torch.meshgrid(torch.linspace(-.5,.5,11,device=dev),torch.linspace(-.8,.8,17,device=dev),indexing='ij')
    outputs=[]
    if gap_axis_enabled is None:gap_axis_enabled=torch.ones(n,device=dev,dtype=torch.bool)
    for start in range(0,n,cfg.chunk_envs):
        sl=slice(start,min(start+cfg.chunk_envs,n));h=terrain.surface_z[sl]
        age=now_ms[sl,None,None]-terrain.seen_ms[sl]
        same=(terrain.epoch[sl]==epoch[sl])[:,None,None]
        fresh=terrain.valid[sl]&(terrain.seen_ms[sl]>=0)&(age>=0)&(age<=500)&same&~terrain.gap[sl]&~terrain.pending[sl]
        raised=cfg.low_object_max_m if cfg.terrain_limits177b else cfg.flat_raised_m
        drop=cfg.flat_step_max_m if cfg.terrain_limits177b else cfg.flat_drop_m
        positive=(h>reference[sl,None,None]+raised+1e-6) if cfg.terrain_limits177b else (h>=reference[sl,None,None]+raised-1e-6)
        negative=(h<reference[sl,None,None]-drop-1e-6) if cfg.terrain_limits177b else (h<=reference[sl,None,None]-drop+1e-6)
        # Explicit native GAP remains a veto while the unmodified mapper reports
        # it. Its ORIGINAL time is retained, never renewed by tap/query.
        explicit_gap=terrain.gap[sl]&same&(terrain.seen_ms[sl]>=0)&(age>=0)
        flat_block=(positive|negative)&~stair[sl,None,None]
        candidate=fresh&~flat_block
        normals=fit_normals177(h,candidate,cfg)
        near=(xx.abs()<=cfg.seed_near_x_m)&(yy.abs()<=cfg.seed_near_y_m)
        seeds=candidate&near&((h-reference[sl,None,None]).abs()<=cfg.flat_support_tolerance_m+1e-6)
        if contact_seed_mask is not None:seeds |=contact_seed_mask[sl]&candidate
        ref,_,accepted=grow_support(h,fresh,normals,seeds,segment_type=segment[sl],current_support_height=reference[sl],surface_candidate=candidate,config=c2)
        if cfg.terrain_limits177b:
            # Directly measured near-floor rough/step surfaces; no invented normals.
            norm=torch.linalg.vector_norm(normals,dim=-1)
            horizontal=torch.isfinite(normals).all(-1)&(norm>0)&(normals[...,2]/norm.clamp_min(1e-9)>=math.cos(math.radians(cfg.normal_max_slope_deg)))
            import torch.nn.functional as F
            local_max=F.max_pool2d(torch.where(candidate,h,-torch.inf)[:,None],3,1,1)[:,0]
            local_min=-F.max_pool2d(torch.where(candidate,-h,-torch.inf)[:,None],3,1,1)[:,0]
            full=F.avg_pool2d(candidate.float()[:,None],3,1,1,count_include_pad=True)[:,0]>=1.-1e-6
            rough=full&(local_max-local_min<=cfg.rough_max_m+1e-6)
            flat_ok=candidate&(horizontal|rough)&((h-reference[sl,None,None]).abs()<=max(cfg.flat_step_max_m,cfg.rough_max_m)+1e-6)
            # Stair growth retains connected measured treads with per-step limit.
            stair_cfg=replace(c2,max_step_m=cfg.stair_step_max_m)
            ref_st,_,accepted_st=grow_support(h,fresh,normals,seeds,segment_type=segment[sl],current_support_height=reference[sl],surface_candidate=candidate,config=stair_cfg)
            accepted=torch.where(stair[sl,None,None],accepted_st,flat_ok)
            ref=torch.where(accepted,h,ref_st)
        occupied=fresh&positive&~accepted&((h-reference[sl,None,None])<=grid.body_height_m+1e-6)
        gap=explicit_gap|(fresh&negative&~accepted)
        support=accepted&~gap&~occupied
        state=torch.where(occupied,OCCUPIED,torch.where(support,FREE,UNKNOWN)).to(torch.uint8)
        # For flat obstacles the classification reference is measured floor, not
        # the top of the object. Accepted stair support keeps its own tread z.
        rz=torch.where(occupied,reference[sl,None,None],ref)
        wz=rz+terrain.pose[sl,2,None,None].float()
        bits=(1<<terrain.source[sl].long()).to(torch.uint8)
        cap=torch.zeros_like(terrain.seen_ms[sl]);profile=torch.full_like(cap,-1,dtype=torch.int32)
        args=(terrain.seen_ms[sl],bits,wz,cap,profile,terrain.pose[sl],pose[sl],source_grid,grid)
        transformed=resample(state,*args)
        gap_proj=resample(torch.where(gap,OCCUPIED,UNKNOWN).to(torch.uint8),*args)
        # Any overlap belongs to terrain for conservative ROI borders. Whole
        # target cell support still requires every overlap to be accepted.
        roi=resample(torch.full_like(state,OCCUPIED),*args)[0]==OCCUPIED
        ts,tt,tbits,tz,_,_=transformed
        tbits |=gap_proj[2]
        tg=gap_proj[0]==OCCUPIED
        if head_gap_mask is not None:tg |=head_gap_mask[sl]
        to=ts==OCCUPIED;tv=(ts==FREE)&torch.isfinite(tz)
        tz=tz-pose[sl,2,None,None].float()
        permission=torch.zeros_like(tg);permission_seen=torch.full_like(tt,-1)
        if cfg.terrain_limits177b:
            from .gap177b import walkable_gap177b
            # Projected GAP is an observation, but only fresh gaps can grant transit.
            gap_seen=gap_proj[1]
            gap_fresh=tg&(gap_seen>=0)&(now_ms[sl,None,None]-gap_seen<=500)
            seen_for_run=torch.where(gap_fresh,gap_seen,tt)
            permission,permission_seen=walkable_gap177b(gap_fresh,tv,seen_for_run,cell_m=grid.cell_m,max_width_m=cfg.gap_max_width_m,enabled=gap_axis_enabled[sl])
            # An explicit separate head gap is not a terrain-qualified width proof.
            if head_gap_mask is not None:permission &=~head_gap_mask[sl]
        ha=now_ms[sl,None,None]-head.seen_ms[sl]
        alive=(head.seen_ms[sl]>=0)&(ha>=0)&(ha<=round(cfg.obstacle_ttl_s*1000))&(head.epoch[sl]==epoch[sl])[:,None,None]
        hs=torch.where(alive,head.state[sl],UNKNOWN)
        hsup=head.support;sa=now_ms[sl,None,None]-hsup.seen_ms[sl]
        hv=hsup.observed[sl]&torch.isfinite(hsup.reference_z[sl])&(hsup.seen_ms[sl]>=0)&(sa>=0)&(sa<=500)&(head.epoch[sl]==epoch[sl])[:,None,None]
        conflict=roi&tv&hv&((tz-hsup.reference_z[sl]).abs()>cfg.support_agreement_m+1e-6)
        foot_valid=torch.where(roi,tv&~conflict,hv)&~tg&~to
        fz=torch.where(roi,tz,hsup.reference_z[sl]);fstamp=torch.where(roi,tt,hsup.seen_ms[sl])
        # Existing occupied wins and keeps historical classification height/time.
        oldocc=hs==OCCUPIED
        out=torch.where(oldocc|to,OCCUPIED,hs).to(torch.uint8)
        out=torch.where(((tg&~permission)|conflict)&~oldocc&~to,UNKNOWN,out)
        out=torch.where((out==FREE)&~(foot_valid|permission),UNKNOWN,out).to(torch.uint8)
        stamp=torch.where(oldocc,head.seen_ms[sl],torch.where(to,tt,head.seen_ms[sl]))
        if cfg.terrain_free180:
            # Current body ROI, not an old chart that can drift outside it.
            gy,gx=torch.meshgrid(torch.arange(grid.ny,device=dev),torch.arange(grid.nx,device=dev),indexing='ij')
            x=grid.x_min_m+(gx+.5)*grid.cell_m;y=grid.y_min_m+(gy+.5)*grid.cell_m
            inside=(x.abs()+grid.cell_m/2<=.85+1e-6)&(y.abs()+grid.cell_m/2<=.55+1e-6)
            grant=inside[None]&roi&(foot_valid|permission)&~conflict&~oldocc&~to
            out=torch.where(grant,FREE,out).to(torch.uint8)
            evidence_time=torch.where(permission,permission_seen,tt)
            stamp=torch.where(grant,evidence_time,stamp)
        stamp=torch.where(out!=UNKNOWN,stamp,-1)
        class_z=torch.where(oldocc,hsup.reference_z[sl],torch.where(to,tz,fz))
        # A terrain gap never turns into support, including overlapping head floor.
        foot_valid &=out!=OCCUPIED
        supstamp=torch.where(oldocc,hsup.seen_ms[sl],fstamp)
        support_seen=torch.where(to,tt,supstamp)
        outputs.append((out,stamp,class_z,foot_valid,support_seen,fz,fstamp,roi,tg,to,conflict,torch.where(tg,gap_proj[1],tt),tbits,permission&~oldocc&~to&~conflict,permission_seen))
    v=[torch.cat([b[i] for b in outputs]) for i in range(15)]
    st,seen,z,valid,ss,fz,fs,roi,gap,occ,conflict,tseen,bits,permission,permission_seen=v
    # This is a fused snapshot, not a synthetic new sensor capture. Keep both
    # ORIGINAL stream capture times as explicit health sidecars.
    capture=torch.maximum(head.capture_ms,terrain.last_capture_ms).clamp_min(0)
    frame=OccFrame176(st,seen,Support176(z,valid,ss),epoch.clone(),capture)
    return TerrainFusion177(frame,Support176(fz,valid.clone(),fs),replace(grid,ttl_s=cfg.obstacle_ttl_s,support_ttl_s=.5),roi,gap,occ,conflict,tseen,bits,head.capture_ms.clone(),terrain.last_capture_ms.clone(),permission,permission_seen)
