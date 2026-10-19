"""Bounded oracle world-axis 2D cache. No actor/terrain-input state is changed."""
from dataclasses import dataclass, replace
from pathlib import Path
import math
import torch
import yaml
from ..oracle_occgrid176 import OccConfig176, OccFrame176, Support176, UNKNOWN, FREE, OCCUPIED
from .resample import resample


def validate_settings(s):
    if s['rear_source'] not in ('assumed_sensor','temporal_map'):raise ValueError('rear_source')
    if s['pose_level']!='P0':raise ValueError('175B requires GT pose P0')
    extended=s.get('static_oracle183',False)
    if extended and (s['context']!='training_static' or s['support_ttl_s'] not in (.5,1.,2.)):raise ValueError('183 support TTL requires static oracle .5/1/2')
    if not extended and s['support_ttl_s']!=.5:raise ValueError('foot support TTL must remain 0.5s')
    if s['memory_free_ttl_s'] not in (.5,1.,2.,3.):raise ValueError('comparison TTL .5/1/2/3')
    if s['retention_s']<s['memory_free_ttl_s']:raise ValueError('retention')
    if s['initial_prior_for_certification']:raise ValueError('no initial free/support prior')
    if s['context']!='training_static' and s['memory_free_for_certification']:raise ValueError('memory certification requires explicitly static oracle context')
    if any(s['sensor_sim'][k]!=0 for k in ('range_noise_std_m','dropout_probability')):raise ValueError('oracle means no noise/dropout')
    if s['resolution_m'] not in (.1,.05) or len(s['world_cells'])!=2 or min(s['world_cells'])<3:raise ValueError('world grid')
    if s['query_chunk_envs']<=0:raise ValueError('chunk')
    if s['sensor_sim']['max_range_m']<=0 or s['sensor_sim']['ray_chunk']<=0:raise ValueError('sensor range/chunk')


def load_settings(path=None):
    root=Path(__file__).resolve().parents[4]
    path=Path(path) if path else root/'configs/gast_nav/temporal_map.yaml'
    s=yaml.safe_load(path.read_text())
    for key in ('extrinsics_file','mid360_pattern_file','ouster_emitter_file'):
        p=Path(s[key]);s[key]=str((root/p).resolve() if not p.is_absolute() else p)
    s['gdm']={k:str((root/Path(v)).resolve()) if not Path(v).is_absolute() else v for k,v in s['gdm'].items()}
    validate_settings(s)
    return s


@dataclass(frozen=True)
class TemporalEvidence175B:
    frame: OccFrame176
    cert_cfg: OccConfig176
    foot_support: Support176
    obstacle_state: torch.Tensor  # obstacle TTL only; foot support remains independent
    source_bits: torch.Tensor    # 1 OS1, 2 Mid360, 4 HEAD D455
    current_free: torch.Tensor   # all contributors from latest AVAILABLE capture
    memory_free: torch.Tensor
    capture_id: torch.Tensor     # oldest contributing capture, -1 unknown
    profile_index: torch.Tensor  # -2 mixed, -1 unknown; registry on map
    last_capture_ms: torch.Tensor # sensor-health age must NOT use query time


class TemporalMap:
    def __init__(self,n,settings=None,*,device='cpu',output_cfg=None):
        self.s=load_settings() if settings is None else settings;validate_settings(self.s)
        self.n=n;self.device=torch.device(device)
        self.cfg=output_cfg or OccConfig176(cell_m=self.s['resolution_m'])
        if self.cfg.cell_m!=self.s['resolution_m']:raise ValueError('grid resolution mismatch')
        r=self.cfg.cell_m;nx,ny=self.s['world_cells']
        self.world_cfg=replace(self.cfg,x_min_m=-nx*r/2,x_max_m=nx*r/2,y_min_m=-ny*r/2,y_max_m=ny*r/2)
        # The obstacle comparison knob applies to BOTH free and occupied. Foot is fixed.
        self.cert_cfg=replace(self.cfg,ttl_s=self.s['memory_free_ttl_s'],support_ttl_s=self.s['support_ttl_s'])
        shape=(n,ny,nx);d=self.device
        self.state=torch.zeros(shape,dtype=torch.uint8,device=d)
        self.seen=torch.full(shape,-1,dtype=torch.int64,device=d)
        self.source=torch.zeros_like(self.state)
        self.height=torch.full(shape,torch.nan,device=d) # immutable historical classification world-z
        self.cap=torch.full_like(self.seen,-1)
        self.profile=torch.full(shape,-1,dtype=torch.int32,device=d)
        self.foot_z=self.height.clone();self.foot_seen=self.seen.clone()
        self.anchor=torch.zeros((n,4),dtype=torch.float64,device=d) # yaw=0,z=0
        self.epoch=torch.full((n,),-1,dtype=torch.int64,device=d)
        self.last_ms=self.epoch.clone();self.sequence=self.epoch.clone()
        self.profiles=[]

    def _pose(self,p):
        p=torch.as_tensor(p,device=self.device,dtype=torch.float64)
        if p.shape!=(self.n,4) or not bool(torch.isfinite(p).all()):raise ValueError('GT pose [N,x/y/z/yaw]')
        return p

    def _time(self,t):
        t=torch.as_tensor(t,device=self.device)
        if t.dtype!=torch.int64 or t.shape!=(self.n,) or bool((t<0).any()):raise ValueError('int64 time/epoch [N]')
        return t

    def _recenter(self,pose,reset):
        r=self.cfg.cell_m;target=torch.round(pose[:,:2]/r)*r
        delta=torch.round((target-self.anchor[:,:2])/r).long()
        ny,nx=self.state.shape[-2:];yy,xx=torch.meshgrid(torch.arange(ny,device=self.device),torch.arange(nx,device=self.device),indexing='ij')
        x=xx[None]+delta[:,0,None,None];y=yy[None]+delta[:,1,None,None]
        valid=(x>=0)&(x<nx)&(y>=0)&(y<ny)&~reset[:,None,None]
        idx=(y.clamp(0,ny-1)*nx+x.clamp(0,nx-1)).flatten(1)
        for name,fill in (('state',0),('seen',-1),('source',0),('height',torch.nan),('cap',-1),('profile',-1),('foot_z',torch.nan),('foot_seen',-1)):
            v=getattr(self,name);setattr(self,name,torch.where(valid,v.flatten(1).gather(1,idx).reshape_as(v),fill))
        self.anchor[:,:2]=target

    def push(self,frame,pose,source_bits,*,profile_hash,occluded_mask=None):
        """Input already rasterized from first-hit head measurements, never GT map.

        occluded_mask is the current first-hit shadow in input grid. If omitted,
        discard ALL prior free (occupied retention still applies): no silent
        memory certification without a current visibility-cleanup result.
        It INVALIDATES past free only; it cannot create occupied or free evidence.
        Caller must supply it for visible dynamic blockers. Hidden rear intruders
        cannot be detected; disable memory via query(dynamic_context=True).
        """
        pose=self._pose(pose);t=self._time(frame.capture_ms);epoch=self._time(frame.epoch)
        shape=(self.n,self.cfg.ny,self.cfg.nx)
        if frame.state.shape!=shape or frame.state.dtype!=torch.uint8 or bool((frame.state>2).any()):raise ValueError('2D state')
        if source_bits.shape!=shape or source_bits.dtype!=torch.uint8 or bool((source_bits>(127 if self.s.get('static_oracle183',False) else 7)).any()):raise ValueError('head source bitmask')
        if frame.seen_ms.shape!=shape or frame.seen_ms.dtype!=torch.int64:raise ValueError('observation timestamps')
        known=frame.state!=UNKNOWN
        if bool((known&((frame.seen_ms!=t[:,None,None])|(source_bits==0))).any()):raise ValueError('state provenance')
        su=frame.support
        if su.reference_z.shape!=shape or su.observed.shape!=shape or su.seen_ms.shape!=shape or su.observed.dtype!=torch.bool or su.seen_ms.dtype!=torch.int64:raise ValueError('support metadata')
        if bool((su.observed&(~torch.isfinite(su.reference_z)|(su.seen_ms<0)|(su.seen_ms>t[:,None,None]))).any()):raise ValueError('support provenance')
        if not isinstance(profile_hash,str) or len(profile_hash)!=64 or any(c not in '0123456789abcdef' for c in profile_hash):raise ValueError('SHA256 effective sensor profile')
        reset=epoch!=self.epoch
        if bool((~reset&(t<=self.last_ms)).any()):raise ValueError('non-increasing capture; queries never refresh evidence')
        if occluded_mask is not None and (occluded_mask.shape!=shape or occluded_mask.dtype!=torch.bool):raise ValueError('shadow shape/type')
        if profile_hash not in self.profiles:
            if len(self.profiles)>=64:raise ValueError('profile registry full; explicit new map required')
            self.profiles.append(profile_hash)
        tag=self.profiles.index(profile_hash)
        self._recenter(pose,reset)
        self.sequence=torch.where(reset,0,self.sequence+1)
        chunk=self.s['query_chunk_envs']
        for start in range(0,self.n,chunk):
            sl=slice(start,min(start+chunk,self.n));p=pose[sl];anchor=self.anchor[sl]
            worldz=su.reference_z[sl]+p[:,2,None,None].float()
            cap=self.sequence[sl,None,None].expand_as(frame.state[sl]);pr=torch.full_like(frame.state[sl],tag,dtype=torch.int32)
            incoming=resample(frame.state[sl],frame.seen_ms[sl],source_bits[sl],worldz,cap,pr,p,anchor,self.cfg,self.world_cfg)
            st,seen,src,h,ca,prof=incoming
            age=t[sl,None,None]-self.seen[sl]
            alive=(self.seen[sl]>=0)&(age>=0)&(age<=round(self.s['memory_free_ttl_s']*1000))
            old=torch.where(alive,self.state[sl],UNKNOWN)
            # Foot observations are accumulated independently, NEVER refreshed by obstacle updates.
            footstate=torch.where(su.observed[sl],FREE,UNKNOWN).to(torch.uint8)
            foot=resample(footstate,su.seen_ms[sl],source_bits[sl],worldz,cap,pr,p,anchor,self.cfg,self.world_cfg)
            freshfoot=(foot[0]==FREE)&torch.isfinite(foot[3])&(foot[1]>=0)
            changed=freshfoot&torch.isfinite(self.foot_z[sl])&((foot[3]-self.foot_z[sl]).abs()>.020001)
            shadow=torch.full_like(freshfoot,occluded_mask is None)
            if occluded_mask is not None:
                marker=torch.where(occluded_mask[sl],OCCUPIED,UNKNOWN).to(torch.uint8)
                shadow=resample(marker,frame.seen_ms[sl],source_bits[sl],worldz,cap,pr,p,anchor,self.cfg,self.world_cfg)[0]==OCCUPIED
            old=torch.where((old==FREE)&(changed|shadow),UNKNOWN,old)
            # Old occupied classification AND its height remain immutable until expiry.
            keepocc=(old==OCCUPIED)&(st!=OCCUPIED)
            take=(st!=UNKNOWN)&~keepocc
            old=torch.where(take,st,old)
            self.state[sl]=old
            for name,v in zip(('seen','source','height','cap','profile'),(seen,src,h,ca,prof)):
                original=getattr(self,name)[sl]
                getattr(self,name)[sl]=torch.where(old==UNKNOWN,torch.nan if name=='height' else (0 if name=='source' else -1),torch.where(take,v,original))
            self.foot_z[sl]=torch.where(freshfoot,foot[3],self.foot_z[sl])
            self.foot_seen[sl]=torch.where(freshfoot,foot[1],self.foot_seen[sl])
            expired=(t[sl,None,None]-self.foot_seen[sl]>round(self.s['retention_s']*1000))
            self.foot_z[sl].masked_fill_(expired,torch.nan);self.foot_seen[sl].masked_fill_(expired,-1)
        self.epoch=epoch.clone();self.last_ms=t.clone()

    def query(self,pose,now_ms,epoch,*,dynamic_context=False):
        pose=self._pose(pose);now=self._time(now_ms);epoch=self._time(epoch)
        if bool((now<self.last_ms).any()):raise ValueError('query before latest capture')
        batches=[];chunk=self.s['query_chunk_envs']
        for start in range(0,self.n,chunk):
            sl=slice(start,min(start+chunk,self.n));p=pose[sl];age=now[sl,None,None]-self.seen[sl]
            valid=(self.seen[sl]>=0)&(age>=0)&(age<=round(self.s['memory_free_ttl_s']*1000))&(epoch[sl]==self.epoch[sl])[:,None,None]
            st=torch.where(valid,self.state[sl],UNKNOWN)
            result=resample(st,self.seen[sl],self.source[sl],self.height[sl],self.cap[sl],self.profile[sl],self.anchor[sl],p,self.world_cfg,self.cfg)
            st,seen,src,h,cap,prof=result
            fa=now[sl,None,None]-self.foot_seen[sl]
            footvalid=torch.isfinite(self.foot_z[sl])&(self.foot_seen[sl]>=0)&(fa>=0)&(fa<=round(self.s['support_ttl_s']*1000))&(epoch[sl]==self.epoch[sl])[:,None,None]
            foot=resample(torch.where(footvalid,FREE,UNKNOWN).to(torch.uint8),self.foot_seen[sl],self.source[sl],self.foot_z[sl],self.cap[sl],self.profile[sl],self.anchor[sl],p,self.world_cfg,self.cfg)
            fz=foot[3]-p[:,2,None,None].float();fo=(foot[0]==FREE)&torch.isfinite(fz)
            agrees=torch.isfinite(h)&((h-foot[3]).abs()<=.020001)
            ref=h-p[:,2,None,None].float()
            # Foot support is genuine latest support, but classification frame must
            # not silently change a retained occupied cell's height reference.
            sup=fo&agrees
            current=(st==FREE)&(cap==self.sequence[sl,None,None])
            memory=(st==FREE)&~current
            allow=self.s['memory_free_for_certification'] and not dynamic_context
            publish=torch.where(memory&~torch.full_like(memory,allow),UNKNOWN,st)
            # A stale stream cannot be treated as current merely by repeated queries.
            publish=torch.where(current&((now[sl]-self.last_ms[sl])>500)[:,None,None],UNKNOWN,publish)
            batches.append((publish,seen,ref,sup,foot[1],fz,fo,st,src,current,memory,cap,prof))
        values=[torch.cat([b[i] for b in batches]) for i in range(13)]
        pub,seen,ref,sup,fs,fz,fo,st,src,cur,mem,cap,prof=values
        frame=OccFrame176(pub,seen,Support176(ref,sup,fs),epoch.clone(),self.last_ms.clamp_min(0).clone())
        return TemporalEvidence175B(frame,self.cert_cfg,Support176(fz,fo,fs),st,src,cur,mem,cap,prof,self.last_ms.clone())


def select_rear_source(settings,*,assumed_sensor=None,temporal_map=None):
    selected={'assumed_sensor':assumed_sensor,'temporal_map':temporal_map}[settings['rear_source']]
    if selected is None:raise ValueError('selected rear source unavailable; no silent fallback')
    return selected
