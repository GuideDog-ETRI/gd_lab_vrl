"""Opt-in static oracle camera + temporal-map supervisor; actor path unchanged."""
from dataclasses import replace
import torch
from .isaac_occgrid176 import IsaacOccGrid176,original_timestamp176
from .oracle_occgrid176 import OccFrame176,Support176
from .front_depth183 import merge_camera_frame183,virtual_entry183
from .temporal_map.memory import TemporalMap,load_settings
from .camera_scene183 import Cameras183

class IsaacFront183(IsaacOccGrid176):
 def __init__(self,nav):
  opt=nav.cfg['front_depth183']
  if not opt.get('enabled') or opt.get('context')!='training_static' or not opt.get('oracle'):raise ValueError('183 requires explicit static oracle')
  if opt['obstacle_ttl_s']!=opt['support_ttl_s'] or opt['support_ttl_s'] not in (.5,1.,2.):raise ValueError('183 comparison .5/1/2')
  super().__init__(nav)
  self.ocfg=replace(self.ocfg,front_depth183=True)
  self.camera=Cameras183(self);self.memcfg=load_settings()
  self.memcfg.update(static_oracle183=True,context='training_static',support_ttl_s=opt['support_ttl_s'],memory_free_ttl_s=opt['obstacle_ttl_s'],memory_free_for_certification=True)
  self.memory=TemporalMap(self.env.num_envs,self.memcfg,device=self.env.device,output_cfg=self.ocfg)
  self.memory_capture=-1;self.entry_pending=torch.ones(self.env.num_envs,device=self.env.device,dtype=torch.bool);self.entry_stats={};self.entry_history=[];self.offset_ms=20000

 def capture(self,**kw):
  result=super().capture(**kw)
  if hasattr(self.env,'_e1terrain177'):
   extra=self.camera.capture(kw['pose'],kw['time_ms'],kw['epoch'],kw['config'])
   self.frame=merge_camera_frame183(self.frame,extra)
   self.last_query169['occupancy_frame']=self.frame
   self.stats.update(self.camera.last_stats)
  return result

 def reset(self,ids):
  super().reset(ids)
  if hasattr(self,'memory'):
   self.memory.epoch[ids]=-1
   self.entry_pending[ids]=True
   # Other episodes retain their histories; these rows wait for a new capture.

 def _push(self,frame,pose,*,time_offset=True):
  t=frame.capture_ms+(self.offset_ms if time_offset else 0)
  seen=torch.where(frame.seen_ms>=0,frame.seen_ms+(self.offset_ms if time_offset else 0),-1)
  ss=torch.where(frame.support.seen_ms>=0,frame.support.seen_ms+(self.offset_ms if time_offset else 0),-1)
  # A held prior state is NOT a new capture. The map keeps original evidence.
  state=torch.where(seen==t[:,None,None],frame.state,0).to(torch.uint8)
  f=OccFrame176(state,torch.where(state!=0,seen,-1),Support176(frame.support.reference_z,frame.support.observed,ss),frame.epoch,t)
  # Bit 64 denotes fused observed supervisor sources (native head/front/terrain),
  # explicitly distinct from the original 175B head-only source bit assignments.
  bits=torch.where((state!=0)|f.support.observed,64,0).to(torch.uint8)
  # Static scene: no moving object can invalidate previously measured free.
  # Known occupied still wins in TemporalMap. This is not a deployment setting.
  shadow=torch.zeros_like(state,dtype=torch.bool)
  self.memory.push(f,pose,bits,profile_hash=self.camera.hash,occluded_mask=shadow)

 def bootstrap(self,pose,time_ms,epoch,config):
  if not bool(self.entry_pending.any()) or not bool((self.support_debug174['contact_valid'].any(-1)&self.entry_pending).any()):return
  todo=self.entry_pending.clone();self.entry_pending[todo]=False
  previous_memory=self.memory
  self.memory=TemporalMap(self.env.num_envs,self.memcfg,device=self.env.device,output_cfg=self.ocfg)
  cmds=self.nav.control.commands.commands
  entry=self.nav.cfg['front_depth183']['virtual_entry']
  speed=pose.new_tensor([entry['turn_speed_mps'] if c.requested_mode=='TURN' else c.v_cap for c in cmds])
  axis=pose.new_tensor([c.route_hint.route_axis_yaw if c.route_hint else float(pose[i,3]) for i,c in enumerate(cmds)])
  seen=0;skipped=0;support=0;collision_parts=None;accepted_counts=torch.zeros_like(epoch)
  for past,stamp in virtual_entry183(pose,axis,speed,time_ms+self.offset_ms,distance_m=entry['distance_m'],spacing_m=entry['spacing_m']):
   shift=past[:,:3]-pose[:,:3];valid=self.camera.collision_free(past,shift)&todo
   seen+=int(valid.sum());skipped+=int((~valid&todo).sum());accepted_counts+=valid.long()
   per_part=(self.camera.last_overlap.any(-1)&todo[:,None]).sum(0)
   collision_parts=per_part if collision_parts is None else collision_parts+per_part
   if bool(valid.any()):
    f=self.camera.capture(past,stamp,epoch,config,shift=shift,eligible=valid)
    # Invalid virtual poses cannot supply support either.
    f.support.observed[~valid]=False;f.support.reference_z[~valid]=torch.nan;f.support.seen_ms[~valid]=-1
    support+=int(f.support.observed.sum())
   else:
    z=torch.full_like(self.frame.support.reference_z,torch.nan);st=torch.zeros_like(self.frame.state);tt=torch.full_like(self.frame.seen_ms,-1)
    f=OccFrame176(st,tt,Support176(z,torch.zeros_like(st,dtype=torch.bool),tt.clone()),epoch,stamp)
   self._push(f,past,time_offset=False)
  from .entry_memory183 import copy_seeded_rows183
  seeded=self.memory;self.memory=previous_memory;copy_seeded_rows183(self.memory,seeded,todo)
  self.entry_stats=dict(env_ids=todo.nonzero().flatten().tolist(),accepted_counts_by_env=accepted_counts.tolist(),at_sim_ms=time_ms.tolist(),label='가상 진입 관측(측정된 보행 이력 아님)',accepted_pose_envs=seen,collision_skipped_pose_envs=skipped,support_observations=support,speeds_mps=speed.tolist(),collision_part_counts=[dict(link=self.nav.robot.body_names[p[0]],count=int(count)) for p,count in zip(self.nav.footprint.parts,collision_parts)])
  self.entry_history.append(self.entry_stats)

 def remember(self,observation,pose,time_ms,epoch):
  cap=int(self.frame.capture_ms.max())
  if cap!=self.memory_capture:
   self.bootstrap(pose,time_ms,epoch,self.nav.pipeline.config)
   self.memory_capture=cap
   # fuse_query177/contact carry their actual timestamps; held evidence removed.
   f=OccFrame176(observation.state.clone(),original_timestamp176(time_ms,observation.age_s),Support176(observation.support.clone(),torch.isfinite(observation.support),original_timestamp176(time_ms,observation.support_age_s)),epoch.clone(),time_ms.clone())
   self._push(f,pose)
  now=time_ms+self.offset_ms
  result=self.memory.query(pose,now,epoch)
  f=result.frame;support=result.foot_support
  self.stats.update(memory183_free=int(result.memory_free.sum()),memory183_support=int(support.observed.sum()),support183_ttl_s=self.memcfg['support_ttl_s'])
  self.last_query169['memory183']=result;self.last_query169['entry183']=self.entry_stats
  remembered=replace(observation,state=f.state,age_s=(now[:,None,None]-f.seen_ms).float()/1000,
   support=torch.where(support.observed,support.reference_z,torch.nan),support_age_s=(now[:,None,None]-support.seen_ms).float()/1000)
  from .front_depth183 import supplement_memory183
  self.last_query169['current183']=observation
  return supplement_memory183(observation,remembered,obstacle_ttl_s=self.opts['obstacle_ttl_s'],support_ttl_s=self.opts['support_ttl_s'])
