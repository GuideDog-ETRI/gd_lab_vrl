"""Query-only terrain fusion; never writes fused evidence into sensor memory."""
from dataclasses import fields,replace
import torch
from .oracle_occgrid176 import OccFrame176,Support176
from .isaac_occgrid176 import original_timestamp176
from .terrain_fusion177 import FusionConfig177,fuse_terrain177
from .support174 import actual_contacts174
from .obs_terms import world_to_sample

def fuse_query177(producer,observation,pose,time_ms,epoch):
 env=producer.env;terrain=getattr(env,'_terrain177',None)
 if terrain is None:return replace(observation,state=torch.zeros_like(observation.state))
 head=OccFrame176(observation.state.clone(),original_timestamp176(time_ms,observation.age_s),Support176(observation.support.clone(),torch.isfinite(observation.support),original_timestamp176(time_ms,observation.support_age_s)),epoch.clone(),producer.frame.capture_ms.clone())
 feet=[producer.nav.robot.body_names[i] for i in producer.nav.foot_ids]
 points,valid=actual_contacts174(env,feet,terrain.pose.to(pose.dtype),world_to_sample)
 reference=points[...,2].masked_fill(~valid,torch.nan).nanmedian(-1).values
 reference_valid=torch.isfinite(reference)
 ids=reference_valid.nonzero().flatten()
 gap_permission=torch.zeros_like(head.state,dtype=torch.bool);gap_seen=torch.full_like(head.seen_ms,-1)
 # No measured current support: cannot evaluate low objects/gap connectivity.
 head.state[~reference_valid]=torch.where(head.state[~reference_valid]==2,2,0).to(head.state.dtype)
 head_fresh=(time_ms-producer.frame.capture_ms>=0)&(time_ms-producer.frame.capture_ms<=300)
 terrain_fresh=(time_ms-terrain.last_capture_ms>=0)&(time_ms-terrain.last_capture_ms<=300)
 if len(ids):
  t=type(terrain)(**{f.name:getattr(terrain,f.name)[ids] for f in fields(terrain)})
  h=OccFrame176(head.state[ids],head.seen_ms[ids],Support176(head.support.reference_z[ids],head.support.observed[ids],head.support.seen_ms[ids]),head.epoch[ids],head.capture_ms[ids])
  from .command_generator import route_hint
  from .schema import SEGMENT_TYPES
  xy=(pose[:,:2]-env.scene.env_origins[:,:2]).tolist()
  segments=torch.tensor([SEGMENT_TYPES.index(route_hint(meta,p)[0].segment_type) for meta,p in zip(producer.nav.index.lookup(range(len(pose))),xy)],device=pose.device)
  cfg=replace(FusionConfig177.load(),obstacle_ttl_s=producer.opts['obstacle_ttl_s'],terrain_free180=bool(getattr(producer.nav,'cfg',{}).get('pipeline180',False)))
  result=fuse_terrain177(h,t,target_pose=pose[ids].double(),now_ms=time_ms[ids],epoch=epoch[ids],grid=producer.ocfg,config=cfg,measured_current_support=reference[ids],segment_type=segments[ids],gap_axis_enabled=torch.tensor([c.requested_mode=='NAV' for c in producer.nav.control.commands.commands],device=pose.device)[ids])
  gap_permission[ids]=result.walkable_gap;gap_seen[ids]=result.walkable_gap_seen_ms
  producer.stats['walkable_gap_cells']=int(result.walkable_gap.sum())
  f=result.frame
  terrain_only=(f.state==1)&(h.state!=1)
  producer.last_query169['terrain_assumed_free180']=dict(ids=ids.clone(),mask=terrain_only.clone())
  producer.stats['terrain_assumed_free180_cells']=int(terrain_only.sum())
  head.state[ids]=f.state;head.seen_ms[ids]=f.seen_ms;head.support.reference_z[ids]=f.support.reference_z;head.support.observed[ids]=f.support.observed;head.support.seen_ms[ids]=f.support.seen_ms
  producer.stats.update(terrain_fused_envs=len(ids),terrain_gap_cells=int(result.gap.sum()),terrain_occupied_cells=int(result.terrain_occupied.sum()),terrain_conflict_cells=int(result.support_conflict.sum()),terrain_valid_cells=int(terrain.valid.sum()))
 else:producer.stats['terrain_fused_envs']=0
 head.state[~(head_fresh&terrain_fresh)]=0
 producer.stats.update(terrain_fresh_envs=int(terrain_fresh.sum()),head_fresh_envs=int(head_fresh.sum()))
 producer.last_query169['fusion_frame177']=head
 return replace(observation,state=head.state,age_s=(time_ms[:,None,None]-head.seen_ms).float()/1000,support=torch.where(head.support.observed,head.support.reference_z,torch.nan),support_age_s=(time_ms[:,None,None]-head.support.seen_ms).float()/1000,walkable_gap=gap_permission,walkable_gap_age_s=(time_ms[:,None,None]-gap_seen).float()/1000)
