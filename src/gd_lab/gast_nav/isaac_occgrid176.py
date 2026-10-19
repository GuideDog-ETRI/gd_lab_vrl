"""Native oracle supervisor adapter. Scene geometry stays inside ray producer."""
from dataclasses import replace
from pathlib import Path
import time,torch,yaml
from .isaac_coverage174 import IsaacCoverage174
from .native176 import Native176
from .oracle_patterns176 import mounted_rays176,apply_self_sectors176
from .oracle_occgrid176 import OccConfig176,Support176,OccFrame176,oracle_first_hits176,rasterize176
from .coverage174 import scene_in_sample174
from .map_types import WIDE_GRID,NEAR_GRID
from .occgrid_gate176 import OccupancyInput176
from .support174 import actual_contacts174
from .support import encode_support
from .warp import warp_categorical
from .free_evidence167 import _rotation
from gd_lab.core.camera_geometry import camera_rotation_matrix

def original_timestamp176(now_ms,age_s):
 valid=torch.isfinite(age_s)&(age_s>=0)&(age_s*1000<=now_ms[:,None,None]+.001)
 age=torch.where(valid,age_s,0.)
 return torch.where(valid,now_ms[:,None,None]-(age*1000).round().long(),-1)

class IsaacOccGrid176(IsaacCoverage174):
 def __init__(self,nav):
  super().__init__(nav) # reuse audited physical inventory only; no 174 captures
  self.native=Native176(self.env.device);self.frame=None;self.frame_pose=None;self.capture_index=0
  self.ocfg=OccConfig176.load(Path(__file__).resolve().parents[3]/'configs/gast_nav/oracle_occgrid176.yaml')
  self.opts=nav.cfg['supervisor176'];self.ocfg=replace(self.ocfg,ttl_s=self.opts['obstacle_ttl_s'])
  if self.opts.get('stopping_model')=='analytic177':
   from .analytic_sweep177 import AnalyticSweep177
   settings=yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/gast_nav/analytic_sweep177.yaml').read_text())
   self.envelope=AnalyticSweep177(settings)
  if nav.cfg.get("pipeline179",False):
   if not nav.cfg.get('terrain_fusion177',False):raise ValueError('D-5 requires measured terrain fusion')
   from .sweep179 import MeasuredSweep179
   settings=yaml.safe_load((Path(__file__).resolve().parents[3]/("configs/gast_nav/pipeline180.yaml" if nav.cfg.get("pipeline180",False) else "configs/gast_nav/pipeline179.yaml")).read_text())
   self.envelope=MeasuredSweep179(settings)
  self.grid=WIDE_GRID
  if self.opts['support_ttl_s']!=.5 and not nav.cfg.get('front_depth183',{}).get('enabled',False):raise ValueError('UPDATE176 support TTL remains 0.5s')
  self.wide_support_known=torch.zeros(self.env.num_envs,WIDE_GRID.ny,WIDE_GRID.nx,device=self.env.device,dtype=torch.bool)
  self.near_support_known=torch.zeros(self.env.num_envs,NEAR_GRID.ny,NEAR_GRID.nx,device=self.env.device,dtype=torch.bool)
  self.safety_clearance=torch.full((self.env.num_envs,2,NEAR_GRID.ny,NEAR_GRID.nx),torch.inf,device=self.env.device)
 def reset(self,ids):
  super().reset(ids)
  if self.frame is not None:
   self.frame.state[ids]=0;self.frame.seen_ms[ids]=-1;self.frame.support.observed[ids]=False;self.frame.support.seen_ms[ids]=-1
 def capture(self,*,pose,time_ms,epoch,config,**ignored):
  from .obs_terms import observed_support,world_to_sample
  from .command_generator import route_hint
  from . import schema as S
  start=time.perf_counter();nav=self.nav;n=len(pose);env=self.env
  if bool(nav.cylinder.enabled.any()):raise ValueError('C static inventory only')
  complete=((pose[:,:2]-env.scene.env_origins[:,:2]).abs().amax(-1)+4.<12.)
  complete &= ((nav.robot.data.body_pos_w-env.scene.env_origins[:,None])[...,:2].abs().amax((1,2))+.6<12.).all()
  scene=scene_in_sample174(replace(self.scene,complete=complete),pose)
  dirs,src,rings,mount=self.native.pattern_at(self.capture_index);self.capture_index+=1
  root=_rotation(-pose[:,3])@camera_rotation_matrix(nav.robot.data.root_quat_w)
  rotations=root[:,None]@self.native.rot[None];origins=torch.einsum('nij,cj->nci',root,self.native.pos)
  # Unspecified rear coverage is defined in current body-yaw hip halfspace.
  # It is not a hardware mount rotated out of its own assumed boundary.
  rotations[:,2]=torch.eye(3,device=pose.device);origins[:,2]=pose.new_tensor([-.31,0.,0.])
  rays,local=mounted_rays176(dirs,src,rings,mount,rotations,origins,self.native.profile)
  # Filter excluded rays before raycasting; their data never creates free.
  keep=rays.coverage_valid.any(0)
  rays=replace(rays,**{k:getattr(rays,k)[:,keep] for k in rays.__dataclass_fields__});local=local[:,keep]
  ids=time_ms+epoch*1000000000
  hits=apply_self_sectors176(oracle_first_hits176(scene,rays,scene_capture_id=ids,ray_capture_id=ids),local,self.native.profile)
  hit_valid=hits.valid&torch.isfinite(hits.distance_m)&~hits.self_hit
  points=rays.origins+rays.directions*torch.nan_to_num(hits.distance_m,posinf=0.)[...,None]
  contact,cv=actual_contacts174(env,[nav.robot.body_names[i] for i in nav.foot_ids],pose,world_to_sample)
  xy=(pose[:,:2]-env.scene.env_origins[:,:2]).tolist()
  segments=torch.tensor([S.SEGMENT_TYPES.index(route_hint(t,p)[0].segment_type) for t,p in zip(nav.index.lookup(range(n)),xy)],device=env.device)
  # Exact support-only cull: points outside every cell's normal-fit radius
  # cannot contribute. Occupancy still consumes ALL native first returns.
  radius=nav.pipeline.settings['support_normal_radius_m']
  boundlo=points.new_tensor([self.grid.x0-radius,self.grid.y0-radius])
  boundhi=points.new_tensor([self.grid.x0+(self.grid.nx-1)*self.grid.res+radius,self.grid.y0+(self.grid.ny-1)*self.grid.res+radius])
  relevant=hit_valid&((points[...,:2]>=boundlo)&(points[...,:2]<=boundhi)).all(-1)
  count=1<<max(0,(max(1,int(relevant.sum(-1).max()))-1).bit_length())
  order=relevant.to(torch.uint8).argsort(dim=1,descending=True,stable=True)[:,:count]
  fit_points=points.gather(1,order[:,:,None].expand(-1,-1,3));fit_valid=relevant.gather(1,order)
  if fit_points.shape[1]<count:
   pad=count-fit_points.shape[1]
   fit_points=torch.cat((fit_points,points.new_zeros(n,pad,3)),1);fit_valid=torch.cat((fit_valid,torch.zeros(n,pad,device=pose.device,dtype=torch.bool)),1)
  support_fn=observed_support
  if nav.cfg.get('pipeline179',False):
   from .head_support182 import observed_head_support182
   support_fn=observed_head_support182
  support=support_fn(fit_points,fit_valid,contact,cv,segments,self.grid,nav.pipeline.settings,config=config)
  support_time=time_ms[:,None,None].expand_as(support).clone();support_time[~torch.isfinite(support)]=-1
  previous=None
  if self.frame is not None:
   previous,_=self.query_occupancy176(pose=pose,time_ms=time_ms,epoch=epoch)
   old_ok=torch.isfinite(previous.support)&(previous.support_age_s<=.5)&(previous.support_age_s>=0)
   take=~torch.isfinite(support)&old_ok
   support=torch.where(take,previous.support,support)
   support_time=torch.where(take,original_timestamp176(time_ms,previous.support_age_s),support_time)
  s=Support176(support,torch.isfinite(support),support_time)
  threshold=None
  if self.nav.cfg.get('terrain_fusion177',False):
   from .terrain_fusion177 import FusionConfig177
   limits=FusionConfig177.load()
   if limits.terrain_limits177b:
    stairs=(segments==S.SEGMENT_TYPES.index('STAIR_UP'))|(segments==S.SEGMENT_TYPES.index('STAIR_DOWN'))|(segments==S.SEGMENT_TYPES.index('LANDING'))
    threshold=torch.where(stairs,limits.stair_step_max_m,limits.low_object_max_m).to(pose.dtype)
  if self.nav.cfg.get('pipeline179',False):
   threshold=torch.full((n,),limits.low_object_max_m,device=pose.device,dtype=pose.dtype)
  frame=rasterize176(hits,s,self.ocfg,capture_ms=time_ms,epoch=epoch,obstacle_min_height_m=threshold)
  if previous is not None:
   # Preserve old occupied with its original height/time, but NEVER old free.
   keep=(previous.state==2)&(previous.age_s<=self.opts['obstacle_ttl_s'])&(previous.age_s>=0)&(frame.state!=2)
   frame.state[keep]=2
   frame.seen_ms[keep]=(original_timestamp176(time_ms,previous.age_s))[keep]
   frame.support.reference_z[keep]=previous.support[keep]
   frame.support.seen_ms[keep]=(original_timestamp176(time_ms,previous.support_age_s))[keep]
   frame.support.observed[keep]=torch.isfinite(previous.support[keep])
  self.frame=frame;self.frame_pose=pose.clone();self.last_time=time_ms.clone();self.last_epoch=epoch.clone()
  self.stats=dict(oracle=True,native_rays=rays.near_m.shape[1],support_fit_points=count,capture_wall_s=time.perf_counter()-start,free_cells=int((frame.state==1).sum()),occupied_cells=int((frame.state==2).sum()),support_cells=int(torch.isfinite(support).sum()),captures=self.capture_index,volume_proof=False)
  self.support_debug174=dict(contact=contact,contact_valid=cv,support=support)
  self.last_query169=dict(occupancy_frame=frame,pose=pose.clone())
  self.swept_status='OCCGRID_STOPPING_SWEEP'
  # These legacy 3-band diagnostic tensors do not enter actor OBS1 or the gate.
  wide=torch.zeros(n,3,WIDE_GRID.ny,WIDE_GRID.nx,device=env.device,dtype=torch.uint8)
  near=torch.zeros(n,3,NEAR_GRID.ny,NEAR_GRID.nx,device=env.device,dtype=torch.uint8)
  return (wide,torch.full_like(wide,torch.inf,dtype=torch.float32)),(near,torch.full_like(near,torch.inf,dtype=torch.float32))
 def query(self,*,pose,time_ms,epoch,config):
  self.query_occupancy176(pose=pose,time_ms=time_ms,epoch=epoch)
  n=len(pose);dev=pose.device
  wide=torch.zeros(n,3,WIDE_GRID.ny,WIDE_GRID.nx,device=dev,dtype=torch.uint8)
  near=torch.zeros(n,3,NEAR_GRID.ny,NEAR_GRID.nx,device=dev,dtype=torch.uint8)
  return (wide,torch.full_like(wide,torch.inf,dtype=torch.float32)),(near,torch.full_like(near,torch.inf,dtype=torch.float32))
 def query_occupancy176(self,*,pose,time_ms,epoch):
  f=self.frame
  if f is None:raise RuntimeError('capture required')
  cfg=replace(self.nav.pipeline.config,ttl_s=self.opts['obstacle_ttl_s'])
  age=(time_ms[:,None,None]-f.seen_ms).float()/1000
  a=age[:,None].expand(-1,3,-1,-1);st=f.state[:,None].expand_as(a)
  sup=f.support.reference_z
  ws,wa,_=warp_categorical(st,a,self.frame_pose,pose,grid=self.grid,config=cfg,support=encode_support(sup),support_world_z=sup+self.frame_pose[:,2,None,None],return_support_height=True)
  sa=(time_ms[:,None,None]-f.support.seen_ms).float()/1000
  marker=f.support.observed[:,None].expand_as(a).to(torch.uint8)
  ss,sage,height=warp_categorical(marker,sa[:,None].expand_as(a),self.frame_pose,pose,grid=self.grid,config=replace(cfg,ttl_s=.5),support=encode_support(sup),support_world_z=sup+self.frame_pose[:,2,None,None],return_support_height=True)
  eligible=(epoch==f.epoch)&(time_ms>=f.capture_ms)
  state=torch.where(eligible[:,None,None],ws[:,0],0)
  support=torch.where((ss[:,0]==1)&eligible[:,None,None],height,torch.nan)
  self.wide_support_known=torch.isfinite(support)
  return OccupancyInput176(state,wa[:,0],support,sage[:,0],f.epoch),self.grid
