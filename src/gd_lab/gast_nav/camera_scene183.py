"""Full-pixel front-camera oracle on the audited easy-C physical inventory.

Scene geometry is used solely for nearest ray intersections and virtual-entry
collision rejection. It is never copied into the occupancy/support grid.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json,torch,yaml
from .oracle_occgrid176 import OracleRays176,Support176,OccFrame176
from .producer168 import SensorPrimitives
from .coverage174 import scene_in_sample174
from .free_evidence167 import _rotation
from gd_lab.core.camera_geometry import camera_rotation_matrix
from .temporal_map.sensors import rotation
from .front_depth183 import merge_camera_frame183

class Cameras183:
 def __init__(self,producer):
  self.p=producer;self.nav=producer.nav;self.env=producer.env;dev=self.env.device
  path=Path('/home/user/gd_project/gdm_docker/workspace/src/gdm_planning/resource/sensor_configuration.yaml')
  raw=path.read_bytes();data=yaml.safe_load(raw)['transforms'];os=data['os_sensor'];cam=data['camera_link']
  if cam['parent']!='os_sensor' or os['parent']!='base_link':raise ValueError('gdm camera chain')
  ro=rotation(os['quat_xyzw'],dev);rc=rotation(cam['quat_xyzw'],dev)
  self.head_rot=ro@rc;self.head_pos=torch.tensor(os['xyz'],device=dev)+ro@torch.tensor(cam['xyz'],device=dev)
  w,h=1280,720;f=1.93/3.896*w
  v,u=torch.meshgrid(torch.arange(h,device=dev),torch.arange(w,device=dev),indexing='ij')
  local=torch.stack((torch.ones_like(u),(w/2-u-.5)/f,(h/2-v-.5)/f),-1).float().reshape(-1,3)
  self.head_length=local.norm(dim=-1);self.head_dirs=(local/self.head_length[:,None])@self.head_rot.T
  self.manifest=dict(gdm_file=str(path),gdm_sha256=hashlib.sha256(raw).hexdigest(),os_sensor=os,camera_link=cam,
   d455=dict(resolution=[w,h],fx=f,fy=f,optical_clip_m=[.1,100.],stride=1),
   lower=dict(indices=[0,1],source='UPDATE177 OracleCamera177 intrinsics and GT first-hit inventory',optical_clip_m=[.15,5.],stride=1,trunk='same flush-mount trunk exclusion as UPDATE177; legs/feet retained'),
   support_fit='actual-return representatives on 5cm XY subcells; occupancy consumes every pixel',oracle=True)
  self.hash=hashlib.sha256(json.dumps(self.manifest,sort_keys=True).encode()).hexdigest()

 def self_scene(self,pose,shift,*,lower=False):
  nav=self.nav;data=nav.robot.data;rot=camera_rotation_matrix(data.body_quat_w);parts=nav.footprint.parts
  centres=[];axes=[];halves=[];kinds=[]
  for i,kind,offset,local,size in parts:
   if lower and nav.robot.body_names[i]=='trunk':continue
   centres.append(data.body_pos_w[:,i]+shift+torch.einsum('nij,j->ni',rot[:,i],offset))
   axes.append(rot[:,i]@local);halves.append(size.expand(3) if kind=='sphere' else size);kinds.append(2 if kind=='sphere' else 0)
  n=len(pose);dev=pose.device
  return scene_in_sample174(SensorPrimitives(torch.stack(centres,1),torch.stack(axes,1),torch.stack(halves)[None].expand(n,-1,-1).contiguous(),torch.tensor(kinds,device=dev)[None].expand(n,-1).contiguous(),torch.ones(n,len(kinds),device=dev,dtype=torch.bool),torch.ones(n,device=dev,dtype=torch.bool)),pose)

 def combined_scene(self,pose,shift,lower):
  complete=(pose[:,:2]-self.env.scene.env_origins[:,:2]).abs().amax(-1)+4.<12.
  complete &= ((self.nav.robot.data.body_pos_w-self.env.scene.env_origins[:,None])[...,:2].abs().amax((1,2))+.6<12.).all()
  scene=scene_in_sample174(replace(self.p.scene,complete=complete),pose);body=self.self_scene(pose,shift,lower=lower)
  def cat(k):return torch.cat((getattr(scene,k),getattr(body,k)),1)
  mask=torch.cat((torch.zeros_like(scene.active),body.active),1)
  return SensorPrimitives(*(cat(k) for k in ('centres','rotations','half','kind','active')),scene.complete),mask

 def collision_free(self,pose,shift):
  from .entry_collision183 import overlapping_parts183
  scene=scene_in_sample174(self.p.scene,pose);body=self.self_scene(pose,shift)
  overlap=overlapping_parts183(body,scene)
  self.last_overlap=overlap
  return ~overlap.flatten(1).any(-1)&scene.complete

 def streams(self,pose,*,shift=None,time_ms=None):
  from .camera_cuda183 import cast183
  nav=self.nav;data=nav.robot.data;n=len(pose);live=shift is None;shift=torch.zeros_like(pose[:,:3]) if live else shift
  root=camera_rotation_matrix(data.root_quat_w);chart=_rotation(-pose[:,3]);base=chart@root
  sensor=self.env._e1terrain177.sensor
  k=torch.as_tensor(sensor.K,device=pose.device,dtype=pose.dtype)
  from .e1_terrain177 import CAM_H,CAM_W
  v,u=torch.meshgrid(torch.arange(CAM_H,device=pose.device),torch.arange(CAM_W,device=pose.device),indexing='ij')
  local=torch.stack(((u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-torch.ones_like(u)),-1).float().reshape(-1,3)
  length=local.norm(dim=-1)
  qs=torch.as_tensor(sensor.Q,device=pose.device,dtype=pose.dtype)
  lowerrot=camera_rotation_matrix(qs);lowerpos=torch.as_tensor(sensor.P,device=pose.device,dtype=pose.dtype)
  trunk=sensor.trunk;tr=chart@camera_rotation_matrix(data.body_quat_w[:,trunk])
  specs=[(5+c,lowerpos[c],(local/length[:,None])@lowerrot[c].T,length,.15,5.,tr,data.body_pos_w[:,trunk]+shift) for c in (0,1)]
  specs.append((4,self.head_pos,self.head_dirs,self.head_length,.1,100.,base,data.root_pos_w+shift))
  self.manifest['lower']['resolution']=[CAM_W,CAM_H];self.manifest['lower']['K']=sensor.K.tolist()
  self.hash=hashlib.sha256(json.dumps(self.manifest,sort_keys=True).encode()).hexdigest()
  if live:
   # Consume the original unfiltered UPDATE177 depth, not a second camera model.
   import numpy as np
   from .front_depth183 import packet_returns183
   packet=getattr(sensor,'raw_front183',None)
   if packet is None or not np.array_equal(packet['capture_ms'],time_ms.cpu().numpy()):
    sensor.capture(np.zeros(n,dtype=np.int64),round(float(time_ms.max())/(self.env.step_dt*1000)))
    packet=sensor.raw_front183
   if not np.array_equal(packet['capture_ms'],time_ms.cpu().numpy()):raise ValueError('183 raw depth timestamp mismatch')
   yield from packet_returns183(packet,pose)
   specs=specs[-1:]
  for source,pos,directions,lens,near,far,rot,basepos in specs:
   scene,selfmask=self.combined_scene(pose,shift,source!=4)
   origin=torch.einsum('nij,j->ni',rot,pos)+torch.einsum('nij,nj->ni',chart,basepos-pose[:,:3])
   for start in range(0,len(lens),32768):
    locald=directions[start:start+32768];ll=lens[start:start+32768]
    direction=torch.einsum('nij,rj->nri',rot,locald);o=origin[:,None].expand_as(direction).contiguous()
    # A tile's complete physical inventory covers the grid; rays leaving the
    # tile could hit a foreign robot. Such rays are clipped to inventory boundary
    # only for free, never manufactured as a floor hit. All grid crossings precede it.
    r=OracleRays176(o,direction,(ll*near)[None].expand(n,-1),(ll*far)[None].expand(n,-1),torch.ones(n,len(ll),device=pose.device,dtype=torch.bool),torch.full((n,len(ll)),source,device=pose.device,dtype=torch.long))
    yield cast183(scene,r,selfmask)

 def capture(self,pose,time_ms,epoch,config,*,shift=None,eligible=None):
  from .camera_cuda183 import raster183
  from .head_support182 import observed_head_support182
  from .support174 import actual_contacts174
  from .obs_terms import world_to_sample
  from .command_generator import route_hint
  from .schema import SEGMENT_TYPES
  grid=self.p.grid;n=len(pose);device=pose.device
  # Bound the fit cloud to real representatives; no mean point/interpolation.
  nx,ny=grid.nx*2,grid.ny*2;total=nx*ny
  best=torch.full((n,total),torch.inf,device=device);points=torch.zeros(n,total,3,device=device)
  streams=[];raycount=0
  for ret in self.streams(pose,shift=shift,time_ms=time_ms):
   if eligible is not None:ret=replace(ret,valid=ret.valid&eligible[:,None])
   streams.append(ret);raycount+=ret.distance_m.shape[1]
   hit=ret.valid&torch.isfinite(ret.distance_m)&~ret.self_hit
   pts=ret.rays.origins+ret.rays.directions*torch.nan_to_num(ret.distance_m,posinf=0)[...,None]
   ix=torch.floor((pts[...,0]-self.p.ocfg.x_min_m)/.05).long();iy=torch.floor((pts[...,1]-self.p.ocfg.y_min_m)/.05).long()
   inside=hit&(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny);index=iy.clamp(0,ny-1)*nx+ix.clamp(0,nx-1)
   z=torch.full_like(best,torch.inf);z.scatter_reduce_(1,index,torch.where(inside,pts[...,2],torch.inf),reduce='amin')
   chosen=inside&(pts[...,2]==z.gather(1,index));ids=torch.arange(pts.shape[1],device=device)[None].expand(n,-1)
   pick=torch.full((n,total),pts.shape[1],device=device,dtype=torch.long);pick.scatter_reduce_(1,index,torch.where(chosen,ids,pts.shape[1]),reduce='amin')
   take=z<best;found=pts.gather(1,pick.clamp_max(pts.shape[1]-1)[...,None].expand(-1,-1,3))
   points=torch.where(take[...,None],found,points);best=torch.minimum(best,z)
  contact,cv=actual_contacts174(self.env,[self.nav.robot.body_names[i] for i in self.nav.foot_ids],pose,world_to_sample)
  xy=(pose[:,:2]-self.env.scene.env_origins[:,:2]).tolist()
  segments=torch.tensor([SEGMENT_TYPES.index(route_hint(t,p)[0].segment_type) for t,p in zip(self.nav.index.lookup(range(n)),xy)],device=device)
  h=observed_head_support182(points,torch.isfinite(best),contact,cv,segments,grid,self.nav.pipeline.settings,config=config)
  seen=torch.where(torch.isfinite(h),time_ms[:,None,None],-1);support=Support176(h,torch.isfinite(h),seen)
  state=torch.zeros_like(h,dtype=torch.uint8);frame=OccFrame176(state,torch.full_like(seen,-1),support,epoch,time_ms)
  for ret in streams:frame=merge_camera_frame183(frame,raster183(ret,support,self.p.ocfg,capture_ms=time_ms,epoch=epoch))
  self.last_stats=dict(camera183_rays_per_env=raycount,camera183_free=int((frame.state==1).sum()),camera183_occupied=int((frame.state==2).sum()),camera183_support=int(support.observed.sum()))
  return frame
