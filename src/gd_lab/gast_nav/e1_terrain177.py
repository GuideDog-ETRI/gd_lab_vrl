"""Diagnostic-only E1 camera oracle -> unchanged camfilter/v25or -> actor + read-only tap."""
import os,sys
from pathlib import Path
import numpy as np
import torch
from isaaclab.managers import ManagerTermBase
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/rl_replay'))
# Fixed before mapper/camera imports. Native map settings/history are unchanged.
os.environ.update(U130_DUAL='2',U130_DDEEP='.7',U136_CAM_FILTER='1')
from lidar126 import Maps
from cam130 import CamSensor,qmul,CAM_H,CAM_W
from heightmap116 import rotation
from camfilter136 import reject_outliers
from rayir_e1 import CellIr,RayIrHistory
from .terrain_tap177 import tap_v25or177
import structure130

class OracleCamera177(CamSensor):
 def capture(self,episodes,step):
  from isaaclab.utils.warp import raycast_mesh
  d=self.robot.data;n=len(episodes);p=d.body_pos_w[:,self.trunk].cpu().numpy();q=d.body_quat_w[:,self.trunk].cpu().numpy()
  positions=p[:,None]+np.einsum('nij,cj->nci',rotation(q),self.P)
  quats=qmul(np.repeat(q[:,None],4,1),np.broadcast_to(self.Q,(n,4,4)))
  v,u=np.mgrid[:CAM_H,:CAM_W];local=np.stack(((u-self.K[0,2])/self.K[0,0],-(v-self.K[1,2])/self.K[1,1],-np.ones_like(u)),-1)
  rays=np.einsum('hwj,ncij->nchwi',local,rotation(quats));length=np.linalg.norm(rays,axis=-1);unit=rays/length[...,None]
  starts=np.broadcast_to(positions[:,:,None,None],unit.shape).copy()
  _,dist,_,_=raycast_mesh(torch.as_tensor(starts.reshape(1,-1,3),device=self.base.device,dtype=torch.float32),torch.as_tensor(unit.reshape(1,-1,3),device=self.base.device,dtype=torch.float32),self.model.mesh,max_dist=100.,return_distance=True)
  clean=dist.cpu().numpy().reshape(length.shape)
  leg,_=self.legs.distances(starts.reshape(n,-1,3),unit.reshape(n,-1,3));leg=leg.reshape(length.shape)
  blocked=np.isfinite(leg)&(leg<clean);ranges=np.minimum(clean,leg)
  depth=ranges/length;valid=np.isfinite(ranges)&(depth>=.15)&(depth<5.)
  if getattr(self.base.cfg,'nav_settings',{}).get('front_depth183',{}).get('enabled',False):
   # Read-only supervisor tap BEFORE camfilter/mapper; keep raw first-hit ranges,
   # optical lengths and explicit no-hit validity. Rear lower cameras excluded.
   self.raw_front183=dict(origins=positions[:,:2].copy(),dirs=unit[:,:2].copy(),
    ranges=ranges[:,:2].copy(),pixel_length=length[:,:2].copy(),
    measured=(~np.isnan(clean[:,:2])&~np.isnan(leg[:,:2])).copy(),
    self_hit=blocked[:,:2].copy(),near_z=.15,far_z=5.,
    capture_ms=np.full(n,round(step*self.base.step_dt*1000),np.int64),
    body_pos=d.root_pos_w.cpu().numpy().copy(),body_quat=d.root_quat_w.cpu().numpy().copy(),
    camera_indices=(0,1),resolution=(CAM_H,CAM_W))
  valid=reject_outliers(depth,valid);ranges=np.where(valid,ranges,10.)
  self.add('beams',valid.size);self.add('valid_returns',valid.sum());self.add('leg_occluded',blocked.sum())
  return dict(origins=positions,dirs=unit,ranges=ranges,valid=valid,body_pos=d.root_pos_w.cpu().numpy().copy(),body_quat=d.root_quat_w.cpu().numpy().copy(),self_hit=blocked&valid)

class E1Terrain177(ManagerTermBase):
 def __init__(self,cfg,env):
  super().__init__(cfg,env);self.env=env;self.maps=Maps(env.num_envs,'L0S');self.sensor=OracleCamera177(env,None)
  self.history=RayIrHistory(env);self.ir=CellIr(env,env.cfg.seed);self.last=-1;self.capture=-100;self.cached=None
  self.epochs=np.zeros(env.num_envs,np.int64);env._e1terrain177=self;env._gast_terrain_term=self
  env._oracle177_profile=dict(depth='camera GT first-hit optical depth',gap_fill=False,depth_noise=False,dropout=False,pose_error=False,leg_noise=False,near_m=.15,far_m=5.,resolution=[CAM_H,CAM_W],cameras='vendor_new x4',filter='camfilter136',mapper='L0S h0d dual-OR DD=.7',ir='existing E1 CellIr unchanged; actor IR proxy noise remains',capture_period_s=.1)
 def reset(self,env_ids=None):
  ids=np.arange(self.env.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu().numpy()
  self.maps.reset(ids);self.epochs[ids]+=1;self.history.reset(env_ids);self.ir.reset(ids);self.last=-1;self.capture=-100
 def __call__(self,env):
  step=int(env.common_step_counter)
  if self.last==step and self.cached is not None:return self.cached.clone()
  stamp=step*env.step_dt
  if step-self.capture>=10:
   packet=self.sensor.capture(self.epochs,step);self.maps.integrate(range(env.num_envs),packet,stamp,stamp);self.capture=step
  from gd_lab.gast.observations import pose_xyyaw
  pose=pose_xyyaw(env).cpu().numpy();scanner=env.scene['height_scanner'].data.pos_w[:,2].cpu().numpy()
  packed=self.maps.sample(pose,scanner,stamp)
  p4=np.column_stack((pose[:,:2],env.scene['robot'].data.root_pos_w[:,2].cpu().numpy(),pose[:,2]))
  epochs=env.nav.pipeline.mapper.epoch.cpu().numpy() if getattr(env,'nav',None) is not None else self.epochs
  env._terrain177=tap_v25or177(self.maps,packed,p4,scanner,sample_ms=np.full(env.num_envs,round(stamp*1000),np.int64),epoch=epochs.astype(np.int64),structure_rules=structure130.S,device=env.device)
  self.cached=torch.as_tensor(packed,device=env.device);full=torch.cat((self.cached,self.ir(env,self.cached[:,187:]>0)),-1)
  env._gast_history=self.history.update(env,full);self.last=step
  return self.cached.clone()
