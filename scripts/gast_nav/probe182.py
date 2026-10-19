"""Diagnostic-only env0 first-return recorder; no sensor/runtime mutation."""
from dataclasses import fields,is_dataclass
from pathlib import Path
import torch

def slice0(x):
 if isinstance(x,torch.Tensor):return x[:1].detach().cpu().clone()
 if is_dataclass(x):return type(x)(**{f.name:slice0(getattr(x,f.name)) for f in fields(x)})
 return x

def install182(out):
 import gd_lab.gast_nav.isaac_occgrid176 as mod
 old_capture=mod.IsaacOccGrid176.capture;old_mount=mod.mounted_rays176;old_hits=mod.oracle_first_hits176;old_raster=mod.rasterize176
 state={}
 def mount(*args,**kw):
  result=old_mount(*args,**kw)
  if state.get('record'):state['pre_filter_rays']=slice0(result[0])
  return result
 def hits(scene,rays,**kw):
  result=old_hits(scene,rays,**kw)
  if state.get('record'):state.update(scene=slice0(scene),raw_hits=slice0(result))
  return result
 def raster(returns,support,cfg,**kw):
  result=old_raster(returns,support,cfg,**kw)
  if state.get('record'):state.update(returns=slice0(returns),support=slice0(support),raw_frame=slice0(result),config=cfg)
  return result
 def capture(self,**kw):
  index=self.capture_index;record=index in (0,10)
  state.clear();state['record']=record
  result=old_capture(self,**kw)
  if record:
   data={k:v for k,v in state.items() if k!='record'}
   data.update(pose=slice0(kw['pose']),root_quat=slice0(self.nav.robot.data.root_quat_w),profile=self.native.profile,settings=self.nav.pipeline.settings,contact=slice0(self.support_debug174['contact']),contact_valid=slice0(self.support_debug174['contact_valid']),native_rot=self.native.rot.cpu(),native_pos=self.native.pos.cpu(),time_ms=slice0(kw['time_ms']))
   terrain=getattr(self.env,'_terrain177',None)
   if terrain is not None:data['terrain']=slice0(terrain)
   torch.save(data,Path(out)/f'head_probe182_{index}.pt')
  state['record']=False
  return result
 mod.mounted_rays176=mount;mod.oracle_first_hits176=hits;mod.rasterize176=raster;mod.IsaacOccGrid176.capture=capture

 # Record lower-camera first hits without changing its returned packet. Extra
 # scene-only raycast is diagnostic ground-truth evaluation, never map input.
 from gd_lab.gast_nav.e1_terrain177 import OracleCamera177
 original_camera=OracleCamera177.capture
 def camera(self,episodes,step):
  packet=original_camera(self,episodes,step)
  if step==100:
   from isaaclab.utils.warp import raycast_mesh
   p={k:torch.as_tensor(v[:1]).cpu().clone() for k,v in packet.items()}
   dirs=packet['dirs'][:1];origins=packet['origins'][:1]
   starts=torch.as_tensor(origins,device=self.base.device,dtype=torch.float32)[:,:,None,None,:].expand_as(torch.as_tensor(dirs,device=self.base.device,dtype=torch.float32))
   _,distance,_,_=raycast_mesh(starts.reshape(1,-1,3),torch.as_tensor(dirs,device=self.base.device,dtype=torch.float32).reshape(1,-1,3),self.model.mesh,max_dist=100.,return_distance=True)
   p['scene_only_distance']=distance.reshape(dirs.shape[:-1]).cpu();p['K']=torch.as_tensor(self.K)
   torch.save(p,Path(out)/'lower_probe182_100.pt')
  return packet
 OracleCamera177.capture=camera
