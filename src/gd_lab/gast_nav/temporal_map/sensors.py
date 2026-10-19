"""Actual sim_docker sensor transforms and full-resolution head ray streams.

No new self mesh. A first-hit callback must include all scene occluders and
return nearest distance plus self-hit. Finite ray depth alone is NOT volume_valid.
"""
from dataclasses import dataclass
from pathlib import Path
import json,math
import torch,numpy as np,yaml

@dataclass(frozen=True)
class RaySample:
 origins: torch.Tensor
 directions: torch.Tensor
 distances: torch.Tensor
 valid: torch.Tensor
 self_hit: torch.Tensor
 source_id: int
 near_m: float
 far_m: float


def rotation(q,device):
 q=torch.tensor(q,device=device,dtype=torch.float32);q=q/q.norm();x,y,z,w=q
 return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))).reshape(3,3)

class HeadSensors:
 def __init__(self,settings,device='cpu'):
  from .memory import validate_settings
  validate_settings(settings)
  self.cfg=settings;self.device=device;self.sim=settings['sensor_sim'];self.data=yaml.safe_load(Path(settings['extrinsics_file']).read_text())
  self.transforms={'base_link':(torch.eye(3,device=device),torch.zeros(3,device=device))}
  def resolve(name):
   if name in self.transforms:return self.transforms[name]
   t=self.data['transforms'][name];r,p=resolve(t['parent']);rr=rotation(t['quat_xyzw'],device);pp=torch.tensor(t['xyz'],device=device)
   self.transforms[name]=(r@rr,p+r@pp);return self.transforms[name]
  for name in self.data['transforms']:resolve(name)
  from .gdm_settings import GdmCoverage
  self.coverage=GdmCoverage(settings)
  self.pattern=torch.as_tensor(np.load(settings['mid360_pattern_file']),device=device)
  if self.pattern.ndim!=2 or self.pattern.shape[0]!=2 or not bool(torch.isfinite(self.pattern).all()):raise ValueError('actual Mid pattern shape/values')
  self.head_r,self.head_p=self.transforms['os_sensor']

 def directions(self,sensor,*,capture_index=0):
  if sensor=='ouster':
   p=self.cfg['ouster_emitter_file']
   if not p:raise FileNotFoundError('exact OS1 emitter metadata required; no uniform-channel fallback')
   em=json.loads(Path(p).read_text())['profile']['emitterStates'][0]
   el=torch.tensor(em['elevationDeg'][::2],device=self.device)*math.pi/180
   azoff=torch.tensor(em['azimuthDeg'][::2],device=self.device)*math.pi/180
   if len(el)!=64:raise ValueError('expected 128-channel source, stride2 ->64')
   lo,hi=self.coverage.rings
   if [lo,hi]!=[-1,-1]:el=el[lo:hi+1];azoff=azoff[lo:hi+1]
   az=torch.arange(1024,device=self.device)*2*math.pi/1024
   az=(az[:,None]+azoff[None]).flatten();el=el[None].expand(1024,-1).flatten()
   d=torch.stack((el.cos()*az.cos(),el.cos()*az.sin(),el.sin()),-1)
   return 'os_sensor',d,.03618,self.coverage.ouster_min_range,1
  if sensor=='mid360':
   # 200k rays/s over each 100ms capture, source 4s non-repeating rosette.
   ids=(torch.arange(20000,device=self.device)+capture_index*20000)%self.pattern.shape[1]
   az=self.pattern[0,ids];el=-self.pattern[1,ids]
   return 'livox_frame',torch.stack((el.cos()*az.cos(),el.cos()*az.sin(),el.sin()),-1),0.,self.coverage.mid_min_range,2
  if sensor=='d455':
   w,h=self.sim['d455_resolution'];f=self.sim['d455_focal_mm']/self.sim['d455_horizontal_aperture_mm']*w
   v,u=torch.meshgrid(torch.arange(h,device=self.device),torch.arange(w,device=self.device),indexing='ij')
   d=torch.stack((torch.ones_like(u),(w/2-u-.5)/f,(h/2-v-.5)/f),-1).float().reshape(-1,3)
   return 'camera_link',d/d.norm(dim=-1,keepdim=True),0.,self.sim.get('d455_min_range_m',.15),4
  raise ValueError('head sensor')

 def capture(self,sensor,first_hit,*,n,capture_index=0,generator=None,screen_door=False):
  """Yield bounded ray chunks. Callback distances are before noise/near filtering.

  Oracle only: no RNG, dropout, mount error or pose error. Callback returns
  (distance,self_hit,measurement_valid) from COMPLETE first-hit inventory.
  Explicit valid + infinite distance means measured no-hit to far. Near/self
  returns remain in raw distances and never expose a farther return.
  Origins/directions are base-link coordinates; caller applies full live pose.
  """
  if screen_door:raise ValueError('screen_door needs full live body-yaw pose: use HeadOracle175B')
  frame,dirs,zoff,near,src=self.directions(sensor,capture_index=capture_index)
  r,t=self.transforms[frame];origin=t+r[:,2]*zoff;direction=dirs@r.T
  # Sensor-frame filters match gdm; no blanket forward-180-degree crop.
  facing=self.coverage.keep_directions(sensor,dirs,screen_door=screen_door);ids=facing.nonzero().flatten()
  far=min(self.sim['max_range_m'],self.coverage.mid_max_range) if sensor=='mid360' else self.sim['max_range_m']
  chunk=self.sim['ray_chunk']
  for k in range(0,len(ids),chunk):
   ix=ids[k:k+chunk];d=direction[ix][None].expand(n,-1,-1);o=origin[None,None].expand(n,len(ix),-1)
   result=first_hit(o,d,far)
   if len(result)!=3:raise ValueError('explicit measurement-valid/complete metadata required')
   dist,self_hit,measured=result
   if dist.shape!=(n,len(ix)) or self_hit.shape!=dist.shape:raise ValueError('first-hit callback shape')
   sh=self_hit.clone();local=dirs[ix];az=torch.atan2(local[:,1],local[:,0])*180/math.pi;el=torch.asin(local[:,2].clamp(-1,1))*180/math.pi
   for sec in self.coverage.self_hits.get(frame,[]):
    da=(az-sec['bearing_deg']+180)%360-180
    sector=(da.abs()<=sec['half_angle_deg'])&(el>=sec['elev_min_deg'])&(el<=sec['elev_max_deg'])
    sh |= sector[None]&(dist<=sec['max_range'])
   if measured.shape!=dist.shape or measured.dtype!=torch.bool or self_hit.dtype!=torch.bool:raise ValueError('measurement metadata')
   if bool(torch.isnan(dist).any()|(dist<0).any()):raise ValueError('first hit distance')
   keep=measured&(dist>=near)&~sh
   # Far clipping means a complete, valid no-hit measurement; raw nearest
   # near/self hits are retained even though they cannot carve free.
   distance=torch.where(dist<=far,dist,torch.inf)
   yield RaySample(o,d,distance,keep,sh&torch.isfinite(distance),src,near,far)
