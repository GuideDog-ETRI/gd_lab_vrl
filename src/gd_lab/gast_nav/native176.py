"""Native sim OS1/Mid patterns and resolved gdm profile, no temporal175 runtime."""
import ast,json,math,hashlib
from pathlib import Path
import numpy as np,torch,yaml
from .oracle_patterns176 import load_gdm_profile176,rear_cube_pattern176

def parameter(path,key):
 vals=[]
 for n in ast.walk(ast.parse(path.read_text())):
  if isinstance(n,ast.Dict):
   for k,v in zip(n.keys,n.values):
    if isinstance(k,ast.Constant) and k.value==key:vals.append(ast.literal_eval(v))
 if len(vals)!=1:raise ValueError(key)
 return vals[0]

class Native176:
 def __init__(self,device):
  base=Path('/home/user/gd_project');gdm=base/'gdm_docker/workspace/src';launch=gdm/'gdm_planning/launch/rbq.launch.py'
  sensor=base/'sim_docker/_sensor_configuration.yaml';rbq=gdm/'trav_map/trav_map/config/setups/rbq/rbq.yaml'
  meta=Path('/usr/local/lib/python3.11/dist-packages/isaacsim/exts/isaacsim.sensors.rtx/data/lidar_configs/Ouster/OS1/OS1_REV6_128ch10hz1024res.json')
  mid=base/'sim_docker/workspace/scripts/mid360_pattern.npy'
  self.device=device;self.pattern=torch.as_tensor(np.load(mid),device=device)
  effective=dict(ouster_min_range_m=parameter(launch,'sensor_blind_radius'),ouster_max_range_m=parameter(launch,'max_range'),livox_min_range=parameter(launch,'livox_min_range'),livox_max_range=parameter(launch,'livox_max_range'),livox_user_sector_deg=parameter(launch,'livox_user_sector_deg'),livox_vertical_fov_deg=[-7.,52.])
  self.profile=load_gdm_profile176(rbq,sensor,effective_parameters=effective)
  e=json.loads(meta.read_text())['profile']['emitterStates'][0]
  el=torch.tensor(e['elevationDeg'][::2],device=device)*math.pi/180
  azoff=torch.tensor(e['azimuthDeg'][::2],device=device)*math.pi/180
  assert len(el)==64
  az=torch.arange(1024,device=device)[:,None]*2*math.pi/1024+azoff
  el=el[None].expand_as(az);self.os=torch.stack((el.cos()*az.cos(),el.cos()*az.sin(),el.sin()),-1).reshape(-1,3)
  self.rings=torch.arange(64,device=device).expand(1024,-1).flatten()
  az=(torch.arange(2048,device=device)+.5)*2*math.pi/2048
  el=(torch.arange(64,device=device)+.5)*math.pi/64-math.pi/2
  aa,ee=torch.meshgrid(az,el,indexing='ij')
  self.rear=torch.stack((ee.cos()*aa.cos(),ee.cos()*aa.sin(),ee.sin()),-1).reshape(-1,3)
  self.rear=self.rear[self.rear[:,0]<0]  # rear half-space, numerical quadrature only
  from gd_lab.core.camera_geometry import camera_rotation_matrix
  transforms={'base_link':(torch.eye(3,device=device),torch.zeros(3,device=device))}
  def resolve(name):
   if name in transforms:return transforms[name]
   t=self.profile['transforms'][name];r,p=resolve(t['parent']);x,y,z,w=t['quat_xyzw']
   rr=camera_rotation_matrix(torch.tensor([[w,x,y,z]],device=device))[0]
   transforms[name]=(r@rr,p+r@torch.tensor(t['xyz'],device=device));return transforms[name]
  r0,t0=resolve('os_sensor');r1,t1=resolve('livox_frame')
  self.rot=torch.stack((r0,r1,torch.eye(3,device=device)))
  # Native OS1 emitter plane displacement used by sim, not a mount error.
  self.pos=torch.stack((t0+r0[:,2]*.03618,t1,torch.tensor([-.31,0.,0.],device=device)))
  files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (launch,sensor,rbq,meta,mid)}
  self.manifest=dict(files=files,effective=effective,profile=self.profile,rear=dict(azimuth_samples=2048,elevation_samples=64,backward_only=True,origin_base=[-.31,0,0],far_m=6),scan_model='instantaneous same-capture true pose; native phase advances every 100ms',vertical_fov_source='Mid360 native pattern generation contract -7..52 deg')
  self.manifest['sha256']=hashlib.sha256(json.dumps(self.manifest,sort_keys=True).encode()).hexdigest()
 def pattern_at(self,index):
  ids=(torch.arange(20000,device=self.device)+index*20000)%self.pattern.shape[1]
  az=self.pattern[0,ids];el=-self.pattern[1,ids]
  mid=torch.stack((el.cos()*az.cos(),el.cos()*az.sin(),el.sin()),-1)
  dirs=torch.cat((self.os,mid,self.rear));sizes=(len(self.os),len(mid),len(self.rear))
  sources=torch.cat([torch.full((k,),i,device=self.device,dtype=torch.long) for i,k in enumerate(sizes)])
  rings=torch.cat((self.rings,torch.full((sizes[1]+sizes[2],),-1,device=self.device,dtype=torch.long)))
  return dirs,sources,rings,sources
