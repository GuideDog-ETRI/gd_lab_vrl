"""review-approved temporary C envelope. Empirical domain, NOT deployment assurance."""
from pathlib import Path
import torch,math
from .swept174 import FutureSweep174
from .obs_terms import world_to_sample
from .free_evidence167 import _rotation
from gd_lab.core.camera_geometry import camera_rotation_matrix

class DiagnosticEnvelope174:
 def __init__(self,path,device):
  self.models=torch.load(Path(path),map_location=device,weights_only=False)
  self.device=device
  if any(m.get('posture_format')!='collision_corners_sphere_centres' for m in self.models.values()):raise ValueError('obsolete posture metric')
 def build(self,nav,support,grid,*,projection_only=False):
  d=nav.robot.data;pose=nav.pose();n=len(pose);
  if any(list(m['body_names'])!=list(nav.robot.body_names) for m in self.models.values()):raise ValueError('envelope body ordering mismatch')
  xy=grid.xy(device=self.device,dtype=pose.dtype).reshape(-1,2)
  local=world_to_sample(d.body_pos_w,pose);rot=_rotation(-pose[:,3])[:,None]@camera_rotation_matrix(d.body_quat_w)
  from .posture174 import collision_points174
  posture=collision_points174(local,rot,nav.footprint.parts)
  keys=[];matches=[]
  positions=(pose[:,:2]-nav.env.scene.env_origins[:,:2]).detach().cpu().tolist()
  for tile,cmd,p in zip(nav.index.lookup(range(n)),nav.control.commands.commands,positions):
   kind=tile.kind;mode=cmd.requested_mode;value=cmd.turn_rate if mode=='TURN' else cmd.v_cap
   if kind=='nav_stair_landing':
    if mode=='TURN':keys.append(None);continue  # not an approved stair TURN domain
    lower,upper=tile.landing[0][0],tile.landing[1][0]
    steps=[b for b in tile.obstacles if b.kind=='step']
    # Metadata is used only to select an empirical command/posture domain.
    # It never creates sensor free/support evidence.
    start=min((b.center[0]-b.half[0] for b in steps),default=lower)
    end=max((b.center[0]+b.half[0] for b in steps),default=upper)
    if start<=p[0]<=end:kind+='/'+('up' if p[0]<lower else 'down' if p[0]>upper else 'landing')
   key=f'{kind}:{mode}:{value:.1f}'
   keys.append(key if abs(value-round(value,1))<1e-6 and mode in ('NAV','TURN') else None)
  q=max(len(m['centres']) for m in self.models.values());centres=torch.zeros(n,q,3,device=self.device);half=torch.zeros_like(centres)
  required=torch.zeros(n,q,dtype=torch.bool,device=self.device);qualified=torch.zeros(n,dtype=torch.bool,device=self.device)
  foot_required=torch.zeros(n,len(xy),dtype=torch.bool,device=self.device);distance=torch.full((n,),torch.inf,device=self.device);references=torch.zeros(n,dtype=torch.bool,device=self.device);inside_foot=torch.zeros_like(references);domains=0
  for e,key in enumerate(keys):
   model=self.models.get(key)
   if model is None or not model['qualified_geometry']:continue
   error=(model['postures']-posture[e]).norm(dim=-1).amax(-1).amin();distance[e]=error
   qualified[e]=(error<=.05)&(d.root_lin_vel_b[e,:2].norm()<=model['speed_max']+.05)&(d.root_ang_vel_b[e,2].abs()<=model['omega_max']+.05)
   domains+=1
   m=len(model['centres']);c=model['centres'];h=model['half'];low=c-h;high=c+h
   footprint=(xy[None]>=low[:,None,:2]).all(-1)&(xy[None]<=high[:,None,:2]).all(-1)
   feet=torch.tensor(['foot' in name for name in model['part_names']],device=self.device)
   foot_required[e]=footprint[feet].any(0)
   inside_foot[e]=(low[feet,:2]>=xy.amin(0)-grid.res/2).all() & (high[feet,:2]<=xy.amax(0)+grid.res/2).all()
   qualified[e] &= inside_foot[e]
   if projection_only:
    # UPDATE176 projects complete collision paths, including feet below B1.
    # Support is checked cell-by-cell by the occupancy gate, not used to clip
    # away required geometry. Existing 3D deployment proof path is unchanged.
    centres[e,:m]=c;half[e,:m]=h;required[e,:m]=True
    references[e]=True
    continue
   # Contact-space below B1 is governed by observed support, never assumed free.
   ref=torch.where(footprint&torch.isfinite(support[e].flatten())[None],support[e].flatten()[None],torch.inf).amin(-1)
   low=low.clone();high=high.clone();low[:,2]=torch.maximum(low[:,2],ref+.03);high[:,2]=torch.minimum(high[:,2],ref+.80)
   valid=torch.isfinite(ref)&(high>low).all(-1)
   # Do not let unavailable references erase required collision volumes.
   references[e]=torch.isfinite(ref).all() & foot_required[e].any()
   qualified[e] &= references[e]
   centres[e,:m]=torch.where(valid[:,None],(low+high)/2,0.);half[e,:m]=torch.where(valid[:,None],(high-low).clamp_min(0)/2,0.);required[e,:m]=valid
  self.last=dict(qualified=int(qualified.sum()),total=n,domain_available=domains,support_reference_available=int(references.sum()),footprint_inside=int(inside_foot.sum()),matched_posture=int((distance<=.05).sum()),keys=keys,posture_distance_m=distance.detach().cpu().tolist(),temporary_diagnostic_only=True)
  return FutureSweep174(centres,half,required,qualified,xy[None].expand(n,-1,-1),foot_required)
