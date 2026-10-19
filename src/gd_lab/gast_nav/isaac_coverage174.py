"""UPDATE174B scene-only coverage adapter. D4xx/support/robot mesh inputs ignored."""
from pathlib import Path
from dataclasses import replace
import time,torch,yaml
from .coverage174 import CoverageConfig174,sample_mounts174,scene_in_sample174,measure_coverage174
from .producer168 import SensorPrimitives
from .temporal169 import TemporalEvidence169,transform_returns169
from .support174 import actual_contacts174,historical_support174
from .map_types import WIDE_GRID,NEAR_GRID
from .near_clearance import extended_near_grid,clearance_from_state
from .obstacle_map import ray_evidence
from .ray_geometry import resolve_first_hits
from .free_evidence167 import limit_by_current_returns

class IsaacCoverage174:
 def __init__(self,nav):
  self.nav=nav;self.env=nav.env;env=self.env;root=Path(__file__).resolve().parents[3]
  self.cfg=CoverageConfig174.load(root/'configs/gast_nav/coverage174.yaml')
  cfg=yaml.safe_load((root/'configs/gast_nav/temporal169_training_c.yaml').read_text())
  self.temporal=TemporalEvidence169(env.num_envs,env.device,cfg)
  self.temporal.settings=replace(self.temporal.settings,same_sample_pose_cancellation=True)
  for h in self.temporal.histories:h.settings=self.temporal.settings
  self.generator=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed)+174)
  self.mounts=sample_mounts174(env.num_envs,self.cfg,device=env.device,generator=self.generator)
  self.motion_time=None;self.last_time=None;self.stats={}
  from .isaac_closure169 import audit_closed_scene169
  self.audit169=audit_closed_scene169(env)
  gen=env.scene.terrain.cfg.terrain_generator;ids=nav.tile_ids.detach().cpu().tolist()
  if len(set(ids))!=env.num_envs:raise ValueError('C174 requires isolated unique tiles')
  solidsets=[gen.sensor_solids169[(i//gen.num_cols,i%gen.num_cols)] for i in ids]
  maxp=max(map(len,solidsets));n=env.num_envs;dev=env.device
  centres=torch.zeros(n,maxp,3,device=dev);half=torch.ones_like(centres);yaw=torch.zeros(n,maxp,device=dev);active=torch.zeros(n,maxp,device=dev,dtype=torch.bool)
  for e,solids in enumerate(solidsets):
   for j,b in enumerate(solids):centres[e,j]=centres.new_tensor(b.center);half[e,j]=half.new_tensor(b.half);yaw[e,j]=b.yaw;active[e,j]=True
  centres+=env.scene.env_origins[:,None];c,s=yaw.cos(),yaw.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
  rot=torch.stack((c,-s,z,s,c,z,z,z,one),-1).reshape(n,maxp,3,3)
  self.scene=SensorPrimitives(centres,rot,half,torch.zeros_like(active,dtype=torch.long),active,torch.ones(n,device=dev,dtype=torch.bool))
  self.envelope=None
  if nav.cfg.get('diagnostic_envelope174'):
   from .envelope174 import DiagnosticEnvelope174
   self.envelope=DiagnosticEnvelope174(nav.cfg['diagnostic_envelope174'],dev)
  self.swept_certified=torch.zeros(n,device=dev,dtype=torch.bool)
  self.swept_status='missing validated current-posture stopping trajectory; no 2D/full-band fallback'

 def reset(self,ids):
  self.temporal.reset(ids);self.mounts[ids]=sample_mounts174(len(ids),self.cfg,device=self.env.device,generator=self.generator)

 def sample_motion(self):
  now=self.env._sim_step_counter*self.env.physics_dt
  if self.motion_time is not None and now<=self.motion_time:return
  dt=0. if self.motion_time is None else now-self.motion_time;d=self.nav.robot.data
  self.temporal.observe(self.nav.pose(),d.root_lin_vel_w.norm(dim=-1),d.root_ang_vel_w[:,2],dt);self.motion_time=now

 def capture(self,*,rays,pose,time_ms,wide_support,fine_support,epoch,capsules,config):
  # Those legacy observation arguments are intentionally never consumed here.
  from .obs_terms import observed_support,world_to_sample
  from .command_generator import route_hint
  from . import schema as S
  start=time.perf_counter();n=len(pose);env=self.env;nav=self.nav
  local=(pose[:,:2]-env.scene.env_origins[:,:2]).abs()
  # All queries (6m radius plus jitter/mount) stay in their own 24m tile.
  complete=(local.max(-1).values+self.cfg.query_radius_m+.6<12.)
  # Foreign robots are not scene-only sensor primitives. Every robot's full
  # collision extent must remain inside its distinct tile, or all queries close.
  body_local=nav.robot.data.body_pos_w-env.scene.env_origins[:,None]
  complete &= (body_local[...,:2].abs().amax((1,2))+.6<12.).all()
  if bool(nav.cylinder.enabled.any()):raise ValueError('static C only; dynamic collider inventory required before enabling cylinders')
  scene=replace(self.scene,complete=complete)
  ids=time_ms+epoch*1000000000
  pkt=measure_coverage174(scene_in_sample174(scene,pose),self.cfg,scene_only=True,scene_capture_id=ids,pose_capture_id=ids,mount_deltas=self.mounts,generator=self.generator,backend='triton')
  self.last_packet174=pkt;self.last_measured=pkt.free
  contact,contact_valid=actual_contacts174(env,[nav.robot.body_names[i] for i in nav.foot_ids],pose,world_to_sample)
  xy=(pose[:,:2]-env.scene.env_origins[:,:2]).tolist()
  segments=torch.tensor([S.SEGMENT_TYPES.index(route_hint(t,p)[0].segment_type) for t,p in zip(nav.index.lookup(range(n)),xy)],device=env.device)
  hits=pkt.rays.origins+pkt.rays.directions*torch.nan_to_num(pkt.rays.distances,posinf=0.)[...,None]
  self.support_debug174=dict(hits=hits,valid=pkt.valid_hit,contact=contact,contact_valid=contact_valid,segments=segments,settings=nav.pipeline.settings)
  supports=tuple(observed_support(hits,pkt.valid_hit,contact,contact_valid,segments,g,nav.pipeline.settings,config=config) for g in (WIDE_GRID,extended_near_grid()[0]))
  self.temporal.push(pkt.free.frusta,pose,supports,time_ms,epoch,dynamic_enabled=nav.cylinder.enabled)
  self.last_resolved=resolve_first_hits(pkt.rays,config=config);self.last_pose=pose.clone();self.last_epoch=epoch.clone();self.last_time=time_ms.clone()
  torch.cuda.synchronize()
  self.stats=dict(complete_envs=int(complete.sum()),envs=n,valid_blocks=int(pkt.free.volume_valid.sum()),blocks=pkt.free.volume_valid.numel(),producer_wall_s=time.perf_counter()-start,d4xx_used=False,actual_contact_points=int(contact_valid.sum()))
  return self.query(pose=pose,time_ms=time_ms,epoch=epoch,config=config)

 def query(self,*,pose,time_ms,epoch,config):
  start=time.perf_counter();results=[];supports=[];measurements=[];support_ages=[]
  valid=(self.last_epoch==epoch)&((time_ms-self.last_time)<=round(config.ttl_s*1000))
  resolved=transform_returns169(self.last_resolved,self.last_pose,pose,valid)
  for i,grid in enumerate((WIDE_GRID,extended_near_grid()[0])):
   m=self.temporal.measurement(i,pose,time_ms,epoch)
   support,age=historical_support174(m,grid,config)
   state,_,ev_age=ray_evidence(resolved,support,grid=grid,config=config,measurement=m,certified=True,return_age=True)
   results.append((state,ev_age));supports.append(support);measurements.append(m);support_ages.append(age)
  self.last_query169=dict(resolved=resolved,supports=supports,measurements=measurements,support_ages=support_ages)
  from .swept174 import prove_future174
  if self.envelope is not None:
   self.nav.future_sweep174=self.envelope.build(self.nav,supports[0],WIDE_GRID)
   self.stats.update(envelope_qualified=self.envelope.last['qualified'],posture_matched=self.envelope.last['matched_posture'],domain_available=self.envelope.last['domain_available'],support_reference_available=self.envelope.last['support_reference_available'],footprint_inside=self.envelope.last['footprint_inside'])
  if self.nav.cfg.get('supervisor176',{}).get('certification_mode')=='occupancy_grid':
   # UPDATE176 uses its own measured occupancy provider and 2D gate in env.py.
   # Preserve this deployment-validation proof implementation, but do not run it.
   self.swept_certified=torch.zeros_like(self.swept_certified)
   self.swept_diagnostics={'reason':'VOLUME_PROOF_DISABLED_UPDATE176'}
  else:
   self.swept_certified,self.swept_diagnostics=prove_future174(getattr(self.nav,'future_sweep174',None),measurements[0],resolved,supports[0],WIDE_GRID,config)
  self.swept_status=self.swept_diagnostics['reason']
  pad=extended_near_grid()[1];sl=(slice(None),slice(pad,pad+NEAR_GRID.ny),slice(pad,pad+NEAR_GRID.nx))
  self.stats.update(wide_support_cells=int(torch.isfinite(supports[0]).sum()))
  self.wide_support_known=torch.isfinite(supports[0]);self.near_support_known=torch.isfinite(supports[1][sl])
  self.safety_clearance=clearance_from_state(results[1][0],grid=extended_near_grid()[0],config=config)[...,pad:pad+NEAR_GRID.ny,pad:pad+NEAR_GRID.nx]
  torch.cuda.synchronize();self.stats.update(proof_wall_s=time.perf_counter()-start,history_captures=len(self.temporal.anchors),bound_violation_samples=int(self.temporal.violations.sum()),query_time_ms=int(time_ms.max()))
  return results[0],tuple(v[...,pad:pad+NEAR_GRID.ny,pad:pad+NEAR_GRID.nx] for v in results[1])
