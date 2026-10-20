"""Isaac environment adapter. Policy selection happens before action execution."""
import copy
import os
import math
from pathlib import Path
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_euler_xyz
from . import schema as S
from .scene_adapter import bind_scene,EpisodeCylinder
from .scene_geometry import point_box_distance,floor_height
from .feasibility import passage_feasible,turn_feasible
from .scene_types import project_route
from .command_generator import route_hint
from .route_obs179 import route_features179
from .control_runtime import ControlRuntime,zero_legacy_command_history
from .landing_fsm import LandingObservation
from .supervisor import HealthInput,HoldReasonMetrics,required_observed,required_band_indices
from .person_contact import PersonContactTracker,IsaacResetOverlapQuery,reset_overlap_mask
from .train_inertia import TrainInertia,ExternalForceAccumulator
from .obs_terms import (ObservationPipeline,static_sensor_rays,world_to_sample,capsules_to_sample,
                        randomize_sensor_pattern,critic_privileged,invalidate_held_rows)
from .map_types import DynamicCapsules,NEAR_GRID,WIDE_GRID
from .rewards import reward_terms,legacy_reward_masks

class NavState:
    def __init__(self,env):
        self.env=env;self.robot=env.scene['robot'];self.cfg=env.cfg.nav_settings
        self.stage=env.cfg.nav_stage_settings;n=env.num_envs;dev=env.device
        from .scene_config import load_scene_config
        self.scene_settings=load_scene_config()
        runtime=yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/gast_nav/runtime.yaml').read_text())
        # Explicit simulation-only cap; deployment runtime.yaml remains unverified.
        runtime['command']['omega_max_verified']=self.cfg['commands']['simulation_turn_cap_radps']
        if 'curriculum_C_limits' in self.cfg:
            from .curriculum167 import easy_command_settings
            runtime['command']=easy_command_settings(runtime['command'],self.cfg['curriculum_C_limits'])
        self.index=bind_scene(env.scene.terrain)
        from .batched_oracle import BatchedOracle
        self.oracle=BatchedOracle(self.index.tiles,self.scene_settings,device=dev)
        self.tile_ids=torch.tensor(self.index.env_tile_ids,device=dev)
        self.tile_spawn=torch.tensor([t.turn_spot[0] if t.turn_spot else t.route_polyline[0] for t in self.index.tiles],device=dev)
        self.tile_axis=torch.tensor([project_route(t,t.route_polyline[0])[2] for t in self.index.tiles],device=dev)
        self.control=ControlRuntime(self.index.lookup(range(n)),self.cfg,runtime,seed=self.cfg['seed'])
        self.inertia=TrainInertia(n,self.cfg['inertia'],device=dev,seed=self.cfg['seed'])
        self.forces=ExternalForceAccumulator(n,self.robot.num_bodies,device=dev)
        self.pipeline=ObservationPipeline(n,self.cfg['sensors'],device=dev)
        self.pipeline.certified_no_producer=self.cfg.get('free_evidence_backend')=='measured167_unqualified'
        if self.pipeline.certified_no_producer and any(self.cfg['observability'][k] for k in ('r1_static_history','r2_unknown_components')):
            raise ValueError('measured167 certificate cannot be bypassed by legacy R1/R2 relaxations')
        from .body_projection import CollisionFootprint
        self.footprint=CollisionFootprint(self.robot.body_names,dev)
        from .certification165 import CertificateHistory
        ttl=self.cfg['observability']['certification_ttl_s']
        if self.cfg['observability']['r1_static_history'] and self.stage['moving_cylinder']:
            raise ValueError('UPDATE165 R1 diagnostic requires static scene configuration; no observed-motion filter yet')
        self.cert_near_history=CertificateHistory(NEAR_GRID,ttl)
        self.cert_wide_history=CertificateHistory(WIDE_GRID,ttl)
        self.cert_near=self.cert_wide=None
        self.envelope_masks=None
        if self.cfg.get('swept_envelope_file'):
            if self.cfg.get('free_evidence_backend') not in ('pending167','measured167_unqualified','producer168'):
                raise ValueError('UPDATE167 empirical envelope integration is diagnostic-only until free-model validation')
            from .envelope167 import resample_required
            artifacts=torch.load(self.cfg['swept_envelope_file'],map_location='cpu',weights_only=True)
            self.envelope_masks={}
            for key,entry in artifacts.items():
                near_mask,_=resample_required(entry,NEAR_GRID,device=dev)
                wide_mask,contained=resample_required(entry,WIDE_GRID,device=dev)
                self.envelope_masks[key]=(near_mask,wide_mask,bool(contained))
        self.lower_depth=None
        if self.cfg['sensors']['lower_depth']['enabled'] and self.cfg.get('free_evidence_backend')!='coverage174':
            from .depth_camera_rays import LowerDepthRays
            self.lower_depth=LowerDepthRays(env,self.cfg['sensors']['lower_depth'])
        self.cylinder=EpisodeCylinder(self.index,stage='B',device=dev,seed=self.cfg['seed'],env_origins=env.scene.env_origins)
        self.person_contacts=PersonContactTracker(n,self.cfg['person_contact']['force_threshold_n'],device=dev)
        import omni.usd
        self.reset_overlap_query=IsaacResetOverlapQuery(omni.usd.get_context().get_stage())
        expected=set(self.cfg['person_contact']['robot_links'])
        if expected!=set(self.robot.body_names):
            raise RuntimeError(f'person-contact link coverage mismatch: configured={expected}, actual={set(self.robot.body_names)}')
        self.rng=torch.Generator(device=dev).manual_seed(self.cfg['seed'])
        self.controller=torch.zeros(n,dtype=torch.long,device=dev)
        from .stair_end184 import StairEnd184
        self.stair_end184=StairEnd184(n,dev,yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/gast_nav/shadow184.yaml').read_text()))
        self.stair_tick184=None
        self.velocity=torch.zeros(n,3,device=dev);self.features=torch.zeros(n,16,device=dev)
        self.mode=torch.zeros(n,dtype=torch.long,device=dev);self.phase=self.mode.clone()
        self.go=torch.zeros(n,dtype=torch.bool,device=dev);self.legacy=torch.ones_like(self.go)
        self.stall_enabled=torch.ones_like(self.go);self.feasible=torch.ones_like(self.go);self.observed=torch.zeros_like(self.go)
        self.actor_observable=self.observed.clone();self.supervisor_certified=self.observed.clone();self.actor_loss_valid=self.observed.clone()
        self.clearance=torch.ones(n,device=dev);self.offset=torch.zeros(n,device=dev);self.axis=self.offset.clone()
        self.unknown_time=self.offset.clone();self.positive_yaw=self.offset.clone();self.drift=self.offset.clone();self.turn_start=torch.zeros(n,2,device=dev)
        self.entry_previous=None;self.entry_valid=torch.zeros(n,dtype=torch.bool,device=dev)
        self.entered_unknown=torch.zeros_like(self.entry_valid);self.support_stable=torch.zeros_like(self.entry_valid)
        # Online Top-5 rate counters (summed per iteration by the runner, then cleared).
        self.top5_counts=torch.zeros(10,device=dev,dtype=torch.float64)
        self.reasons=['']*n;self.last_capture=-math.inf;self.wide=self.near=None;self.reward_log={}
        self.force_fault=False;self.fixed_policy=None
        self.reason_metrics_enabled=os.environ.get('GAST_NAV_HOLD_METRICS')=='1'
        self.decision_rows=[]
        self.hold_reason_metrics=HoldReasonMetrics()
        self.foot_ids=[self.robot.body_names.index(f'{leg}_foot') for leg in ('FL','FR','RL','RR')]
        sensor=env.scene['contact_forces'];self.contact_ids=[sensor.body_names.index(self.robot.body_names[i]) for i in self.foot_ids]
        if self.cfg.get('free_evidence_backend')=='coverage174':
            from .isaac_coverage174 import IsaacCoverage174
            if self.cfg.get('supervisor176',{}).get('certification_mode')=='occupancy_grid':
                from .isaac_occgrid176 import IsaacOccGrid176
                if self.cfg.get('front_depth183',{}).get('enabled',False):
                    from .isaac_front183 import IsaacFront183
                    self.pipeline.certified_producer=IsaacFront183(self)
                else:self.pipeline.certified_producer=IsaacOccGrid176(self)
            else:self.pipeline.certified_producer=IsaacCoverage174(self)
        if self.cfg.get('free_evidence_backend')=='producer168':
            if any(self.cfg['observability'][k] for k in ('r1_static_history','r2_unknown_components')):
                raise ValueError('producer168 cannot use legacy unknown relaxations')
            from .isaac_producer168 import IsaacProducer168
            self.pipeline.certified_producer=IsaacProducer168(self)

    def pose(self):
        d=self.robot.data
        return torch.cat((d.root_pos_w,d.heading_w[:,None]),-1)

    def reset(self,ids):
        env=self.env;ids=torch.as_tensor(ids,device=env.device,dtype=torch.long)
        tm=getattr(env,'termination_manager',None)
        if tm is not None and hasattr(self,'top5_counts'):
            ended=tm.dones[ids];was_legacy=self.legacy[ids]
            # [8] LEGACY episodes ended, [9] of which by time-out (no fall / contact termination).
            self.top5_counts[8]+=(ended&was_legacy).sum();self.top5_counts[9]+=(ended&was_legacy&tm.time_outs[ids]).sum()
        self.entry_valid[ids]=False
        bind_scene(env.scene.terrain,self.index,ids)
        self.tile_ids=torch.tensor(self.index.env_tile_ids,device=env.device)
        tiles=self.index.lookup(ids)
        self.legacy[ids]=torch.tensor([t.kind=='legacy' for t in tiles],device=env.device)
        # Keep env origins at tile centres; only robot spawn uses route start.
        nav_ids=ids[~self.legacy[ids]]
        if len(nav_ids):
            state=self.robot.data.root_state_w[nav_ids].clone()
            state[:,:2]=env.scene.env_origins[nav_ids,:2]+self.tile_spawn[self.tile_ids[nav_ids]]
            state[:,2]=env.scene.env_origins[nav_ids,2]+self.robot.data.default_root_state[nav_ids,2]
            yaw=self.tile_axis[self.tile_ids[nav_ids]];zero=torch.zeros_like(yaw)
            state[:,3:7]=quat_from_euler_xyz(zero,zero,yaw);state[:,7:]=0
            self.robot.write_root_state_to_sim(state,nav_ids)
        now=round(env.common_step_counter*env.step_dt*1000)
        self.control.reset(ids.tolist(),now_ms=now,yaws=self.robot.data.heading_w[ids].tolist(),tiles=tiles)
        enabled=torch.tensor([t.kind=='nav_turn_room' or (t.kind in ('nav_corridor','nav_door') and
                  self.control.commands.rng[int(i)].random()<self.cfg['inertia']['optional_scene_probability']) for i,t in zip(ids,tiles)],device=env.device)
        self.inertia.reset(ids,enabled & self.cfg['inertia']['enabled'],scale=self.stage['inertia_scale'])
        self.forces.reset(ids);self.pipeline.reset(ids)
        self.cert_near_history.reset(ids);self.cert_wide_history.reset(ids)
        if self.cert_near is not None:self.cert_near[ids]=False;self.cert_wide[ids]=False
        self.person_contacts.reset(ids)
        self.stair_end184.reset(ids)
        self.controller[ids]=0;self.velocity[ids]=0;self.unknown_time[ids]=0;self.positive_yaw[ids]=0
        self.turn_start[ids]=self.robot.data.root_pos_w[ids,:2];self.drift[ids]=0
        randomize_sensor_pattern(env.scene['nav_obstacles'],ids,self.cfg['sensors'],generator=self.rng)
        self.cylinder.env_origins.copy_(env.scene.env_origins)
        self.cylinder.reset(ids,stage='C' if self.stage['moving_cylinder'] else 'B',robot_positions=self.robot.data.root_pos_w)
        self.sync_collider_geometry(ids)
        # Tensor reset poses must be reflected in articulated link kinematics before querying.
        env.sim.physics_sim_view.update_articulations_kinematic()
        initial=reset_overlap_mask(self.cylinder,ids,self.reset_overlap_query)
        self.person_contacts.mark_invalid(ids[initial],'INITIAL_OVERLAP')
        # Reset rows fail closed until the next global 10Hz capture. Forcing
        # a whole-batch capture for every individual reset scales to 100Hz.
        if self.wide is not None:
            self.wide,self.near=invalidate_held_rows(self.wide,self.near,ids)

    def sync_collider_geometry(self,ids):
        # Physical primitive radii/height match the sampled observation capsule.
        from pxr import UsdGeom
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        for i in ids.tolist():
            path=f'/World/envs/env_{i}/NavCylinder/geometry/mesh'
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid():raise RuntimeError(f'cylinder geometry missing: {path}')
            shape=UsdGeom.Cylinder(prim)
            shape.GetRadiusAttr().Set(float(self.cylinder.radius[i]))
            shape.GetHeightAttr().Set(float(self.cylinder.height[i]))
        self.cylinder.write_colliders(self.env.scene['nav_cylinder'])

    def physics_step(self):
        if self.pipeline.certified_producer is not None:self.pipeline.certified_producer.sample_motion()
        self.person_contacts.begin_action()
        a=self.inertia.step(self.env.physics_dt)
        self.forces.apply(self.robot,a)
        self.cylinder.update(self.env.physics_dt,a[:,:2])
        self.cylinder.step(self.env.physics_dt,self.robot.data.root_pos_w,self.robot.data.root_lin_vel_w)
        self.cylinder.write_colliders(self.env.scene['nav_cylinder'])

    def capture(self,now):
        env=self.env;pose=self.pose()
        if now-self.last_capture+1e-9 >= self.cfg['capture_period_s']:
            from gd_lab.teachers.bivt.raycast_terrain import leg_capsules
            starts,ends,radii=leg_capsules(self.robot)
            self_caps=DynamicCapsules(starts,ends,radii[None].expand(env.num_envs,-1))
            rays=static_sensor_rays(env.scene['nav_obstacles'],pose,self_capsules=self_caps)
            trace=self.cfg.get('diagnostics167',False)
            if trace:unoccluded=static_sensor_rays(env.scene['nav_obstacles'],pose)
            if self.lower_depth is not None:
                from .map_types import RayBatch
                depth=self.lower_depth.capture(pose)
                rays=RayBatch(*(torch.cat((getattr(rays,k),getattr(depth,k)),1) for k in ('origins','directions','distances','self_hit')))
                if trace:
                    depth_raw=self.lower_depth.unoccluded_rays
                    unoccluded=RayBatch(*(torch.cat((getattr(unoccluded,k),getattr(depth_raw,k)),1) for k in ('origins','directions','distances','self_hit')))
            if trace:self.diagnostic_rays=(rays,unoccluded)
            points=world_to_sample(env.scene['height_scanner'].data.ray_hits_w,pose)
            valid=torch.isfinite(points).all(-1)
            contacts=world_to_sample(self.robot.data.body_pos_w[:,self.foot_ids],pose)
            forces=env.scene['contact_forces'].data.net_forces_w[:,self.contact_ids]
            contact_valid=forces.norm(dim=-1)>self.cfg['sensors']['contact_threshold_n']
            if self.cfg.get('free_evidence_backend')=='coverage174':
                from .support174 import actual_contacts174
                contacts,contact_valid=actual_contacts174(env,[self.robot.body_names[i] for i in self.foot_ids],pose,world_to_sample)
            xy=(self.robot.data.root_pos_w-env.scene.env_origins)[:,:2].tolist()
            segments=torch.tensor([S.SEGMENT_TYPES.index(route_hint(t,p)[0].segment_type)
                                  for t,p in zip(self.index.lookup(range(env.num_envs)),xy)],device=env.device)
            self.wide,self.near=self.pipeline.capture(rays=rays,terrain_points=points,terrain_valid=valid,
                 contact_points=contacts,contact_valid=contact_valid,segment=segments,pose=pose,
                 time_ms=torch.full((env.num_envs,),round(now*1000),device=env.device,dtype=torch.long),
                 capsules=(capsules_to_sample(self.cylinder.capsules(),pose) if self.cylinder.capsules() is not None else None))
            if self.cfg['observability']['r1_static_history']:
                from .support import decode_support
                required=self.cfg['observability']['required_bands'];frame=self.pipeline.mapper.history[0]
                near_known=required_observed(self.pipeline.near_state,self.pipeline.near_support_known,required)
                height,valid=decode_support(frame.support)
                wide_known=required_observed(frame.state,valid,required)
                eligible=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)  # static experiment configuration, never GT object presence
                self.cert_near=self.cert_near_history.update(near_known,self.pipeline.near_support_height,pose,now,eligible)
                self.cert_wide=self.cert_wide_history.update(wide_known,frame.support_world_z-pose[:,2,None,None],pose,now,eligible,
                    fresh=wide_known&(frame.age_s[:,required_band_indices(required)]==0).all(1))
            self.last_capture=now
        # UPDATE162: hold the capture-aligned inputs between 10Hz captures.
        # Sensor age/health still advances each policy tick; reset forces capture.

    def prepare(self,obs):
        env=self.env;d=self.robot.data;now=env.common_step_counter*env.step_dt
        producer=self.pipeline.certified_producer
        if producer is not None:producer.sample_motion()
        self.capture(now)
        if producer is not None and producer.temporal is not None and now-self.last_capture>1e-9:
            times=torch.full((env.num_envs,),round(now*1000),device=env.device,dtype=torch.long)
            wide,fine=producer.query(pose=self.pose(),time_ms=times,epoch=self.pipeline.mapper.epoch,config=self.pipeline.config)
            self.pipeline.cert_wide_state,self.pipeline.cert_wide_age=wide
            self.pipeline.cert_near_state,self.pipeline.cert_near_age=fine
        tiles=self.index.lookup(range(env.num_envs))
        contacts=env.scene['contact_forces'].data.net_forces_w[:,self.contact_ids].norm(dim=-1)>self.cfg['sensors']['contact_threshold_n']
        from .support import decode_support
        height,known=decode_support(self.pipeline.mapper.history[0].support)
        centre=height[:,19:24,22:27].flatten(1);valid=known[:,19:24,22:27].flatten(1)
        front=height[:,19:24,27:33].flatten(1);front_valid=known[:,19:24,27:33].flatten(1)
        flat=valid.all(-1)&(centre.amax(-1)-centre.amin(-1)<self.cfg['landing_sensor']['flat_tolerance_m'])
        median=torch.where(valid,centre,torch.nan).nanmedian(-1).values
        descent=front_valid.any(-1)&valid.any(-1)&(front.masked_fill(~front_valid,torch.inf).amin(-1)<median-self.cfg['landing_sensor']['descent_drop_m'])
        slipping=((self.robot.data.body_lin_vel_w[:,self.foot_ids,:2].norm(dim=-1)>self.cfg['landing_sensor']['slip_speed_mps'])&contacts).any(-1)
        gridxy=NEAR_GRID.xy(device=env.device,dtype=torch.float32);widexy=WIDE_GRID.xy(device=env.device,dtype=torch.float32)
        margins=torch.tensor([c.c.margin for c in self.control.commands.commands],device=env.device)
        turns=torch.tensor([c.requested_mode=='TURN' for c in self.control.commands.commands],device=env.device)
        radius=self.cfg['rewards']['swept_width_m']/2+margins
        sweep=(gridxy[None,...,1].abs()<=radius[:,None,None])&(gridxy[None,...,0]>=0)&(gridxy[None,...,0]<=self.cfg['observability']['forward_m'])
        sweep=torch.where(turns[:,None,None],gridxy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m'],sweep)
        required=self.cfg['observability']['required_bands']
        near_known=required_observed(self.pipeline.near_state,self.pipeline.near_support_known,required)
        body_near=self.footprint.mask(gridxy,d.body_pos_w,d.body_quat_w,self.pose(),self.cfg['observability']['self_body_margin_m'])
        required_sweep=sweep&~body_near
        envelope_available=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        envelope_wide=None
        if self.envelope_masks is not None:
            # Exact measured command combinations only; no extrapolation from
            # a lower-speed gait. Current-body subtraction must not remove
            # future leg sweep from an actual trajectory envelope.
            near_rows=[];wide_rows=[];available=[]
            for tile,command in zip(tiles,self.control.commands.commands):
                mode=command.requested_mode;value=command.turn_rate if mode=='TURN' else command.v_cap
                key=f'{mode}_{abs(value):.1f}'
                found=self.envelope_masks.get(key) if abs(abs(value)-round(abs(value),1))<=1e-7 else None
                valid=found is not None and found[2] and tile.kind!='nav_stair_landing' and (mode!='TURN' or value<0)
                near_rows.append(found[0] if valid else required_sweep[len(near_rows)])
                wide_rows.append(found[1] if valid else widexy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m'])
                available.append(valid)
            required_sweep=torch.stack(near_rows)
            envelope_wide=torch.stack(wide_rows)
            envelope_available=torch.tensor(available,device=env.device)
        self.support_stable=contacts.all(-1)&~slipping
        if self.cfg.get('free_evidence_backend')=='pending167':
            from .evidence167 import certified_cells
            near_known=certified_cells(self.pipeline.near_state,self.pipeline.near_support_known)
        if self.pipeline.certification_enabled:
            near_known=required_observed(self.pipeline.cert_near_state,self.pipeline.near_support_known,required)
        if self.pipeline.certified_producer is not None:
            producer=self.pipeline.certified_producer
            if producer.temporal is None:
                fresh=producer.last_time==round(now*1000)
                near_known &= fresh[:,None,None]
            else:
                near_known=required_observed(self.pipeline.cert_near_state,producer.near_support_known,required)
        from .certification165 import unknown_risk
        opts=self.cfg['observability']
        if opts['r1_static_history']:near_known=self.cert_near
        near_risk,self.near_unknown_fraction=unknown_risk(near_known,required_sweep,resolution=NEAR_GRID.res,
            relaxed=opts['r2_unknown_components'],fraction_limit=opts['unknown_fraction_limit'],cluster_area=opts['unknown_cluster_area_m2'])
        known_sweep=~near_risk
        region=widexy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m']
        band_ids=required_band_indices(required)
        known_bands=(self.wide[:,[3*b for b in band_ids]]+self.wide[:,[3*b+1 for b in band_ids]])>0
        body_wide=self.footprint.mask(widexy,d.body_pos_w,d.body_quat_w,self.pose(),self.cfg['observability']['self_body_margin_m'])
        wide_known=self.cert_wide if opts['r1_static_history'] else known_bands.all(1)&known
        if self.pipeline.certification_enabled:
            wide_known=required_observed(self.pipeline.cert_wide_state,known,required)
        if self.pipeline.certified_producer is not None:
            if producer.temporal is None:wide_known &= fresh[:,None,None]
            else:wide_known=required_observed(self.pipeline.cert_wide_state,producer.wide_support_known,required)
        wide_required=region[None]&~body_wide if envelope_wide is None else envelope_wide
        wide_risk,self.wide_unknown_fraction=unknown_risk(wide_known,wide_required,resolution=WIDE_GRID.res,
            relaxed=opts['r2_unknown_components'],fraction_limit=opts['unknown_fraction_limit'],cluster_area=opts['unknown_cluster_area_m2'])
        if self.cfg.get('free_evidence_backend')=='pending167':
            # Pending measured free evidence cannot certify a turn sweep.
            wide_risk=torch.ones_like(wide_risk)
        known_turn=~wide_risk
        self.required_wide=wide_required;self.envelope_available=envelope_available
        self.required_sweep=required_sweep;self.body_near=body_near;self.body_wide=body_wide
        known_sweep &= ~turns|known_turn
        if self.envelope_masks is not None:known_sweep &= ~wide_risk & envelope_available
        actor_known179=required_observed(self.pipeline.near_state,self.pipeline.near_support_known,('B1','B2','B3'))
        self.actor_observable=((~required_sweep)|actor_known179).flatten(1).all(-1)
        if self.cfg.get('free_evidence_backend')=='coverage174':
            # Actual 3D future-volume gate; map band certificates are diagnostics only.
            known_sweep=self.pipeline.certified_producer.swept_certified
        occupancy176=self.cfg.get('supervisor176',{}).get('certification_mode')=='occupancy_grid'
        if occupancy176:
            from .occgrid_gate176 import certify_stopping_grid176
            producer=self.pipeline.certified_producer
            # No fallback to band certificates or GT if the measured producer
            # has not been integrated. The provider must query at current pose.
            observation,occ_grid=producer.query_occupancy176(
                pose=self.pose(),time_ms=torch.full_like(self.pipeline.mapper.epoch,round(now*1000)),
                epoch=self.pipeline.mapper.epoch)
            if self.cfg.get('terrain_fusion177',False):
                from .isaac_fusion177 import fuse_query177
                observation=fuse_query177(producer,observation,self.pose(),torch.full_like(self.pipeline.mapper.epoch,round(now*1000)),self.pipeline.mapper.epoch)
            if self.cfg.get('pipeline179',False):
                from .support174 import actual_contacts174
                from .pipeline179 import contact_support179,current_body_cells179
                points179,valid179=actual_contacts174(env,[self.robot.body_names[j] for j in self.foot_ids],self.pose(),world_to_sample)
                observation=contact_support179(observation,points179,valid179,occ_grid)
            if self.cfg.get('front_depth183',{}).get('enabled',False):
                observation=producer.remember(observation,self.pose(),torch.full_like(self.pipeline.mapper.epoch,round(now*1000)),self.pipeline.mapper.epoch)
            producer.last_query169['supervisor_observation179']=observation
            future=None if producer.envelope is None else producer.envelope.build(
                self,observation.support,occ_grid,projection_only=True)
            known_sweep,self.occupancy_diagnostics176=certify_stopping_grid176(
                observation,future,occ_grid,epoch=self.pipeline.mapper.epoch,
                margin_m=margins,ttl_s=self.cfg['supervisor176']['obstacle_ttl_s'],
                support_ttl_s=self.cfg['supervisor176']['support_ttl_s'],
                static_oracle183=self.cfg.get('front_depth183',{}).get('enabled',False),
                current_body_mask=current_body_cells179(self.footprint,occ_grid,d.body_pos_w,d.body_quat_w,self.pose(),self.cfg['current_body_margin_m']) if self.cfg.get('pipeline179',False) else None)
            actor_known179=((self.wide[:,[0,3,6]]+self.wide[:,[1,4,7]])>0).all(1)
            actor_required179=self.occupancy_diagnostics176['required']
            self.actor_observable=actor_required179.flatten(1).any(-1)&((~actor_required179)|actor_known179).flatten(1).all(-1)
            self.future_sweep174=future
            producer.swept_certified=known_sweep
            producer.swept_diagnostics=self.occupancy_diagnostics176
            producer.stats.update(qualified=int(self.occupancy_diagnostics176.get('qualified',torch.zeros_like(known_sweep)).sum()),observed=int(known_sweep.sum()),cells_satisfied=int(self.occupancy_diagnostics176['cells_satisfied'].sum()),required_cells=int(self.occupancy_diagnostics176['required'].sum()),body_excluded_cells=int((self.occupancy_diagnostics176['required_before_exclusion']&~self.occupancy_diagnostics176['required']).sum()))
        self.unknown_time=torch.where(known_sweep,0.,self.unknown_time+env.step_dt)
        risky=((self.near[:,:2]*S.NEAR_CLEAR_MAX<margins[:,None,None,None])&sweep[:,None]).flatten(1).any(-1)
        if self.cfg.get('free_evidence_backend')=='coverage174':
            risky=((self.pipeline.certified_producer.safety_clearance<margins[:,None,None,None])&sweep[:,None]).flatten(1).any(-1)
        if occupancy176:
            # Occupied risk and unknown certification use the same single grid.
            risky=self.occupancy_diagnostics176.get('occupied',torch.zeros_like(known_sweep))
        clearance_risk=risky
        unknown_risk=self.unknown_time>=self.cfg['observability']['unknown_hold_s']
        self.replan_requested=unknown_risk&~self.legacy
        risky=(risky|unknown_risk)&~self.legacy
        diagnostics=None
        if self.reason_metrics_enabled:
            known_fraction=(near_known&required_sweep).flatten(1).sum(-1)/required_sweep.flatten(1).sum(-1).clamp_min(1)
            wide_sweep=(widexy[None,...,1].abs()<=radius[:,None,None])&(widexy[None,...,0]>=0)&(widexy[None,...,0]<=self.cfg['observability']['forward_m'])
            wide_sweep=torch.where(turns[:,None,None],region,wide_sweep)
            denom=wide_sweep.flatten(1).sum(-1).clamp_min(1)
            coverage=[((self.wide[:,channel]>0)&wide_sweep).flatten(1).sum(-1)/denom for channel in (0,1,3,4,6,7)]
            support_fraction=(known&wide_sweep).flatten(1).sum(-1)/denom
            diagnostics=torch.stack((clearance_risk,unknown_risk,self.unknown_time,known_sweep,known_turn,known_fraction,
                support_fraction,*coverage,clearance_risk&~unknown_risk,unknown_risk&~clearance_risk,clearance_risk&unknown_risk,envelope_available),-1).tolist()
            self.decision_rows=[]
        # CPU protocol state machines receive one batched transfer, never one
        # CUDA scalar read per env/field. Sensor geometry above stays on GPU.
        host=torch.cat(((d.root_pos_w-env.scene.env_origins)[:,:2],d.heading_w[:,None],
            d.root_lin_vel_b[:,:2].norm(dim=-1)[:,None],d.root_ang_vel_b[:,2,None],
            torch.stack((flat,descent,contacts.all(-1),slipping,risky,(~known_sweep)&~self.legacy),-1).float(),
            env.command_manager.get_command('base_velocity')),dim=-1).tolist()
        epochs=self.pipeline.mapper.epoch.tolist()
        controllers=[];velocities=[];features=[];phases=[];go=[];route_features=[]
        age=(now-self.last_capture)*1000
        for i,(tile,row) in enumerate(zip(tiles,host)):
            x,y,yaw,speed,yaw_rate,flat_i,descent_i,contact_i,slip_i,risk_i,unknown_wait_i,*legacy=row
            xy=(x,y);route_info=route_hint(tile,xy);hint=route_info[0]
            hint_landing=hint.segment_type=='LANDING' or hint.next_segment_type=='LANDING'
            o=LandingObservation(hint_landing or hint.segment_type=='STAIR_DOWN',bool(flat_i) and hint_landing,
                hint.dist_to_next_m<=self.cfg['commands']['brake_distance_m'],bool(flat_i) and hint.segment_type=='LANDING',
                bool(contact_i),speed,yaw_rate,bool(slip_i),bool(descent_i),hint.segment_type=='STAIR_DOWN',hint.segment_type=='FLAT')
            fault=self.force_fault and i==self.cfg['smoke']['fault_env_id']
            health=HealthInput(age if not fault else 1000.,age,0.,risk=bool(risk_i))
            output=self.control.before_policy(i,now_ms=round(now*1000),dt=env.step_dt,xy=xy,yaw=yaw,
                health=health,landing_observation=o,odom_epoch=epochs[i],legacy_velocity=legacy,route_info=route_info,unknown_wait=bool(unknown_wait_i))
            controllers.append(S.CONTROLLERS.index(output.controller));self.reasons[i]=output.reason
            if diagnostics is not None:
                cmd=self.control.commands.commands[i];adapter=self.control.commands.adapters[i]
                self.decision_rows.append(dict(kind=tile.kind,requested_mode=cmd.requested_mode,
                    mode=S.MODES[max(range(3),key=lambda j:output.features[j])],phase=output.phase,
                    handshake=adapter.handshake,controller=output.controller,reason=output.reason,
                    requested_go=cmd.requested_go,effective_go=output.effective_go,speed=speed,
                    abs_yaw_rate=abs(yaw_rate),stable_contacts=bool(contact_i),slipping=bool(slip_i),
                    **dict(zip(('clearance_risk','unknown_risk','unknown_s','observed','known_turn','known_fraction','support_fraction','b1_occupied','b1_free','b2_occupied','b2_free','b3_occupied','b3_free','clearance_only','unknown_only','both_risks','envelope_available'),diagnostics[i]))))
            route_features.append(route_features179(hint,yaw,mode=S.MODES[max(range(3),key=lambda j:output.features[j])]))
            velocities.append(output.velocity);features.append(output.features);phases.append(S.PHASES.index(output.phase));go.append(output.effective_go)
        self.controller=torch.tensor(controllers,device=env.device,dtype=torch.long)
        self.velocity=torch.tensor(velocities,device=env.device,dtype=torch.float32)
        self.features=torch.tensor(features,device=env.device,dtype=torch.float32)
        self.route_features=torch.tensor(route_features,device=env.device,dtype=torch.float32)
        self.mode=self.features[:,:3].argmax(-1);self.phase=torch.tensor(phases,device=env.device,dtype=torch.long);self.go=torch.tensor(go,device=env.device)
        self.positive_yaw+=d.root_ang_vel_b[:,2].clamp_min(0)*env.step_dt*(self.mode==2)
        self.drift=torch.maximum(self.drift,(d.root_pos_w[:,:2]-self.turn_start).norm(dim=-1))
        self.supervisor_certified=known_sweep
        self.actor_loss_valid=self.controller==S.CONTROLLERS.index("NAV_POLICY")
        self.observed=self.supervisor_certified # compatibility; reward certificate, not actor visibility
        self.stall_enabled=legacy_reward_masks(self.mode,self.phase,self.go,self.controller!=0)['stair_stall']
        command=env.command_manager.get_term('base_velocity')
        command.vel_command_b[~self.legacy]=self.velocity[~self.legacy];command.vel_command_b[self.controller!=0]=0
        # Replace current command history sample; older samples remain actual history.
        from .obs_terms import replace_latest_command
        replace_latest_command(obs['policy'],command.vel_command_b,env.cfg.observations.policy.velocity_commands.scale)
        self.compute_rewards()
        if self.reason_metrics_enabled:
            metric_names179=['actor_observable','supervisor_certified','physical_feasible','actor_loss_valid']
            metric_values179=[self.actor_observable,self.supervisor_certified,self.feasible,self.actor_loss_valid]
            metric_names179 += ['unknown_entry184','unknown_stop184','stair_end_event184','stair_end_wait184','ccw_angle184']
            metric_values179 += [self.entered_unknown & ~self.legacy, (~self.observed)&~self.legacy&self.support_stable&(d.root_lin_vel_b[:,:2].norm(dim=-1)<self.cfg['rewards']['stopped_speed_mps']),self.stair_end184.event,self.stair_end184.wait,d.root_ang_vel_b[:,2].clamp_min(0)*env.step_dt*(self.mode==2)]
            if occupancy176:
                for name179 in ('qualified','cells_satisfied','outside','valid_epoch'):
                    metric_names179.append('sweep_'+name179);metric_values179.append(self.occupancy_diagnostics176[name179])
            separated=torch.stack(metric_values179,-1).tolist()
            for row,values179 in zip(self.decision_rows,separated):
                row.update(dict(zip(metric_names179,values179)))
        if 'unknown_entry' in self.cfg['rewards']:
            actual_body=self.footprint.mask(gridxy,d.body_pos_w,d.body_quat_w,self.pose(),NEAR_GRID.res/math.sqrt(2))
            if occupancy176:
                entry_grid=occ_grid
                actual_body=self.footprint.mask(occ_grid.xy(device=env.device,dtype=torch.float32),d.body_pos_w,d.body_quat_w,self.pose(),occ_grid.res/math.sqrt(2))
                entry_known=(observation.state!=S.UNKNOWN)&(observation.age_s<=self.cfg['supervisor176']['obstacle_ttl_s'])&(observation.age_s>=0)&torch.isfinite(observation.age_s)
                entry_known &= (observation.epoch==self.pipeline.mapper.epoch)[:,None,None]
            else:entry_grid=NEAR_GRID;entry_known=near_known
            self.entry_previous=(self.pose().clone(),actual_body,entry_known.clone(),entry_grid)
            self.entry_valid[:]=True
        obs.update(nav_wide=self.wide,nav_near=self.near,nav_features=self.features.clone(),
                   nav_route_features=self.route_features.clone(),
                   nav_privileged=critic_privileged(sdf=self.clearance[:,None],feasible=self.feasible,observed=self.observed,
                       acceleration=self.inertia.acceleration,controller=self.controller),
                   nav_controller=self.controller.clone(),nav_mode=self.mode.clone())
        return obs

    def top5_take(self):
        out=self.top5_counts.clone();self.top5_counts.zero_();return out

    def compute_rewards(self):
        env=self.env;d=self.robot.data
        # RewardManager calls this after physics, BEFORE prepare(). Read the
        # previous policy-step certificate without advancing that baseline.
        if 'unknown_entry' in self.cfg['rewards']:
            from .evidence167 import entered_unknown_cells
            if self.entry_previous is not None:
                prev_pose,prev_body,prev_known,entry_grid=self.entry_previous
                gridxy=entry_grid.xy(device=env.device,dtype=torch.float32)
                current_body=self.footprint.mask(gridxy,d.body_pos_w,d.body_quat_w,prev_pose,entry_grid.res/math.sqrt(2))
                self.entered_unknown=entered_unknown_cells(current_body,prev_body,prev_known,self.entry_valid)
            else:self.entered_unknown.zero_()
            contacts=env.scene['contact_forces'].data.net_forces_w[:,self.contact_ids].norm(dim=-1)>self.cfg['sensors']['contact_threshold_n']
            slip=(d.body_lin_vel_w[:,self.foot_ids,:2].norm(dim=-1)>self.cfg['landing_sensor']['slip_speed_mps'])&contacts
            self.support_stable=contacts.all(-1)&~slip.any(-1)
        margins=torch.tensor([a.c.margin for a in self.control.commands.adapters],device=env.device)
        offsets=torch.tensor([a.c.y_off for a in self.control.commands.adapters],device=env.device)
        values=self.oracle.evaluate(self.tile_ids,(d.root_pos_w-env.scene.env_origins)[:,:2],self.mode==2,margins,
            swept_width=self.cfg['rewards']['swept_width_m'],turn_clear_radius=self.cfg['observability']['turn_radius_m'],sdf_clip=self.cfg['observability']['sdf_clip_m'])
        self.axis=values['axis'];self.offset=values['offset']-offsets;self.feasible=values['feasible'];self.clearance=values['clearance']
        radius=torch.where(self.mode==2,self.cfg['observability']['turn_radius_m'],self.cfg['rewards']['swept_width_m']/2)
        gap=(d.root_pos_w[:,:2]-self.cylinder.position[:,:2]).norm(dim=-1)-self.cylinder.radius-radius
        self.clearance=torch.where(self.cylinder.enabled&~self.legacy,torch.minimum(self.clearance,gap),self.clearance)
        segment=values['segment'];stair=(segment==S.SEGMENT_TYPES.index('STAIR_UP'))|(segment==S.SEGMENT_TYPES.index('STAIR_DOWN'))|(segment==S.SEGMENT_TYPES.index('LANDING'))
        stair &= ~self.legacy
        v=d.root_lin_vel_w[:,0]*self.axis.cos()+d.root_lin_vel_w[:,1]*self.axis.sin()
        yaw=torch.atan2(torch.sin(d.heading_w-self.axis),torch.cos(d.heading_w-self.axis))
        contact=(env.scene['contact_forces'].data.net_forces_w.norm(dim=-1)>self.cfg['sensors']['contact_threshold_n'])
        contact[:,self.contact_ids]=False
        # RewardManager and prepare both call compute_rewards in one tick.
        # Advance reward-only temporal labels once, never in actor assembly.
        tick184=int(env.common_step_counter)
        if self.stair_tick184!=tick184:
            self.stair_tick184=tick184
            q=d.root_quat_w;w,x,y,z=q.unbind(-1)
            pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1));roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
            feet_contact=env.scene['contact_forces'].data.net_forces_w[:,self.contact_ids].norm(dim=-1)>self.cfg['sensors']['contact_threshold_n']
            cmds=self.control.commands.commands
            ahead_flat=None
            if self.stair_end184.cfg.get('rule','ahead_flat')=='ahead_flat':
                # Reward-only GT label: privileged height scan ahead along the route axis.
                hits=env.scene['height_scanner'].data.ray_hits_w
                rel=hits[...,:2]-d.root_pos_w[:,None,:2]
                fwd=rel[...,0]*self.axis.cos()[:,None]+rel[...,1]*self.axis.sin()[:,None]
                lat=-rel[...,0]*self.axis.sin()[:,None]+rel[...,1]*self.axis.cos()[:,None]
                c=self.stair_end184.cfg
                sel=(fwd>=c['ahead_min_m'])&(fwd<=c['ahead_max_m'])&(lat.abs()<=c['ahead_half_width_m'])&torch.isfinite(hits[...,2])
                z=hits[...,2];big=torch.full_like(z,1e3)
                zmax=torch.where(sel,z,-big).amax(-1);zmin=torch.where(sel,z,big).amin(-1)
                foot=d.body_pos_w[:,self.foot_ids,2].mean(-1)-c['foot_radius_m']
                ahead_flat=(sel.sum(-1)>=c['ahead_min_points'])&((zmax-zmin)<=c['ahead_flat_m'])&((0.5*(zmax+zmin)-foot).abs()<=c['ahead_level_m'])
            # Online Top-5 rate counters (D-9): P progress, T stop, U unknown entry, R reverse rotation.
            vcap=self.features[:,4].clamp_min(self.cfg['commands']['minimum_cap_mps'])
            vpath=d.root_lin_vel_w[:,0]*self.axis.cos()+d.root_lin_vel_w[:,1]*self.axis.sin()
            nav=(self.mode==1)&~self.legacy;turn=(self.mode==2)&~self.legacy;nav_go=nav&self.go.bool()
            speed_b=d.root_lin_vel_b[:,:2].norm(dim=-1);yaw_b=d.root_ang_vel_b[:,2].abs()
            stopped=(speed_b<self.cfg['rewards']['stopped_speed_mps'])&(yaw_b<self.cfg['rewards'].get('stopped_yaw_rate_radps',.05))
            need_stop=(nav|turn)&(~self.go.bool()|self.stair_end184.wait)
            self.last_need_stop=need_stop;self.last_stopped=stopped
            c=self.top5_counts
            c[0]+=(vpath/vcap).clamp(0,1)[nav_go].sum();c[1]+=nav_go.sum()
            c[2]+=(need_stop&stopped).sum();c[3]+=need_stop.sum()
            if os.environ.get('GAST_NAV_STOP_DIAG'):
                # Off by default. Per need-stop tick: why T is low (2026-10-21 diagnosis).
                if not hasattr(self,'stop_off_s'):self.stop_off_s=torch.zeros_like(speed_b);self.stop_rows=[]
                self.stop_off_s=torch.where(need_stop,self.stop_off_s+env.step_dt,torch.zeros_like(self.stop_off_s))
                ids=need_stop.nonzero().flatten()[:64].tolist();tasks=self.control.commands.task
                self.stop_rows+=[(tasks[i],int(self.mode[i]),bool(self.stair_end184.wait[i]),round(float(self.stop_off_s[i]),3),
                                  round(float(speed_b[i]),4),round(float(yaw_b[i]),4)) for i in ids]
                if len(self.stop_rows)>=2000:
                    with open(os.environ['GAST_NAV_STOP_DIAG'],'a') as f:f.writelines(' '.join(map(str,r))+'\n' for r in self.stop_rows)
                    self.stop_rows=[]
            c[4]+=(self.entered_unknown.bool()&nav).sum();c[5]+=nav.sum()
            c[6]+=(torch.rad2deg(self.positive_yaw)/3.).clamp(0,1)[turn].sum();c[7]+=turn.sum()
            self.stair_end184.tick(foot_z=d.body_pos_w[:,self.foot_ids,2],contacts=feet_contact,pitch=pitch,roll=roll,
                stair_surface=stair,nav=(self.mode==1),go=torch.tensor([c.requested_go for c in cmds],device=env.device),
                request_id=torch.tensor([c.manoeuvre_id for c in cmds],device=env.device),dt=env.step_dt,ahead_flat=ahead_flat)
        _,weighted=reward_terms(mode=self.mode,go=self.go,phase=self.phase,feasible=self.feasible,observed=self.observed,supervisor_certified=self.supervisor_certified,
            v_path=v,v_cap=self.features[:,4].clamp_min(self.cfg['commands']['minimum_cap_mps']),speed=d.root_lin_vel_b[:,:2].norm(dim=-1),
            yaw_rate=d.root_ang_vel_b[:,2],turn_rate=self.features[:,5],positive_yaw=self.positive_yaw,clearance=self.clearance,
            margin=self.features[:,14],offset_error=self.offset,yaw_error=yaw,turn_drift=self.drift,contact=contact.any(-1),
            entered_forbidden=self.clearance<self.features[:,14],cfg=self.cfg['rewards'],stair_mask=stair,
            support_stable=self.support_stable,entered_unknown=self.entered_unknown,supervisor_mode=self.cfg.get('supervisor_mode','enforce'),
            stair_end_wait=self.stair_end184.wait if self.cfg.get('supervisor_mode')=='shadow' else None)
        self.reward_weighted=weighted  # per-env rates; RL replay splits nav_total with these
        self.reward_log={k:float(v.mean()) for k,v in weighted.items()}
        return sum(weighted.values())

class GastNavEnv(ManagerBasedRLEnv):
    def __init__(self,*args,**kwargs):
        self.nav=None
        super().__init__(*args,**kwargs)
        self.nav=NavState(self)

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if self.nav is not None:self.nav.reset(env_ids)

    def reset(self,*args,**kwargs):
        obs,extras=super().reset(*args,**kwargs)
        return self.nav.prepare(obs),extras

    def step(self,action):
        controller=self.nav.controller.clone();reasons=list(self.nav.reasons)
        replan_requested=self.nav.replan_requested.clone()
        decision_rows=self.nav.decision_rows
        executed=action.clone();hold=controller==1;damp=controller==2
        if hold.any():
            if self.nav.fixed_policy is None:raise RuntimeError('fixed 3906 policy required for HOLD')
            from tensordict import TensorDict
            obs=TensorDict(self.obs_buf,batch_size=[self.num_envs])
            fixed=zero_legacy_command_history(obs,term_dims=[3,3,3,12,12,12,1],history_length=5)
            with torch.inference_mode():executed[hold]=self.nav.fixed_policy.act_inference(fixed)[hold]
        if damp.any():
            # Simulation VISION_HOLD uses the fixed policy with zero commands;
            # hardware DAMP dispatch is outside this Isaac environment.
            if self.nav.fixed_policy is None:raise RuntimeError('fixed policy required for simulation fallback')
            from tensordict import TensorDict
            obs=TensorDict(self.obs_buf,batch_size=[self.num_envs])
            fixed=zero_legacy_command_history(obs,term_dims=[3,3,3,12,12,12,1],history_length=5)
            with torch.inference_mode():executed[damp]=self.nav.fixed_policy.act_inference(fixed)[damp]
        obs,reward,terminated,truncated,extras=super().step(executed)
        obs=self.nav.prepare(obs)
        if self.nav.reason_metrics_enabled:
            self.nav.hold_reason_metrics.add(decision_rows)
            extras['nav_decision_rows']=decision_rows
            if self.common_step_counter%100==0:
                extras['nav_hold_reason_metrics']=self.nav.hold_reason_metrics.snapshot()
        extras['person_contact']=self.nav.person_contacts.last_contact.clone()
        extras['person_invalid_code']=self.nav.person_contacts.last_invalid_code.clone()
        extras['person_contact_counters']=self.nav.person_contacts.summary()
        extras['cylinder_sway_stub']=getattr(self.nav.cylinder,'sway_api_stub',True)
        extras['nav_controller']=controller;extras['nav_executed_action']=executed
        extras['nav_replan_requested']=replan_requested
        extras['nav_override_reason']=torch.tensor([0 if not x or x=='OK' else 1 for x in reasons],device=self.device);extras['nav_rewards']=dict(self.nav.reward_log)
        self.obs_buf=obs
        return obs,reward,terminated,truncated,extras
