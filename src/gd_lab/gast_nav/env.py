"""Isaac environment adapter. Policy selection happens before action execution."""
import copy
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
from .control_runtime import ControlRuntime,zero_legacy_command_history
from .landing_fsm import LandingObservation
from .supervisor import HealthInput
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
        self.cylinder=EpisodeCylinder(self.index,stage='B',device=dev,seed=self.cfg['seed'],env_origins=env.scene.env_origins)
        self.person_contacts=PersonContactTracker(n,self.cfg['person_contact']['force_threshold_n'],device=dev)
        import omni.usd
        self.reset_overlap_query=IsaacResetOverlapQuery(omni.usd.get_context().get_stage())
        expected=set(self.cfg['person_contact']['robot_links'])
        if expected!=set(self.robot.body_names):
            raise RuntimeError(f'person-contact link coverage mismatch: configured={expected}, actual={set(self.robot.body_names)}')
        self.rng=torch.Generator(device=dev).manual_seed(self.cfg['seed'])
        self.controller=torch.zeros(n,dtype=torch.long,device=dev)
        self.velocity=torch.zeros(n,3,device=dev);self.features=torch.zeros(n,16,device=dev)
        self.mode=torch.zeros(n,dtype=torch.long,device=dev);self.phase=self.mode.clone()
        self.go=torch.zeros(n,dtype=torch.bool,device=dev);self.legacy=torch.ones_like(self.go)
        self.stall_enabled=torch.ones_like(self.go);self.feasible=torch.ones_like(self.go);self.observed=torch.zeros_like(self.go)
        self.clearance=torch.ones(n,device=dev);self.offset=torch.zeros(n,device=dev);self.axis=self.offset.clone()
        self.unknown_time=self.offset.clone();self.positive_yaw=self.offset.clone();self.drift=self.offset.clone();self.turn_start=torch.zeros(n,2,device=dev)
        self.reasons=['']*n;self.last_capture=-math.inf;self.wide=self.near=None;self.reward_log={}
        self.force_fault=False;self.fixed_policy=None
        self.foot_ids=[self.robot.body_names.index(f'{leg}_foot') for leg in ('FL','FR','RL','RR')]
        sensor=env.scene['contact_forces'];self.contact_ids=[sensor.body_names.index(self.robot.body_names[i]) for i in self.foot_ids]

    def pose(self):
        d=self.robot.data
        return torch.cat((d.root_pos_w,d.heading_w[:,None]),-1)

    def reset(self,ids):
        env=self.env;ids=torch.as_tensor(ids,device=env.device,dtype=torch.long)
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
        self.person_contacts.reset(ids)
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
            points=world_to_sample(env.scene['height_scanner'].data.ray_hits_w,pose)
            valid=torch.isfinite(points).all(-1)
            contacts=world_to_sample(self.robot.data.body_pos_w[:,self.foot_ids],pose)
            forces=env.scene['contact_forces'].data.net_forces_w[:,self.contact_ids]
            contact_valid=forces.norm(dim=-1)>self.cfg['sensors']['contact_threshold_n']
            xy=(self.robot.data.root_pos_w-env.scene.env_origins)[:,:2].tolist()
            segments=torch.tensor([S.SEGMENT_TYPES.index(route_hint(t,p)[0].segment_type)
                                  for t,p in zip(self.index.lookup(range(env.num_envs)),xy)],device=env.device)
            self.wide,self.near=self.pipeline.capture(rays=rays,terrain_points=points,terrain_valid=valid,
                 contact_points=contacts,contact_valid=contact_valid,segment=segments,pose=pose,
                 time_ms=torch.full((env.num_envs,),round(now*1000),device=env.device,dtype=torch.long),
                 capsules=(capsules_to_sample(self.cylinder.capsules(),pose) if self.cylinder.capsules() is not None else None))
            self.last_capture=now
        # UPDATE162: hold the capture-aligned inputs between 10Hz captures.
        # Sensor age/health still advances each policy tick; reset forces capture.

    def prepare(self,obs):
        env=self.env;d=self.robot.data;now=env.common_step_counter*env.step_dt
        self.capture(now)
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
        known_sweep=~((self.near[:,2]>0)&sweep).flatten(1).any(-1)
        region=widexy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m']
        known_bands=(self.wide[:,[0,3,6]]+self.wide[:,[1,4,7]])>0
        known_turn=(known_bands|~region[None,None]).flatten(1).all(-1)
        known_sweep &= ~turns|known_turn
        self.unknown_time=torch.where(known_sweep,0.,self.unknown_time+env.step_dt)
        risky=((self.near[:,:2]*S.NEAR_CLEAR_MAX<margins[:,None,None,None])&sweep[:,None]).flatten(1).any(-1)
        risky=(risky|(self.unknown_time>=self.cfg['observability']['unknown_hold_s']))&~self.legacy
        # CPU protocol state machines receive one batched transfer, never one
        # CUDA scalar read per env/field. Sensor geometry above stays on GPU.
        host=torch.cat(((d.root_pos_w-env.scene.env_origins)[:,:2],d.heading_w[:,None],
            d.root_lin_vel_b[:,:2].norm(dim=-1)[:,None],d.root_ang_vel_b[:,2,None],
            torch.stack((flat,descent,contacts.all(-1),slipping,risky),-1).float(),
            env.command_manager.get_command('base_velocity')),dim=-1).tolist()
        epochs=self.pipeline.mapper.epoch.tolist()
        controllers=[];velocities=[];features=[];phases=[];go=[]
        age=(now-self.last_capture)*1000
        for i,(tile,row) in enumerate(zip(tiles,host)):
            x,y,yaw,speed,yaw_rate,flat_i,descent_i,contact_i,slip_i,risk_i,*legacy=row
            xy=(x,y);hint,_,_,_=route_hint(tile,xy)
            hint_landing=hint.segment_type=='LANDING' or hint.next_segment_type=='LANDING'
            o=LandingObservation(hint_landing or hint.segment_type=='STAIR_DOWN',bool(flat_i) and hint_landing,
                hint.dist_to_next_m<=self.cfg['commands']['brake_distance_m'],bool(flat_i) and hint.segment_type=='LANDING',
                bool(contact_i),speed,yaw_rate,bool(slip_i),bool(descent_i),hint.segment_type=='STAIR_DOWN',hint.segment_type=='FLAT')
            fault=self.force_fault and i==self.cfg['smoke']['fault_env_id']
            health=HealthInput(age if not fault else 1000.,age,0.,risk=bool(risk_i))
            output=self.control.before_policy(i,now_ms=round(now*1000),dt=env.step_dt,xy=xy,yaw=yaw,
                health=health,landing_observation=o,odom_epoch=epochs[i],legacy_velocity=legacy)
            controllers.append(S.CONTROLLERS.index(output.controller));self.reasons[i]=output.reason
            velocities.append(output.velocity);features.append(output.features);phases.append(S.PHASES.index(output.phase));go.append(output.effective_go)
        self.controller=torch.tensor(controllers,device=env.device,dtype=torch.long)
        self.velocity=torch.tensor(velocities,device=env.device,dtype=torch.float32)
        self.features=torch.tensor(features,device=env.device,dtype=torch.float32)
        self.mode=self.features[:,:3].argmax(-1);self.phase=torch.tensor(phases,device=env.device,dtype=torch.long);self.go=torch.tensor(go,device=env.device)
        self.positive_yaw+=d.root_ang_vel_b[:,2].clamp_min(0)*env.step_dt*(self.mode==2)
        self.drift=torch.maximum(self.drift,(d.root_pos_w[:,:2]-self.turn_start).norm(dim=-1))
        self.observed=known_sweep
        self.stall_enabled=legacy_reward_masks(self.mode,self.phase,self.go,self.controller!=0)['stair_stall']
        command=env.command_manager.get_term('base_velocity')
        command.vel_command_b[~self.legacy]=self.velocity[~self.legacy];command.vel_command_b[self.controller!=0]=0
        # Replace current command history sample; older samples remain actual history.
        obs['policy'][:,30:45].reshape(env.num_envs,5,3)[:,-1]=command.vel_command_b
        self.compute_rewards()
        obs.update(nav_wide=self.wide,nav_near=self.near,nav_features=self.features.clone(),
                   nav_privileged=critic_privileged(sdf=self.clearance[:,None],feasible=self.feasible,observed=self.observed,
                       acceleration=self.inertia.acceleration,controller=self.controller),
                   nav_controller=self.controller.clone(),nav_mode=self.mode.clone())
        return obs

    def compute_rewards(self):
        env=self.env;d=self.robot.data
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
        _,weighted=reward_terms(mode=self.mode,go=self.go,phase=self.phase,feasible=self.feasible,observed=self.observed,
            v_path=v,v_cap=self.features[:,4].clamp_min(self.cfg['commands']['minimum_cap_mps']),speed=d.root_lin_vel_b[:,:2].norm(dim=-1),
            yaw_rate=d.root_ang_vel_b[:,2],turn_rate=self.features[:,5],positive_yaw=self.positive_yaw,clearance=self.clearance,
            margin=self.features[:,14],offset_error=self.offset,yaw_error=yaw,turn_drift=self.drift,contact=contact.any(-1),
            entered_forbidden=self.clearance<self.features[:,14],cfg=self.cfg['rewards'],stair_mask=stair)
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
        extras['person_contact']=self.nav.person_contacts.last_contact.clone()
        extras['person_invalid_code']=self.nav.person_contacts.last_invalid_code.clone()
        extras['person_contact_counters']=self.nav.person_contacts.summary()
        extras['cylinder_sway_stub']=getattr(self.nav.cylinder,'sway_api_stub',True)
        extras['nav_controller']=controller;extras['nav_executed_action']=executed
        extras['nav_override_reason']=torch.tensor([0 if not x or x=='OK' else 1 for x in reasons],device=self.device);extras['nav_rewards']=dict(self.nav.reward_log)
        self.obs_buf=obs
        return obs,reward,terminated,truncated,extras
