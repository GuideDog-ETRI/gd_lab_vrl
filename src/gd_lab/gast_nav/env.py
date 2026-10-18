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
                        randomize_sensor_pattern,critic_privileged)
from .map_types import DynamicCapsules,NEAR_GRID,WIDE_GRID
from .rewards import reward_terms,legacy_reward_masks

class NavState:
    def __init__(self,env):
        self.env=env;self.robot=env.scene['robot'];self.cfg=env.cfg.nav_settings
        self.stage=env.cfg.nav_stage_settings;n=env.num_envs;dev=env.device
        runtime=yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/gast_nav/runtime.yaml').read_text())
        # Explicit simulation-only cap; deployment runtime.yaml remains unverified.
        runtime['command']['omega_max_verified']=self.cfg['commands']['simulation_turn_cap_radps']
        self.index=bind_scene(env.scene.terrain)
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
        tiles=self.index.lookup(ids);meta=self.index.tensors(env.device)
        self.legacy[ids]=torch.tensor([t.kind=='legacy' for t in tiles],device=env.device)
        # Keep env origins at tile centres; only robot spawn uses route start.
        nav_ids=ids[~self.legacy[ids]]
        if len(nav_ids):
            state=self.robot.data.root_state_w[nav_ids].clone()
            state[:,:2]=env.scene.env_origins[nav_ids,:2]+meta['spawn_xy'][nav_ids]
            state[:,2]=env.scene.env_origins[nav_ids,2]+self.robot.data.default_root_state[nav_ids,2]
            yaw=meta['route_axis_yaw'][nav_ids];zero=torch.zeros_like(yaw)
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
        self.last_capture=-math.inf

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
        else:self.wide,self.near=self.pipeline.inputs(pose,round(now*1000))

    def prepare(self,obs):
        env=self.env;d=self.robot.data;now=env.common_step_counter*env.step_dt
        self.capture(now)
        points=(d.root_pos_w-env.scene.env_origins)[:,:2].tolist();tiles=self.index.lookup(range(env.num_envs))
        contacts=env.scene['contact_forces'].data.net_forces_w[:,self.contact_ids].norm(dim=-1)>self.cfg['sensors']['contact_threshold_n']
        ground=self.pipeline.mapper.history[0]
        from .support import decode_support
        height,known=decode_support(ground.support)
        gridxy=NEAR_GRID.xy(device=env.device,dtype=torch.float32)
        widexy=WIDE_GRID.xy(device=env.device,dtype=torch.float32)
        for i,(tile,xy) in enumerate(zip(tiles,points)):
            hint,_,_,_=route_hint(tile,xy)
            # Sensor-only landing evidence; route hint supplies semantics, not GT landing bounds.
            centre=height[i,19:24,22:27];valid=known[i,19:24,22:27]
            flat=bool(valid.all() and (centre.max()-centre.min()<self.cfg['landing_sensor']['flat_tolerance_m']))
            front=height[i,19:24,27:33];front_valid=known[i,19:24,27:33]
            descent=bool(front_valid.any() and valid.any() and
                         front[front_valid].min()<centre[valid].median()-self.cfg['landing_sensor']['descent_drop_m'])
            hint_landing=hint.segment_type=='LANDING' or hint.next_segment_type=='LANDING'
            o=LandingObservation(hint_landing or hint.segment_type=='STAIR_DOWN',flat and hint_landing,
                hint.dist_to_next_m<=self.cfg['commands']['brake_distance_m'],flat and hint.segment_type=='LANDING',
                bool(contacts[i].all()),float(d.root_lin_vel_b[i,:2].norm()),float(d.root_ang_vel_b[i,2]),
                bool((self.robot.data.body_lin_vel_w[i,self.foot_ids,:2].norm(dim=-1)[contacts[i]]>
                      self.cfg['landing_sensor']['slip_speed_mps']).any()),descent,
                hint.segment_type=='STAIR_DOWN',hint.segment_type=='FLAT')
            age=(now-self.last_capture)*1000
            fault=self.force_fault and i==self.cfg['smoke']['fault_env_id']
            radius=self.cfg['rewards']['swept_width_m']/2+self.control.commands.commands[i].c.margin
            sweep=(gridxy[...,1].abs()<=radius)&(gridxy[...,0]>=0)&(gridxy[...,0]<=self.cfg['observability']['forward_m'])
            if self.control.commands.commands[i].requested_mode=='TURN':
                sweep=gridxy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m']
            known_sweep=not bool(self.near[i,2][sweep].any())
            if self.control.commands.commands[i].requested_mode=='TURN':
                region=widexy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m']
                known_bands=(self.wide[i,[0,3,6]]+self.wide[i,[1,4,7]])>0
                known_sweep=known_sweep and bool(known_bands[:,region].all())
            self.unknown_time[i]=0. if known_sweep else self.unknown_time[i]+env.step_dt
            risky=bool((self.near[i,:2,:, :][:,sweep]*S.NEAR_CLEAR_MAX < self.control.commands.commands[i].c.margin).any())
            risky=(risky or float(self.unknown_time[i])>=self.cfg['observability']['unknown_hold_s']) and not bool(self.legacy[i])
            health=HealthInput(age if not fault else 1000.,age,0.,risk=risky)
            output=self.control.before_policy(i,now_ms=round(now*1000),dt=env.step_dt,xy=xy,yaw=float(d.heading_w[i]),
                health=health,landing_observation=o,odom_epoch=int(self.pipeline.mapper.epoch[i]),
                legacy_velocity=env.command_manager.get_command('base_velocity')[i].tolist())
            self.controller[i]=S.CONTROLLERS.index(output.controller);self.reasons[i]=output.reason
            self.velocity[i]=self.velocity.new_tensor(output.velocity);self.features[i]=self.features.new_tensor(output.features)
            self.mode[i]=self.features[i,:3].argmax();self.phase[i]=S.PHASES.index(output.phase);self.go[i]=output.effective_go
            self.positive_yaw[i]+=max(float(d.root_ang_vel_b[i,2]),0)*env.step_dt*(int(self.mode[i])==2)
            self.drift[i]=max(float(self.drift[i]),float((d.root_pos_w[i,:2]-self.turn_start[i]).norm()))
            radius=self.cfg['rewards']['swept_width_m']/2+self.control.commands.adapters[i].c.margin
            sweep=(gridxy[...,1].abs()<=radius)&(gridxy[...,0]>=0)&(gridxy[...,0]<=self.cfg['observability']['forward_m'])
            if int(self.mode[i])==2:sweep=gridxy.norm(dim=-1)<=self.cfg['observability']['turn_radius_m']
            self.observed[i]=known_sweep
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
        env=self.env;d=self.robot.data;points=(d.root_pos_w-env.scene.env_origins)[:,:2].tolist()
        stair=torch.zeros_like(self.go)
        for i,(tile,xy) in enumerate(zip(self.index.lookup(range(env.num_envs)),points)):
            _,offset,axis=project_route(tile,xy);self.axis[i]=axis
            c=self.control.commands.adapters[i].c if self.control.commands.adapters[i] else None
            if c is None:continue
            self.offset[i]=offset-c.y_off
            if tile.kind=='legacy':self.feasible[i]=True;self.clearance[i]=self.cfg['observability']['sdf_clip_m'];continue
            result=(turn_feasible(tile,xy,margin_m=c.margin) if int(self.mode[i])==2 else passage_feasible(tile,xy,margin_m=c.margin))
            self.feasible[i]=result.feasible
            z=floor_height(tile,xy)
            active=[b for b in tile.obstacles if b.kind!='step' and b.center[2]+b.half[2]>z]
            radius=self.cfg['observability']['turn_radius_m'] if int(self.mode[i])==2 else self.cfg['rewards']['swept_width_m']/2
            self.clearance[i]=min([self.cfg['observability']['sdf_clip_m']]+[point_box_distance(xy,b)-radius for b in active])
            if bool(self.cylinder.enabled[i]):
                gap=float((d.root_pos_w[i,:2]-self.cylinder.position[i,:2]).norm())-float(self.cylinder.radius[i])-radius
                self.clearance[i]=min(float(self.clearance[i]),gap)
            stair[i]=route_hint(tile,xy)[0].segment_type in ('STAIR_UP','STAIR_DOWN','LANDING')
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
