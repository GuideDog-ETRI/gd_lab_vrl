"""Sensor-only live primitive bridge for UPDATE168 inference diagnostics.

Inventory comes from the exact box construction used for terrain meshes, not
floor-name inference. Unknown tiles/border near a query invalidate that env.
Default current-capture path remains available; UPDATE169 explicitly opts into
bounded static-C temporal evidence and an audited primitive-only fixture.
"""
from pathlib import Path
import math
import time
import torch
import yaml
from .producer168 import SensorPrimitives,produce_blocks168
from .producer168_adapter import load_self_collisions168,primitive_snapshot168,vendor_cameras168,d4xx_blocks168
from .free_evidence167 import FreeCapture,FreeMeasurement,FreeEvidenceConfig
from .obstacle_map import ray_evidence
from .ray_geometry import resolve_first_hits
from .map_types import WIDE_GRID,NEAR_GRID
from .near_clearance import extended_near_grid

class IsaacProducer168:
    def __init__(self,nav):
        self.nav=nav;self.env=env=nav.env;dev=env.device
        root=Path(__file__).resolve().parents[3]
        self.cfg=yaml.safe_load((root/'configs/gast_nav/producer168_training_c.yaml').read_text())['producer168']
        self.specs=load_self_collisions168(root/'assets/robots/rbq10/urdf/rbq10_simple.urdf')
        self.blocks=d4xx_blocks168(device=dev,block_width=self.cfg['block_width'],block_height=self.cfg['block_height'])
        self.settings=FreeEvidenceConfig(max_captures=1)  # deployment/default current-only
        self.temporal=None;self.motion_time=None
        temporal_file=nav.cfg.get('temporal169_config')
        if temporal_file:
            from .temporal169 import TemporalEvidence169
            temporal_cfg=yaml.safe_load((root/temporal_file).read_text())
            self.temporal=TemporalEvidence169(env.num_envs,dev,temporal_cfg)
            self.settings=self.temporal.settings
        gen=env.scene.terrain.cfg.terrain_generator;origins=env.scene.terrain.terrain_origins
        self.primitive169=bool(nav.cfg.get('primitive_c169',False))
        self.stats={};self.last_time=None
        if self.primitive169:
            from .isaac_closure169 import audit_closed_scene169
            self.audit169=audit_closed_scene169(env)
            return
        centres=[];half=[];rot=[];unknown=[]
        for key,meta in gen.nav_scene_grid.items():
            origin=origins[key[0],key[1]].detach().cpu().tolist()
            if key not in gen.nav_sensor_grid:
                unknown.append([(key[0]+.5-gen.num_rows/2)*gen.size[0],(key[1]+.5-gen.num_cols/2)*gen.size[1]])
                continue
            for b in gen.nav_sensor_grid[key]:
                centres.append([b.center[i]+origin[i] for i in range(3)])
                half.append(b.half);c,s=math.cos(b.yaw),math.sin(b.yaw)
                rot.append([[c,-s,0],[s,c,0],[0,0,1]])
        if not centres:raise ValueError('no complete generated primitive inventory')
        self.centres=torch.tensor(centres,device=dev);self.half=torch.tensor(half,device=dev);self.rot=torch.tensor(rot,device=dev)
        self.unknown=torch.tensor(unknown,device=dev).reshape(-1,2)
        self.tile_half=torch.tensor(gen.size,device=dev)/2
        # The unrepresented border and outside terrain must never be certified.
        self.hi=self.tile_half*torch.tensor([gen.num_rows,gen.num_cols],device=dev)
        self.lo=-self.hi
        self.stats={};self.last_time=None

    def capture(self,*,rays,pose,time_ms,wide_support,fine_support,epoch,capsules,config):
        nav=self.nav;env=self.env;n=env.num_envs;d=nav.robot.data
        start=time.perf_counter()
        # True sensor frame only; no noisy odometry is used for scene transforms.
        ids=time_ms+epoch*1000000000
        k,r,t=vendor_cameras168(d.body_link_pos_w[:,nav.robot.body_names.index('trunk')],
             d.body_link_quat_w[:,nav.robot.body_names.index('trunk')],pose)
        linear=self.temporal.linear if self.temporal else 5.
        angular=self.temporal.angular if self.temporal else 10.
        if self.primitive169:
            from .inventory169 import primitive_c_fixture169,primitive_c_inventory169,self_collision_bounds_xy169
            from .free_evidence167 import _rotation
            camera_world=torch.einsum('nij,ncj->nci',_rotation(pose[:,3]),t)+pose[:,None,:3]
            bounds=self_collision_bounds_xy169(d.body_link_pos_w,d.body_link_quat_w,nav.robot.body_names,self.specs)
            gen=env.scene.terrain.cfg.terrain_generator
            scene=primitive_c_inventory169(primitive_c_fixture169(),env.scene.env_origins,camera_world,k,
                max_m=self.cfg['max_m'],all_body_bounds_xy_w=bounds,unique_tile_ids=nav.tile_ids,
                sealed=gen.primitive_c169_sealed,no_external_colliders=self.audit169['closed'])
            complete=scene.complete & (d.root_lin_vel_w.norm(dim=-1)<=linear)&(d.root_ang_vel_w[:,2].abs()<=angular)
            from dataclasses import replace
            scene=replace(scene,complete=complete)
        else:
            # Conservative sphere enclosing all optical-z max-range image corners,
            # including mounts. Unknown colliders anywhere in this sphere close the env.
            contract_width,contract_height=80,45
            slopes=torch.stack((torch.maximum(k[:,0,2],contract_width-k[:,0,2])/k[:,0,0],
                                torch.maximum(k[:,1,2],contract_height-k[:,1,2])/k[:,1,1]),-1)
            reach=self.cfg['max_m']*torch.sqrt(1+slopes.square().sum(-1)).amax()+t.norm(dim=-1).amax(1)
            complete=((pose[:,:2]-reach[:,None]>=self.lo)&(pose[:,:2]+reach[:,None]<=self.hi)).all(-1)
            if len(self.unknown):
                distance=(pose[:,None,:2]-self.unknown[None]).abs()-self.tile_half
                complete &= (distance.clamp_min(0).norm(dim=-1)>reach[:,None]).all(-1)
            # Diagnostic operating bounds for the 40ms P2 delay; any observed
            # violation closes the capture, rather than reducing the error budget.
            linear=self.temporal.linear if self.temporal else 5.
            angular=self.temporal.angular if self.temporal else 10.
            complete &= (d.root_lin_vel_w.norm(dim=-1)<=linear)&(d.root_ang_vel_w[:,2].abs()<=angular)
            centres=self.centres[None].expand(n,-1,-1)
            half=self.half[None].expand(n,-1,-1);rotation=self.rot[None].expand(n,-1,-1,-1)
            active=torch.ones(centres.shape[:2],device=env.device,dtype=torch.bool)
            kind=torch.zeros_like(active,dtype=torch.long)
            # Own environment cylinder, including the actual live collider pose.
            cpos=env.scene['nav_cylinder'].data.root_pos_w[:,None]
            centres=torch.cat((centres,cpos),1)
            half=torch.cat((half,torch.stack((nav.cylinder.radius,nav.cylinder.radius,nav.cylinder.height/2),-1)[:,None]),1)
            from gd_lab.core.camera_geometry import camera_rotation_matrix
            rotation=torch.cat((rotation,camera_rotation_matrix(env.scene['nav_cylinder'].data.root_quat_w)[:,None]),1)
            kind=torch.cat((kind,torch.ones((n,1),device=env.device,dtype=torch.long)),1)
            active=torch.cat((active,nav.cylinder.enabled[:,None]),1)
            scene=SensorPrimitives(centres,rotation,half,kind,active,complete)
        primitive=primitive_snapshot168(scene,d.body_link_pos_w,d.body_link_quat_w,nav.robot.body_names,self.specs,pose,
            scene_capture_id=ids,body_capture_id=ids,camera_capture_id=ids)
        measured=produce_blocks168(primitive,k,r,t,self.blocks,backend=self.cfg['backend'],
            near_m=self.cfg['near_m'],max_m=self.cfg['max_m'],self_trunk_aperture=self.cfg['self_trunk_aperture'],context=self.cfg['context'])
        self.stats=dict(complete_envs=int(complete.sum()),envs=n,valid_blocks=int(measured.volume_valid.sum()),
                       blocks=measured.volume_valid.numel(),current_only=self.temporal is None,delay_linear_bound_mps=linear,delay_yaw_bound_radps=angular)
        torch.cuda.synchronize()
        self.stats['producer_wall_s']=time.perf_counter()-start
        proof_start=time.perf_counter()
        self.last_measured=measured;self.last_time=time_ms.clone()
        resolved=resolve_first_hits(rays,capsules=capsules,config=config)
        self.last_resolved=resolved;self.last_pose=pose.clone();self.last_epoch=epoch.clone()
        if self.temporal:
            self.temporal.push(measured.frusta,pose,(wide_support,fine_support),time_ms,epoch,dynamic_enabled=nav.cylinder.enabled)
            return self.query(pose=pose,time_ms=time_ms,epoch=epoch,config=config)
        results=[]
        for support,grid in ((wide_support,WIDE_GRID),(fine_support,extended_near_grid()[0])):
            measurement=None
            if bool(measured.volume_valid.any()):
                cap=FreeCapture(measured.frusta,pose,support,time_ms,epoch)
                measurement=FreeMeasurement((cap,),pose,time_ms,epoch,pose.new_zeros(n,1),
                    pose.new_full((n,),5.),pose.new_full((n,),10.),self.settings)
            state,_,age=ray_evidence(resolved,support,grid=grid,config=config,measurement=measurement,certified=True,return_age=True)
            results.append((state,age))
        torch.cuda.synchronize()
        self.stats['proof_wall_s']=time.perf_counter()-proof_start
        pad=extended_near_grid()[1]
        return results[0],tuple(v[...,pad:pad+NEAR_GRID.ny,pad:pad+NEAR_GRID.nx] for v in results[1])

    def sample_motion(self):
        if self.temporal is None:return
        now=self.env._sim_step_counter*self.env.physics_dt
        if self.motion_time is not None and now<=self.motion_time:return
        dt=0. if self.motion_time is None else now-self.motion_time
        d=self.nav.robot.data
        self.temporal.observe(self.nav.pose(),d.root_lin_vel_w.norm(dim=-1),d.root_ang_vel_w[:,2],dt)
        self.motion_time=now

    def reset(self,ids):
        if self.temporal:self.temporal.reset(ids)

    def query(self,*,pose,time_ms,epoch,config):
        from .temporal169 import current_support169,transform_returns169
        start=time.perf_counter();results=[];supports=[];measurements=[]
        valid=(self.last_epoch==epoch)&((time_ms-self.last_time)<=round(config.ttl_s*1000))
        resolved=transform_returns169(self.last_resolved,self.last_pose,pose,valid)
        for index,grid in enumerate((WIDE_GRID,extended_near_grid()[0])):
            history=self.temporal.histories[index]
            support=current_support169(history.captures[0],pose,time_ms,epoch,grid,config)
            measurement=self.temporal.measurement(index,pose,time_ms,epoch)
            measurements.append(measurement)
            state,_,age=ray_evidence(resolved,support,grid=grid,config=config,measurement=measurement,certified=True,return_age=True)
            results.append((state,age));supports.append(support)
        self.last_query169=dict(resolved=resolved,supports=supports,measurements=measurements)
        pad=extended_near_grid()[1]
        self.wide_support_known=torch.isfinite(supports[0])
        self.near_support_known=torch.isfinite(supports[1][...,pad:pad+NEAR_GRID.ny,pad:pad+NEAR_GRID.nx])
        torch.cuda.synchronize()
        self.stats.update(proof_wall_s=time.perf_counter()-start,history_captures=len(self.temporal.anchors),
            bound_violation_samples=int(self.temporal.violations.sum()),
            query_age_ms=int((time_ms-self.last_time).max()),
            query_time_ms=int(time_ms.max()))
        return results[0],tuple(v[...,pad:pad+NEAR_GRID.ny,pad:pad+NEAR_GRID.nx] for v in results[1])
