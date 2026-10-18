"""UPDATE162 V0 inference helpers. No Nav policy, FSM or obstacle map input."""
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from .terrains import KINDS, build_scene, META_KEY


def cases(settings):
    return [dict(kind=k,seed=s,structures=b,mode=m) for k in KINDS
            for s in settings['seeds'] for b in (False,True) for m in ('gt','visible')]


def paired_mesh(case,difficulty,size=(16.,12.)):
    """Remove lateral/overhead solids only; retain every floor, gap and step."""
    meshes,origin,meta=build_scene(case['kind'],difficulty,SimpleNamespace(size=size),seed=case['seed'])
    if not case['structures']:
        removed=[b for b in meta.obstacles if b.kind!='step']
        meshes=[m for m in meshes if not any(np.allclose(m.centroid,origin+np.array(b.center),atol=1e-8,rtol=0) for b in removed)]
        meta=replace(meta,obstacles=tuple(b for b in meta.obstacles if b.kind=='step'))
    return meshes,origin,meta


def generator_class():
    import trimesh
    from gd_lab.mdp.terrains.gap_metadata_generator import GapMetadataTerrainGenerator
    class V0Generator(GapMetadataTerrainGenerator):
        def __init__(self,cfg,device='cpu'):
            self.v0_grid={}
            super().__init__(cfg,device)
            cfg.v0_grid=self.v0_grid
        def _get_terrain_mesh(self,difficulty,cfg):
            meshes,origin,meta=paired_mesh(cfg.v0_case,difficulty,cfg.size)
            mesh=trimesh.util.concatenate(meshes)
            shift=np.array([-cfg.size[0]/2,-cfg.size[1]/2,0.])
            mesh.apply_translation(shift);mesh.metadata[META_KEY]=meta
            return mesh,origin+shift
        def _add_sub_terrain(self,mesh,origin,row,col,cfg):
            self.v0_grid[(row,col)]=(cfg.v0_case,mesh.metadata[META_KEY])
            return super()._add_sub_terrain(mesh,origin,row,col,cfg)
    return V0Generator


def bindings(env):
    t=env.scene.terrain
    return [t.cfg.terrain_generator.v0_grid[(int(r),int(c))] for r,c in zip(t.terrain_levels,t.terrain_types)]


def reset_nominal(env,env_ids):
    import torch
    from .scene_geometry import floor_height
    robot=env.scene['robot'];ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    state=robot.data.default_root_state[ids].clone();state[:,7:]=0
    bound=bindings(env)
    for j,i in enumerate(ids.tolist()):
        meta=bound[i][1];a,b=meta.route_polyline[:2];yaw=math.atan2(b[1]-a[1],b[0]-a[0])
        state[j,:3]+=env.scene.env_origins[i]
        state[j,0:2]+=state.new_tensor(a)
        state[j,2]+=floor_height(meta,a)
        state[j,3:7]=state.new_tensor([math.cos(yaw/2),0,0,math.sin(yaw/2)])
    robot.write_root_pose_to_sim(state[:,:7],env_ids=ids)
    robot.write_root_velocity_to_sim(state[:,7:],env_ids=ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids],torch.zeros_like(robot.data.default_joint_vel[ids]),env_ids=ids)


def command_class():
    from isaaclab.envs.mdp import UniformVelocityCommand
    from .scene_types import project_route
    class LegacyRouteCommand(UniformVelocityCommand):
        def _update_command(self):
            import torch
            xy=(self.robot.data.root_pos_w-self._env.scene.env_origins)[:,:2].detach().cpu().tolist()
            bound=bindings(self._env)
            headings=self.robot.data.heading_w
            settings=self._env.cfg.v0_settings
            self.vel_command_b[:]=0
            for i,(_,meta) in enumerate(bound):
                s,_,yaw=project_route(meta,xy[i]);err=(yaw-float(headings[i])+math.pi)%(2*math.pi)-math.pi
                if s<meta.route_length-settings['completion_margin_m']:
                    self.vel_command_b[i,0]=settings['speed_m_s']
                    self.vel_command_b[i,2]=max(-settings['yaw_limit_rad_s'],min(settings['yaw_limit_rad_s'],settings['yaw_gain']*err))
    return LegacyRouteCommand


def terrain_class():
    import torch
    from gd_lab.gast.rayir import (RaycastIrTerrain,RayIrHistory,scan_normals,IR_AMBIENT,IR_GAIN,IR_R0,IR_FALLOFF,IR_FLOOR,
        mounted_camera_world_poses,leg_capsules,raycast_visible_points)
    class PairedTerrain(RaycastIrTerrain):
        def __init__(self,cfg,env):
            super().__init__(cfg,env)
            self.gt_history=RayIrHistory(env);self.visible_history=RayIrHistory(env)
            self.gt_held=torch.zeros_like(self.held);self.visible_held=torch.zeros_like(self.held)
        def reset(self,env_ids=None):
            super().reset(env_ids)
            if hasattr(self,'gt_history'):
                self.gt_history.reset(env_ids);self.visible_history.reset(env_ids)
                ids=slice(None) if env_ids is None else env_ids
                self.gt_held[ids]=0;self.visible_held[ids]=0
        def __call__(self,env):
            step=env.common_step_counter
            if self.last_step==step and self.cached is not None:return self.cached.clone()
            refresh=step-self.capture_step>=self.period_steps
            if refresh.any():
                scan=env.scene['height_scanner'].data;points=scan.ray_hits_w
                finite=torch.isfinite(points).all(-1);robot=env.scene['robot'];trunk=robot.body_names.index('trunk')
                pos,rot=mounted_camera_world_poses(robot.data.body_pos_w[:,trunk],robot.data.body_quat_w[:,trunk],self.contract)
                occ=leg_capsules(robot);visible=torch.zeros_like(finite)
                distance=torch.full_like(points[...,0],float('inf'));light=torch.zeros_like(points)
                for c in range(pos.shape[1]):
                    visible|=raycast_visible_points(points,pos[:,c],rot[:,c],self.intrinsic,
                        (self.contract.height,self.contract.width),self.contract.depth_clip,self.cast,occluders=occ)
                    vec=pos[:,c,None]-points;r=vec.norm(dim=-1);closer=r<distance
                    distance=torch.where(closer,r,distance);light=torch.where(closer[...,None],vec/r.clamp_min(1e-6)[...,None],light)
                visible &= finite
                h=(scan.pos_w[:,2,None]-points[...,2]-.5).clamp(-1,1)*5
                cosine=(scan_normals(points)*light).sum(-1).clamp_min(0)
                intensity=(IR_AMBIENT+IR_GAIN*env.cfg.v0_settings['ir_reflectance']*cosine*(IR_R0/distance.clamp(.15,1e3))**IR_FALLOFF).clamp(0,1)
                ir=(IR_FLOOR+(1-IR_FLOOR)*intensity).nan_to_num(0.)
                pack=lambda mask:torch.cat((torch.where(mask,h,0.),mask.float(),torch.where(mask,ir,0.)),-1)
                self.gt_held[refresh]=pack(finite)[refresh];self.visible_held[refresh]=pack(visible)[refresh];self.capture_step[refresh]=step
            self.gt_packed=self.gt_history.update(env,self.gt_held);self.visible_packed=self.visible_history.update(env,self.visible_held)
            use_gt=torch.tensor([c['mode']=='gt' for c,_ in bindings(env)],device=env.device)[:,None]
            self.cached=torch.where(use_gt,self.gt_held[:,:374],self.visible_held[:,:374])
            env._gast_history=torch.where(use_gt,self.gt_packed,self.visible_packed)
            self.last_step=step
            return self.cached.clone()
    return PairedTerrain


def make_cfg(settings):
    from isaaclab.managers import EventTermCfg,ObservationTermCfg,RewardTermCfg,TerminationTermCfg
    from isaaclab.envs.mdp import time_out
    from isaaclab.utils import configclass
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
    from gd_lab.gast.tasks import GastRayIrV21Gap50TeacherCfg
    import torch
    @configclass
    class CaseCfg(SubTerrainBaseCfg):
        v0_case:dict=None
    cfg=GastRayIrV21Gap50TeacherCfg();cfg.v0_settings=settings
    plan=cases(settings);cfg.scene.num_envs=len(plan);cfg.scene.terrain.max_init_terrain_level=0
    gen=cfg.scene.terrain.terrain_generator
    gen.class_type=generator_class();gen.num_rows=1;gen.num_cols=len(plan);gen.size=(16.,12.)
    gen.use_cache=False;gen.curriculum=True;gen.difficulty_range=(settings['difficulty'],settings['difficulty']);gen.seed=162
    gen.sub_terrains={f'v0_{i}':CaseCfg(function=lambda d,c:None,proportion=1.,size=gen.size,v0_case=c) for i,c in enumerate(plan)}
    for group in (cfg.events,cfg.curriculum,cfg.rewards,cfg.terminations):
        for key in list(vars(group)):
            if not key.startswith('_'):setattr(group,key,None)
    cfg.events.v0_reset=EventTermCfg(func=reset_nominal,mode='reset')
    cfg.rewards.v0_zero=RewardTermCfg(func=lambda e:torch.zeros(e.num_envs,device=e.device),weight=1.)
    cfg.terminations.time_out=TerminationTermCfg(func=time_out,time_out=True)
    cfg.episode_length_s=settings['duration_s']+1
    cfg.commands.base_velocity.class_type=command_class();cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.resampling_time_range=(1000.,1000.)
    cfg.observations.policy.enable_corruption=False
    cfg.observations.terrain.camera_visible=ObservationTermCfg(func=terrain_class())
    return cfg
