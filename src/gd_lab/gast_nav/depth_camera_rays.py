"""Measured first-return rays from the four vendor_new lower depth cameras.

Uses the same live mounted poses, intrinsics, near plane, terrain mesh and
leg/trunk/foot occluders as BIVT raycast camera visibility. No target/GT floor
heights are synthesized. Unknown and clipped returns are not surface hits.
"""
import torch
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses,camera_rotation_matrix
from gd_lab.teachers.bivt.raycast_visibility import contract_intrinsic,warp_mesh_cast
from .map_types import RayBatch,DynamicCapsules
from .ray_geometry import capsule_intersections,ray_box_interval
from .obs_terms import world_to_sample


class LowerDepthRays:
    def __init__(self,env,settings):
        self.diagnostic_unoccluded=settings.get('diagnostic_unoccluded',False)
        self.unoccluded_rays=None
        self.contract=load_camera_contract(settings['profile'])
        if self.contract.profile!='vendor_new':raise ValueError('C2.3b requires vendor_new')
        self.env=env;self.robot=env.scene['robot'];self.trunk=self.robot.body_names.index('trunk')
        self.feet=[self.robot.body_names.index(f'{leg}_foot') for leg in ('FL','FR','RL','RR')]
        self.cast=warp_mesh_cast(env.scene['height_scanner'].meshes[env.scene['height_scanner'].cfg.mesh_prim_paths[0]])
        self.unit,self.length=camera_pixel_rays(self.contract,env.device,settings.get('sampling_scale',1))
        self.near=self.contract.depth_clip[0]*self.length.repeat(4)[None]
        self.far=self.contract.depth_clip[1]*self.length.repeat(4)[None]
        self.max_distance=float(self.far.max())

    def capture(self,pose):
        d=self.robot.data;contract=self.contract
        positions,quats=mounted_camera_world_poses(d.body_pos_w[:,self.trunk],d.body_quat_w[:,self.trunk],contract)
        rot=camera_rotation_matrix(quats)
        direction=torch.einsum('ncij,pj->ncpi',rot,self.unit)
        origin=positions[:,:,None].expand_as(direction)
        n=len(origin);origin=origin.reshape(n,-1,3);direction=direction.reshape(n,-1,3)
        near,far=self.near,self.far
        hit=self.cast(origin.reshape(-1,3).contiguous(),direction.reshape(-1,3).contiguous(),self.max_distance).reshape(n,-1)
        # Actual self intersection: legs from optical centre, body/feet from near plane.
        from gd_lab.teachers.bivt.raycast_terrain import leg_capsules
        a,b,r=leg_capsules(self.robot)
        leg_hit=capsule_intersections(origin,direction,DynamicCapsules(a,b,r[None].expand(n,-1)),eps=1e-6).amin(-1)
        shifted=origin+direction*near[...,None]
        trunk_rot=camera_rotation_matrix(d.body_quat_w[:,self.trunk])
        local_o=(shifted-d.body_pos_w[:,self.trunk,None])@trunk_rot;local_d=direction@trunk_rot
        half=origin.new_tensor([.4,.10,.06])
        entry,exit=ray_box_interval(local_o,local_d,-half,half,1e-6)
        box=torch.where((exit>=entry.clamp_min(0))&(exit>1e-5),entry.clamp_min(0)+near,torch.inf)
        feet=d.body_pos_w[:,self.feet]
        foot=capsule_intersections(shifted,direction,DynamicCapsules(feet,feet,origin.new_full((n,4),.03)),eps=1e-6).amin(-1)+near
        self_distance=torch.minimum(leg_hit,torch.minimum(box,foot))
        distance,non_surface=clip_depth_returns(hit,self_distance,near,far)
        if self.diagnostic_unoccluded:
            # Diagnostic counterfactual only. Never supplied to policy maps.
            raw_distance,raw_clip=clip_depth_returns(hit,torch.full_like(hit,torch.inf),near,far)
            self.unoccluded_rays=RayBatch(world_to_sample(shifted,pose),world_to_sample(direction,pose,vectors=True),raw_distance,raw_clip)
        return RayBatch(world_to_sample(shifted,pose),world_to_sample(direction,pose,vectors=True),distance,non_surface)


def camera_pixel_rays(contract,device,scale=1):
    """Pinhole samples at up to the vendor source resolution, fixed FoV."""
    if type(scale) is not int or not 1<=scale<=min(contract.source_resolution[0]//contract.width,contract.source_resolution[1]//contract.height):
        raise ValueError('invalid depth sampling_scale')
    k=contract_intrinsic(contract,device)
    y,x=torch.meshgrid(torch.arange(contract.height*scale,device=device)/scale,
                       torch.arange(contract.width*scale,device=device)/scale,indexing='ij')
    rays=torch.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],torch.ones_like(x)),-1).reshape(-1,3)
    length=rays.norm(dim=-1)
    return rays/length[:,None],length


def clip_depth_returns(hit,self_distance,near,far):
    """No carving before the optical near plane or beyond first/self/far stop."""
    stop=torch.minimum(hit,torch.minimum(self_distance,far))
    non_surface=(self_distance<=hit)|(far<=hit)|(stop<=near)
    return (stop-near).clamp_min(0),non_surface
