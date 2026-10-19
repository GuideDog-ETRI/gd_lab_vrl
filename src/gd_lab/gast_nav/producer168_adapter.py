"""Isaac snapshot adapter; all collider data stays INSIDE the sensor producer.

Caller must mirror the actual spawned colliders, including floor/steps, and
assert scene_complete. A mesh/unknown shape or incomplete inventory invalidates
that env. Link poses must be live PhysX poses from the SAME capture as cameras;
USD/Fabric-stale camera transforms are deliberately not consulted.
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import torch
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses,camera_rotation_matrix
from .producer168 import SensorPrimitives,BOX,CYLINDER,SPHERE

@dataclass(frozen=True)
class CollisionSpec168:
    link: str
    kind: int
    half: tuple
    xyz: tuple
    rotation: tuple


def load_self_collisions168(urdf_path):
    specs=[]
    for link in ET.parse(urdf_path).getroot().findall('link'):
        for collision in link.findall('collision'):
            origin=collision.find('origin');xyz=(0.,0.,0.);rpy=(0.,0.,0.)
            if origin is not None:
                xyz=tuple(map(float,origin.get('xyz','0 0 0').split()))
                rpy=tuple(map(float,origin.get('rpy','0 0 0').split()))
            r,p,y=rpy;cr,sr=math.cos(r),math.sin(r);cp,sp=math.cos(p),math.sin(p);cy,sy=math.cos(y),math.sin(y)
            rot=((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr),(sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr),(-sp,cp*sr,cp*cr))
            geom=collision.find('geometry');box=geom.find('box');cyl=geom.find('cylinder');sphere=geom.find('sphere')
            if box is not None:kind=BOX;half=tuple(float(v)/2 for v in box.get('size').split())
            elif cyl is not None:kind=CYLINDER;half=(float(cyl.get('radius')),)*2+(float(cyl.get('length'))/2,)
            elif sphere is not None:kind=SPHERE;half=(float(sphere.get('radius')),)*3
            else:raise ValueError('Unsupported self collider: '+link.get('name'))
            specs.append(CollisionSpec168(link.get('name'),kind,half,xyz,rot))
    if not specs:raise ValueError('empty self collision inventory')
    return tuple(specs)


def sample_frame168(pose):
    """TRUE sensor frame [N,4] world x,y,z,yaw -> sample-to-world rotation.

    Sensor simulation only. Do not pass noisy ODM1 pose here: geometry/camera
    must share a true local capture frame. Tag resulting frusta with a separate
    odometry sample pose when constructing FreeCapture (no GT pose is exported).
    """
    c,s=pose[:,3].cos(),pose[:,3].sin();z=torch.zeros_like(c);o=torch.ones_like(c)
    return torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(-1,3,3)


def vendor_cameras168(trunk_pos_w,trunk_quat_w,sample_pose):
    # sample_pose is the true sensor-internal frame; odometry is attached later.
    contract=load_camera_contract('vendor_new')
    pos,quat=mounted_camera_world_poses(trunk_pos_w,trunk_quat_w,contract)
    world_to_sample=sample_frame168(sample_pose).transpose(-1,-2)
    rotation=torch.einsum('nij,ncjk->ncik',world_to_sample,camera_rotation_matrix(quat))
    translation=torch.einsum('nij,ncj->nci',world_to_sample,pos-sample_pose[:,None,:3])
    k=pos.new_tensor([[contract.fx,0.,contract.width/2],[0.,contract.fy,contract.height/2],[0.,0.,1.]]).expand(4,-1,-1).clone()
    return k,rotation,translation


def d4xx_blocks168(width=80,height=45,block_width=8,block_height=8,device='cpu'):
    return torch.tensor([(c,x,y,min(x+block_width,width),min(y+block_height,height)) for c in range(4) for y in range(0,height,block_height) for x in range(0,width,block_width)],device=device,dtype=torch.long)


def primitive_snapshot168(scene,body_pos_w,body_quat_w,body_names,specs,sample_pose,*,scene_capture_id,body_capture_id,camera_capture_id):
    """Mirror scene primitives + EVERY URDF collision into a sensor snapshot.

    body_pos_w/body_quat_w must be LIVE LINK-frame poses (body_link_pos_w /
    body_link_quat_w), not inertial COM poses. sample_pose is the TRUE yaw capture
    frame, not noisy ODM1. The mapper attaches odometry independently later.
    scene is SensorPrimitives in WORLD coordinates, containing floor, static
    boxes/steps/walls and dynamic cylinders at the current capture (not metadata
    from a previous physics step). No floor/support is invented by this adapter.
    Capture ids [N] include reset identity; exact equality is required.
    """
    if not torch.equal(scene_capture_id,body_capture_id) or not torch.equal(scene_capture_id,camera_capture_id):raise ValueError('unsynchronised capture')
    n=len(sample_pose)
    if scene_capture_id.shape!=(n,) or body_capture_id.shape!=(n,) or camera_capture_id.shape!=(n,):raise ValueError('capture id shape')
    if any(x.dtype!=torch.int64 for x in (scene_capture_id,body_capture_id,camera_capture_id)):raise ValueError('capture id dtype')
    if body_pos_w.shape!=(n,len(body_names),3) or body_quat_w.shape!=(n,len(body_names),4) or sample_pose.shape!=(n,4):raise ValueError('live link/sample pose shape')
    if not bool(torch.isfinite(body_pos_w).all()&torch.isfinite(body_quat_w).all()&torch.isfinite(sample_pose).all()) or bool((body_quat_w.norm(dim=-1)<1e-6).any()):raise ValueError('invalid live link pose')
    if len(set(body_names))!=len(body_names):raise ValueError('duplicate body names')
    ids=torch.tensor([body_names.index(s.link) for s in specs],device=body_pos_w.device)
    r=camera_rotation_matrix(body_quat_w[:,ids]);local=body_pos_w.new_tensor([s.xyz for s in specs])
    self_pos=body_pos_w[:,ids]+torch.einsum('npij,pj->npi',r,local)
    self_rot=r@body_pos_w.new_tensor([s.rotation for s in specs])
    n=len(sample_pose);inv=sample_frame168(sample_pose).transpose(-1,-2)
    centres=torch.cat((scene.centres,self_pos),1);rotations=torch.cat((scene.rotations,self_rot),1)
    centres=torch.einsum('nij,npj->npi',inv,centres-sample_pose[:,None,:3])
    rotations=torch.einsum('nij,npjk->npik',inv,rotations)
    half=torch.cat((scene.half,body_pos_w.new_tensor([s.half for s in specs])[None].expand(n,-1,-1)),1)
    kind=torch.cat((scene.kind,torch.tensor([s.kind for s in specs],device=ids.device)[None].expand(n,-1)),1)
    active=torch.cat((scene.active,torch.ones((n,len(specs)),device=ids.device,dtype=torch.bool)),1)
    trunk=torch.cat((torch.zeros_like(scene.active),torch.tensor([s.link=='trunk' for s in specs],device=ids.device)[None].expand(n,-1)),1)
    return SensorPrimitives(centres,rotations,half,kind,active,scene.complete,trunk)
