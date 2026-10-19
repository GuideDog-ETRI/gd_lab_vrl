"""C2.3b current collision footprint; no environment geometry or map edits."""
from pathlib import Path
import itertools
import math
import xml.etree.ElementTree as ET
import torch
from gd_lab.core.camera_geometry import camera_rotation_matrix


def _rpy_matrix(rpy):
    r,p,y=rpy;cr,sr=math.cos(r),math.sin(r);cp,sp=math.cos(p),math.sin(p);cy,sy=math.cos(y),math.sin(y)
    return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
            [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]


def projected_box_mask(xy, centre, axes, half, margin):
    """Exact horizontal projection of an OBB plus Euclidean margin.

    axes[N,2,3] are projected local basis columns. Projection is a zonotope;
    its supporting normals are perpendicular to those three generators.
    """
    p=xy.reshape(1,-1,2)-centre[:,None,:]
    generators=axes*half[None,None,:]
    normals=torch.stack((-generators[:,1],generators[:,0]),-1)  # N,3,2
    bound=torch.einsum('nki,nij->nkj',normals,generators).abs().sum(-1)
    inside=(torch.einsum('npi,nki->npk',p,normals).abs()<=bound[:,None]+1e-7).all(-1)
    values=list(itertools.product((-1.,1.),repeat=3));signs=xy.new_tensor(values)
    vertices=torch.einsum('nij,vj->nvi',generators,signs)
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if sum(a!=b for a,b in zip(values[i],values[j]))==1]
    a=vertices[:,[i for i,j in edges]];b=vertices[:,[j for i,j in edges]];d=b-a
    delta=p[:,:,None]-a[:,None]
    t=(delta*d[:,None]).sum(-1)/d.square().sum(-1)[:,None].clamp_min(1e-12)
    distance=(delta-t.clamp(0,1)[...,None]*d[:,None]).square().sum(-1).amin(-1)
    return (inside|(distance<=margin*margin)).reshape(centre.shape[0],*xy.shape[:-1])


class CollisionFootprint:
    def __init__(self, body_names, device, urdf=None):
        path=Path(urdf) if urdf else Path(__file__).resolve().parents[3]/'assets/robots/rbq10/urdf/rbq10_simple.urdf'
        self.parts=[]
        for link in ET.parse(path).getroot().findall('link'):
            for c in link.findall('collision'):
                origin=c.find('origin');xyz=[float(v) for v in origin.get('xyz','0 0 0').split()];rpy=[float(v) for v in origin.get('rpy','0 0 0').split()]
                geom=c.find('geometry');box=geom.find('box');sphere=geom.find('sphere')
                if box is not None:kind='box';size=[float(v)/2 for v in box.get('size').split()]
                elif sphere is not None:kind='sphere';size=float(sphere.get('radius'))
                else:raise ValueError('unsupported collision shape')
                self.parts.append((body_names.index(link.get('name')),kind,torch.tensor(xyz,device=device),
                    torch.tensor(_rpy_matrix(rpy),device=device),torch.tensor(size,device=device)))

    def mask(self, xy, body_pos, body_quat, sample_pose, margin):
        rotation=camera_rotation_matrix(body_quat)
        c,s=sample_pose[:,3].cos(),sample_pose[:,3].sin()
        yaw=torch.stack((c,s,torch.zeros_like(c),-s,c,torch.zeros_like(c)),-1).reshape(-1,2,3)
        out=torch.zeros((len(body_pos),*xy.shape[:-1]),dtype=torch.bool,device=xy.device)
        for index,kind,offset,local,size in self.parts:
            centre=body_pos[:,index]+torch.einsum('nij,j->ni',rotation[:,index],offset)
            centre=torch.einsum('nij,nj->ni',yaw,centre-sample_pose[:,:3])
            if kind=='sphere':out|=(xy[None]-centre[:,None,None]).square().sum(-1)<=(size+margin)**2
            else:
                axes=yaw@rotation[:,index]@local
                out|=projected_box_mask(xy,centre,axes,size,margin)
        return out
