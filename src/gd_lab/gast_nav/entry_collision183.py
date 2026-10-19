"""Exact primitive overlap rejection for virtual entry on easy-C boxes.

Touching is not positive volume overlap; no configurable penetration exemption.
Unsupported scene shapes invalidate the pose instead of being ignored.
"""
import torch


def overlapping_parts183(body,scene):
 n,b,_=body.centres.shape;p=scene.centres.shape[1]
 delta=body.centres.double()[:,:,None]-scene.centres.double()[:,None]
 ar=scene.rotations.double()[:,None].expand(n,b,p,3,3);br=body.rotations.double()[:,:,None].expand(n,b,p,3,3)
 ah=scene.half.double()[:,None].expand(n,b,p,3);bh=body.half.double()[:,:,None].expand(n,b,p,3)
 # Sphere vs OBB: closest point in scene-local frame.
 local=torch.einsum('nbpij,nbpj->nbpi',ar.transpose(-1,-2),delta)
 closest=torch.maximum(-ah,torch.minimum(local,ah))
 sphere=(local-closest).square().sum(-1)<bh[...,0].square()
 # Full 15-axis OBB SAT. Zero cross product is an omitted redundant axis.
 axes=[ar[...,i] for i in range(3)]+[br[...,i] for i in range(3)]
 axes += [torch.linalg.cross(ar[...,i],br[...,j],dim=-1) for i in range(3) for j in range(3)]
 box=torch.ones(n,b,p,device=delta.device,dtype=torch.bool)
 for axis in axes:
  radius_a=(torch.einsum('nbpji,nbpj->nbpi',ar,axis).abs()*ah).sum(-1)
  radius_b=(torch.einsum('nbpji,nbpj->nbpi',br,axis).abs()*bh).sum(-1)
  separation=(delta*axis).sum(-1).abs()
  box &= (axis.square().sum(-1)<1e-16)|(separation<radius_a+radius_b)
 overlap=torch.where(body.kind[:,:,None]==2,sphere,box)
 unsupported=(scene.kind[:,None]!=0)|((body.kind[:,:,None]!=0)&(body.kind[:,:,None]!=2))
 return (overlap|unsupported)&body.active[:,:,None]&scene.active[:,None]
