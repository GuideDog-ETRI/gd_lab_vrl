"""Exact collision-point features for the temporary envelope's measured domain."""
import itertools
import torch

def collision_points174(local_positions,local_rotations,parts):
    points=[]
    for index,kind,offset,axis,size in parts:
        if kind=='sphere':vertices=offset[None]  # sphere rotation does not change its solid
        else:
            signs=local_positions.new_tensor(list(itertools.product((-1.,1.),repeat=3)))
            vertices=(signs*size)@axis.T+offset
        points.append(torch.einsum('...ij,pj->...pi',local_rotations[...,index,:,:],vertices)+local_positions[...,index,None,:])
    return torch.cat(points,-2)
