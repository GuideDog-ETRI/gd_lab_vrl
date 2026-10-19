"""D-5 supervisor helpers. Contact evidence never creates obstacle FREE."""
from dataclasses import replace
import torch


def contact_support179(observation, points, valid, grid):
    if points.shape[:2]!=valid.shape or points.shape[-1]!=3:
        raise ValueError('contact shape')
    support=observation.support.clone();age=observation.support_age_s.clone()
    finite=valid & torch.isfinite(points).all(-1)
    x=torch.floor((torch.nan_to_num(points[...,0])-grid.x0)/grid.res+.5).long()
    y=torch.floor((torch.nan_to_num(points[...,1])-grid.y0)/grid.res+.5).long()
    finite &= (x>=0)&(x<grid.nx)&(y>=0)&(y<grid.ny)
    for j in range(points.shape[1]):
        ids=finite[:,j].nonzero().flatten()
        support[ids,y[ids,j],x[ids,j]]=points[ids,j,2]
        age[ids,y[ids,j],x[ids,j]]=0.
    return replace(observation,support=support,support_age_s=age)


def current_body_cells179(footprint, grid, body_pos, body_quat, pose, margin_m):
    """Exclude only full cells inside one convex projected collision + margin.

    A centre-only test would also erase unoccupied future space at the cell edge.
    """
    xy=grid.xy(device=pose.device,dtype=pose.dtype)
    out=torch.zeros((len(pose),grid.ny,grid.nx),device=pose.device,dtype=torch.bool)
    part=object.__new__(type(footprint))
    for primitive in footprint.parts:
        part.parts=[primitive]
        contained=torch.ones_like(out)
        for dx,dy in ((-1,-1),(-1,1),(1,-1),(1,1)):
            contained &= part.mask(xy+xy.new_tensor([dx,dy])*grid.res/2,body_pos,body_quat,pose,margin_m)
        out |= contained
    return out


def stable_contact179(force_history, weight_n, physics_dt):
    """Strict >10% body weight throughout at least 50 ms of measured history."""
    import math
    if not math.isfinite(physics_dt) or physics_dt<=0 or force_history.ndim!=3 or force_history.shape[-1]!=3:
        raise ValueError('contact history [N,T,3] and positive physics dt required')
    if (force_history.shape[1]-1)*physics_dt < .05-1e-9:
        return torch.zeros(len(force_history),device=force_history.device,dtype=torch.bool)
    return torch.isfinite(force_history).all((1,2)) & (force_history.norm(dim=-1)>.1*weight_n[:,None]).all(-1)
