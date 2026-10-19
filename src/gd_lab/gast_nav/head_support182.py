"""Head first-hit support: a blind near-field must not erase observed flat floor.

Only actual hit cells with a fitted walkable normal and a height within the
approved flat-step limit of measured contact support are retained. No cells in
between are filled. Stair tread connectivity keeps the existing growth rule.
This supervisor adapter does not modify actor terrain or the native mapper.
"""
import math
import torch
from . import schema as S
from .obs_terms import fit_observed_surface
from .support import make_seed_mask,grow_support
from .map_config import load_config

def retain_observed_flat182(height,observed,normals,reference,segment,grown,*,limit_m=.16,config=None):
 cfg=config or load_config()
 norm=normals.norm(dim=-1)
 horizontal=torch.isfinite(normals).all(-1)&(norm>0)&(normals[...,2]/norm.clamp_min(1e-9)>=math.cos(math.radians(min(cfg.max_slope_deg,cfg.horizontal_max_slope_deg))))
 segment=torch.as_tensor(segment,device=height.device).expand(len(height))
 stair=(segment==S.SEGMENT_TYPES.index('STAIR_UP'))|(segment==S.SEGMENT_TYPES.index('STAIR_DOWN'))|(segment==S.SEGMENT_TYPES.index('LANDING'))
 measured=observed&torch.isfinite(height)&torch.isfinite(reference[:,None,None])&horizontal
 measured &= (height>=cfg.min_z_m)&(height<=cfg.max_z_m)&((height-reference[:,None,None]).abs()<=limit_m+cfg.intersection_epsilon)
 measured &= ~stair[:,None,None]
 result=torch.where(measured,height,grown)
 return torch.where(torch.isfinite(reference[:,None,None]),result,torch.nan),measured

def observed_head_support182(points,valid,contact_points,contact_valid,segment,grid,settings,*,config=None):
 h,observed,normals=fit_observed_surface(points,valid,grid,settings)
 near=points[...,:2].norm(dim=-1)<=settings['support_body_radius_m']
 seeds=make_seed_mask(contact_points,contact_valid,points,valid,near,grid=grid)
 reference=contact_points[...,2].masked_fill(~contact_valid,torch.nan).nanmedian(-1).values
 grown,_,_=grow_support(h,observed,normals,seeds,segment_type=segment,current_support_height=torch.nan_to_num(reference),config=config)
 from .terrain_fusion177 import FusionConfig177
 support,_=retain_observed_flat182(h,observed,normals,reference,segment,grown,limit_m=FusionConfig177.load().flat_step_max_m,config=config)
 return support
