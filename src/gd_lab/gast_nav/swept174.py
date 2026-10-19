"""3D future collision-volume evidence gate, independent of full-band cells.

No inferred stopping guarantee from a command or a successful subset. Caller
supplies qualified volumes through stop. Missing/unsupported candidate closes.
Current-body subtraction is deliberately NOT applied to future volumes.
"""
from dataclasses import dataclass
import math,torch
from .union174 import UnionProof,root_outer_half
from .free_evidence167 import _rotation,pose_budget,limit_by_current_returns
from .packets import age_codes,age_upper_s

@dataclass(frozen=True)
class FutureSweep174:
    centres: torch.Tensor       # [N,Q,3] current-sample frame
    half: torch.Tensor          # [N,Q,3] individual collision-motion enclosing boxes
    required: torch.Tensor      # [N,Q] no whole-robot bounding-box fill
    qualified: torch.Tensor     # [N] includes current posture, full stop and continuous time
    support_xy: torch.Tensor    # [N,F,2] future support footprints must be separately qualified
    support_required: torch.Tensor # [N,F]

def prove_future174(sweep,measurement,resolved,support,grid,config):
    from .free_evidence167 import _validate
    _validate(measurement,support,grid)
    n=len(measurement.pose);device=support.device
    if sweep is None:return torch.zeros(n,device=device,dtype=torch.bool),{'reason':'NO_QUALIFIED_FUTURE_SWEEP'}
    q=sweep.centres;h=sweep.half
    if q.shape!=h.shape or q.shape[:2]!=sweep.required.shape or q.shape[0]!=n or q.shape[-1]!=3:raise ValueError('future volume shape')
    if not bool(torch.isfinite(q).all()&torch.isfinite(h).all()&(h>=0).all()):raise ValueError('future volume finite')
    m=limit_by_current_returns(measurement,resolved,backend='triton' if support.is_cuda else 'torch');s=m.settings
    covered=torch.zeros_like(sweep.required)
    for i,cap in enumerate(m.captures):
        age=age_upper_s(age_codes(((m.now_ms-cap.time_ms).double()/1000).clamp_min(0))).float()
        same=(cap.time_ms==m.now_ms)&(cap.pose==m.pose).all(-1)&(cap.epoch==m.epoch)&(m.travel_m[:,i]==0)
        eligible=(cap.epoch==m.epoch)&(age<=config.ttl_s)&sweep.qualified
        if s.historical_obstacle_speed_bound_mps is None:eligible &= same
        from .temporal169 import current_support169
        oldh=current_support169(cap,m.pose,m.now_ms,m.epoch,grid,config)
        fxy=sweep.support_xy
        fx=torch.floor((fxy[...,0]-grid.x0)/grid.res+.5).long();fy=torch.floor((fxy[...,1]-grid.y0)/grid.res+.5).long()
        fi=fy.clamp(0,grid.ny-1)*grid.nx+fx.clamp(0,grid.nx-1)
        oldref=oldh.flatten(1).gather(1,fi);newref=support.flatten(1).gather(1,fi)
        references=torch.isfinite(oldref)&torch.isfinite(newref)&((oldref-newref).abs()<=config.intersection_epsilon)
        eligible &= ((~sweep.support_required)|references).all(1)
        displacement=(m.pose[:,:2]-cap.pose[:,:2]).norm(dim=-1)
        eligible &= m.travel_m[:,i]+config.intersection_epsilon>=displacement
        p,y=pose_budget(age,m.travel_m[:,i],m.linear_speed_bound_mps,m.yaw_rate_bound_radps,s)
        if s.same_sample_pose_cancellation:p=torch.where(same,0.,p);y=torch.where(same,0.,y)
        rot=_rotation(cap.pose[:,3]-m.pose[:,3]);shift=torch.einsum('nij,nj->ni',_rotation(-m.pose[:,3]),cap.pose[:,:3]-m.pose[:,:3]);f=cap.frusta
        origins=torch.einsum('nij,nrj->nri',rot,f.origins)+shift[:,None];normals=torch.einsum('nij,nrkj->nrki',rot,f.normals)
        off=f.offsets.clone();off[:,:,5]=torch.maximum(off[:,:,5],-f.far_m)+torch.maximum(torch.full_like(f.far_m,s.range_error_floor_m),s.range_error_fraction*f.far_m)
        for e in eligible.nonzero().flatten().tolist():
            ids=(sweep.required[e]&~covered[e]).nonzero().flatten()
            if not len(ids):continue
            engine=UnionProof(origins[e],normals[e],off[e],f.volume_valid[e],f.origins[e].norm(dim=-1),angle_rad=math.radians(s.sensor_angle_error_deg),support_error=s.support_error_m)
            outer=root_outer_half(q[e,ids],h[e,ids],p[e],y[e],shift[e],float(f.origins[e].norm(dim=-1).max()))
            outer+=age[e]*(s.historical_obstacle_speed_bound_mps or 0.)
            covered[e,ids]=engine.cover(q[e,ids],outer)
    # These support footprints are supplied with uncertainty inflation. Their
    # discretization/coverage is a caller obligation, never inferred from centres.
    xy=sweep.support_xy;ix=torch.floor((xy[...,0]-grid.x0)/grid.res+.5).long();iy=torch.floor((xy[...,1]-grid.y0)/grid.res+.5).long()
    inside=(ix>=0)&(ix<grid.nx)&(iy>=0)&(iy<grid.ny)
    known=torch.isfinite(support.flatten(1).gather(1,(iy.clamp(0,grid.ny-1)*grid.nx+ix.clamp(0,grid.nx-1))))&inside
    enough=((~sweep.required)|covered).all(1)&sweep.required.any(1)&((~sweep.support_required)|known).all(1)&sweep.support_required.any(1)&sweep.qualified
    return enough,dict(reason=('3D_COLLISION_THROUGH_STOP' if bool(sweep.qualified.any()) else 'NO_QUALIFIED_FUTURE_SWEEP'),covered=covered,required=sweep.required,qualified=sweep.qualified,support_known=known)
