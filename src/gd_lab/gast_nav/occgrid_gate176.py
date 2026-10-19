"""UPDATE176 supervisor-only 2-D stopping-sweep gate; no actor observations.

The caller supplies measured first-hit occupancy and observed support. This
module never constructs either from scene ground truth or band-free certificates.
"""
from dataclasses import dataclass
import math
import torch
from . import schema as S


@dataclass(frozen=True)
class OccupancyInput176:
    state: torch.Tensor       # [N,Y,X], UNKNOWN/FREE/OCCUPIED
    age_s: torch.Tensor       # age of actual evidence, not publication
    support: torch.Tensor     # observed support height, NaN when unavailable
    support_age_s: torch.Tensor
    epoch: torch.Tensor      # [N]
    walkable_gap: torch.Tensor | None = None
    walkable_gap_age_s: torch.Tensor | None = None


def certify_stopping_grid176(observation, sweep, grid, *, epoch, margin_m, ttl_s=.5, support_ttl_s=.5, current_body_mask=None, static_oracle183=False):
    """Intersect every enlarged primitive projection with full cells, not centres.

    Out-of-grid swept volume, absent/unsupported trajectories, stale data and
    missing support fail closed. Optional D-5 current-body subtraction. Primitive-wise
    projection preserves holes between disjoint parts instead of filling a hull.
    """
    if not math.isfinite(ttl_s) or ttl_s <= 0:
        raise ValueError('positive obstacle TTL required')
    support_limit=2. if static_oracle183 else .5
    if not math.isfinite(support_ttl_s) or not 0 < support_ttl_s <= support_limit:raise ValueError('support TTL outside explicit context')
    state=observation.state
    if state.ndim!=3 or state.shape[1:]!=(grid.ny,grid.nx):raise ValueError('occupancy grid shape')
    if any(x.shape!=state.shape for x in (observation.age_s,observation.support,observation.support_age_s)):
        raise ValueError('evidence shape')
    n=len(state);dev=state.device
    if epoch.shape!=(n,) or observation.epoch.shape!=(n,):raise ValueError('epoch shape')
    if bool(((state!=S.UNKNOWN)&(state!=S.FREE)&(state!=S.OCCUPIED)).any()):raise ValueError('state code')
    required=torch.zeros_like(state,dtype=torch.bool)
    if sweep is None:
        return torch.zeros(n,device=dev,dtype=torch.bool),dict(reason='NO_QUALIFIED_FUTURE_SWEEP',required=required)
    c,h=sweep.centres,sweep.half
    if c.shape!=h.shape or c.shape[:2]!=sweep.required.shape or c.shape[0]!=n or c.shape[-1]!=3:
        raise ValueError('sweep shape')
    if not bool(torch.isfinite(c).all()&torch.isfinite(h).all()&(h>=0).all()):raise ValueError('invalid sweep')
    margin=torch.as_tensor(margin_m,device=dev,dtype=c.dtype).expand(n)
    if not bool(torch.isfinite(margin).all()&(margin>=0).all()):raise ValueError('invalid margin')
    lo=c[...,:2]-h[...,:2]-margin[:,None,None]
    hi=c[...,:2]+h[...,:2]+margin[:,None,None]
    xy=grid.xy(device=dev,dtype=c.dtype)
    bounds_lo=xy[0,0]-grid.res/2;bounds_hi=xy[-1,-1]+grid.res/2
    inside=((lo>=bounds_lo)&(hi<=bounds_hi)).all(-1)
    # Loop over collision primitives only, vectorized over all envs/cells.
    for j in range(c.shape[1]):
        overlap=((xy[None]+grid.res/2>=lo[:,j,None,None])&
                 (xy[None]-grid.res/2<=hi[:,j,None,None])).all(-1)
        required |= overlap&sweep.required[:,j,None,None]
    before_exclusion=required.clone()
    if current_body_mask is not None:
        if current_body_mask.shape!=state.shape or current_body_mask.dtype!=torch.bool:raise ValueError("current body mask")
        required &= ~current_body_mask
    a=observation.age_s;sa=observation.support_age_s
    fresh=torch.isfinite(a)&(a>=0)&(a<=ttl_s)
    supported=torch.isfinite(observation.support)&torch.isfinite(sa)&(sa>=0)&(sa<=support_ttl_s)
    traversable=supported.clone()
    if observation.walkable_gap is not None:
        ga=observation.walkable_gap_age_s
        if ga is None or ga.shape!=state.shape or observation.walkable_gap.shape!=state.shape:raise ValueError('gap sidecar')
        traversable |= observation.walkable_gap&torch.isfinite(ga)&(ga>=0)&(ga<=support_ttl_s)
    free=(state==S.FREE)&fresh
    occupied=(state==S.OCCUPIED)&fresh
    qualified=sweep.qualified&sweep.required.any(-1)&((~sweep.required)|inside).all(-1)
    valid_epoch=epoch==observation.epoch
    ok=qualified&valid_epoch&required.flatten(1).any(-1)&((~required)|(free&traversable)).flatten(1).all(-1)
    return ok,dict(reason='OCCGRID_STOPPING_SWEEP',required=required,required_before_exclusion=before_exclusion,qualified=qualified,
                   outside=(sweep.required&~inside).any(-1),
                   occupied=(required&occupied).flatten(1).any(-1),
                   unknown=(required&~free&~occupied).flatten(1).any(-1),
                   cells_satisfied=required.flatten(1).any(-1)&((~required)|(free&traversable)).flatten(1).all(-1),
                   missing_support=(required&~supported).flatten(1).any(-1),valid_epoch=valid_epoch,missing_traversable=(required&~traversable).flatten(1).any(-1))
