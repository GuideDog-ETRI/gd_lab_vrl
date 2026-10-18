"""Physical WIDTH/rotation-envelope oracle, for critic/reward only.

This is not observed feasibility, a stopping-distance proof, or a claim that a
trajectory over a gap/stair is dynamically traversable. Sensor unknown remains
an independent axis. Width uses the actual obstacle-reduced metadata profile.
"""
from dataclasses import dataclass
import math
import torch
from .scene_config import load_scene_config
from .scene_geometry import point_box_distance, floor_height
from .scene_types import project_route


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    available: float
    required: float
    remaining_clearance: float
    reason: str


def passage_feasible(meta, position_xy=None, *, swept_width_m=None, margin_m=None, config=None):
    """Minimum free width ahead of current route projection >= W_swept+2*margin.

    No hindsight: the oracle has GT geometry ahead; its result is privileged and
    must NOT be inserted in actor/map input. Requested y_off is handled by the
    command/supervisor's feasible-interval clamp, not by changing this rule.
    """
    cfg=(config or load_scene_config())['oracle']
    width=cfg['swept_width_m'] if swept_width_m is None else swept_width_m
    margin=cfg['margin_m'] if margin_m is None else margin_m
    if not all(math.isfinite(x) and x >= 0 for x in (width,margin)) or width <= 0:
        raise ValueError('invalid swept width/margin')
    if meta.kind=='legacy': raise ValueError('use existing legacy terrain oracle')
    s=0. if position_xy is None else project_route(meta,position_xy)[0]
    # Include both sides of the current profile interval, then all future
    # breakpoints, so a discontinuity is never interpolated away as extra width.
    future=[w for p,w in meta.width_profile if p >= s]
    previous=[w for p,w in meta.width_profile if p <= s]
    if previous: future.append(previous[-1])
    available=min(future) if future else meta.width_profile[-1][1]
    required=width+2*margin
    ok=available >= required
    return FeasibilityResult(ok,available,required,(available-required)/2,'OK' if ok else 'WIDTH_BLOCKED')


def turn_feasible(meta, position_xy=None, *, turn_swept_radius_m=None, margin_m=None, config=None):
    """Actual local obstacle clearance >= provisional measured-envelope radius+margin."""
    settings=config or load_scene_config(); cfg=settings['oracle']
    radius=cfg['turn_swept_radius_m'] if turn_swept_radius_m is None else turn_swept_radius_m
    margin=cfg['margin_m'] if margin_m is None else margin_m
    if not all(math.isfinite(x) and x >= 0 for x in (radius,margin)) or radius <= 0:
        raise ValueError('invalid radius/margin')
    if meta.turn_spot is None: return FeasibilityResult(False,0.,radius+margin,-radius-margin,'NO_TURN_SPOT')
    xy=meta.turn_spot[0] if position_xy is None else position_xy
    z=floor_height(meta,xy)
    active=[b for b in meta.obstacles if b.kind!='step' and b.center[2]+b.half[2] > z+cfg['epsilon_m'] and b.center[2]-b.half[2] < z+cfg['collision_height_m']]
    if position_xy is None:
        available=meta.turn_spot[1]
    else:
        if not active: raise ValueError('turn room lacks bounding geometry')
        available=max(0.,min(point_box_distance(xy,b) for b in active))
    required=radius+margin; ok=available >= required
    return FeasibilityResult(ok,available,required,available-required,'OK' if ok else 'TURN_BLOCKED')


def batch_feasibility(index, positions_xy, turn_mask, *, config=None):
    """Env-wise scalar GT oracle -> tensors on positions device; CPU-first adapter."""
    if positions_xy.ndim!=2 or positions_xy.shape[-1]!=2 or turn_mask.shape!=positions_xy.shape[:1]:
        raise ValueError('oracle batch shapes')
    metas=index.lookup(range(len(positions_xy))); points=positions_xy.detach().cpu().tolist(); turns=turn_mask.detach().cpu().tolist()
    result=[turn_feasible(m,p,config=config) if t else passage_feasible(m,p,config=config) for m,p,t in zip(metas,points,turns)]
    return {'feasible':torch.tensor([r.feasible for r in result],device=positions_xy.device),
            'available':torch.tensor([r.available for r in result],device=positions_xy.device),
            'required':torch.tensor([r.required for r in result],device=positions_xy.device),
            'clearance':torch.tensor([r.remaining_clearance for r in result],device=positions_xy.device)}
