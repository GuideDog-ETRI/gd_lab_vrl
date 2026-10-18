"""Batched analytic first-hit geometry; no simulator or GT visibility shortcuts."""
import torch
from .map_config import load_config
from .map_types import ResolvedRays


def ray_box_interval(origin, direction, low, high, eps):
    """Broadcast slab interval for closed AABBs; parallel rays handled exactly."""
    parallel = direction.abs() <= eps
    outside = (origin < low) | (origin > high)
    denom = torch.where(parallel, torch.ones_like(direction), direction)
    a, b = (low-origin)/denom, (high-origin)/denom
    entry = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(-1)
    exit = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(-1)
    miss = (parallel & outside).any(-1) | (exit < entry)
    return torch.where(miss, torch.inf, entry), torch.where(miss, -torch.inf, exit)


def _sphere_hit(o, d, centre, radius, eps):
    v = o-centre
    b = (v*d).sum(-1)
    c = (v*v).sum(-1)-radius.square()
    disc = b.square()-c
    root = -b-disc.clamp_min(0).sqrt()
    return torch.where(c <= 0, torch.zeros_like(root),
                       torch.where((disc >= 0) & (root >= 0), root, torch.inf))


def capsule_intersections(origins, directions, capsules, *, eps):
    """[N,R,K] exact ray intersections, capsule segment plus endpoint spheres."""
    a, b = capsules.starts[:, None], capsules.ends[:, None]
    radius = capsules.radii[:, None]
    o, d = origins[:, :, None], directions[:, :, None]
    ba, oa = b-a, o-a
    length2 = (ba*ba).sum(-1)
    bard, baoa = (ba*d).sum(-1), (ba*oa).sum(-1)
    qa = length2-bard.square()
    qb = length2*(d*oa).sum(-1)-baoa*bard
    qc = length2*(oa*oa).sum(-1)-baoa.square()-radius.square()*length2
    disc = qb.square()-qa*qc
    t = (-qb-disc.clamp_min(0).sqrt()) / torch.where(qa.abs() > eps, qa, torch.ones_like(qa))
    y = baoa+t*bard
    cylinder = torch.where((qa > eps) & (disc >= 0) & (t >= 0) & (y >= 0) & (y <= length2), t, torch.inf)
    sphere_a = _sphere_hit(o, d, a, radius, eps)
    sphere_b = _sphere_hit(o, d, b, radius, eps)
    projection = (baoa/length2.clamp_min(eps)).clamp(0, 1)
    inside = ((oa-projection[..., None]*ba).square().sum(-1) <= radius.square())
    return torch.where(inside, 0., torch.minimum(cylinder, torch.minimum(sphere_a, sphere_b)))


def resolve_first_hits(rays, *, boxes=None, capsules=None, config=None):
    """Normalize rays, combine static/self and analytic dynamic nearest hits.

    Static self hits terminate carving but are never obstacle points. A closer
    dynamic return remains occupied even if the static ray hit was a self hit.
    +inf is the only valid no-return distance. Rays stop at ray_max_range_m.
    """
    cfg = config or load_config()
    o, d, dist, self_hit = rays.origins, rays.directions, rays.distances, rays.self_hit
    if o.ndim != 3 or o.shape[-1] != 3 or d.shape != o.shape or dist.shape != o.shape[:2] or self_hit.shape != dist.shape:
        raise ValueError('ray tensor shapes')
    if self_hit.dtype != torch.bool or not o.is_floating_point() or not d.is_floating_point() or not dist.is_floating_point():
        raise ValueError('ray dtypes')
    norm = torch.linalg.vector_norm(d, dim=-1)
    if not bool(torch.isfinite(o).all() & torch.isfinite(d).all() & (norm > cfg.intersection_epsilon).all()
                & (~torch.isnan(dist)).all() & (dist >= 0).all()):
        raise ValueError('invalid rays')
    d = d/norm[..., None]
    dynamic = torch.full_like(dist, torch.inf)
    if boxes is not None:
        if boxes.centres.ndim != 3 or boxes.centres.shape[0] != o.shape[0] or boxes.centres.shape[-1] != 3 or boxes.half_extents.shape != boxes.centres.shape or boxes.yaw.shape != boxes.centres.shape[:2]:
            raise ValueError('box shapes')
        if not bool(torch.isfinite(boxes.centres).all() & torch.isfinite(boxes.half_extents).all() & torch.isfinite(boxes.yaw).all() & (boxes.half_extents > 0).all()):
            raise ValueError('invalid boxes')
        for start in range(0, boxes.centres.shape[1], cfg.ray_chunk):
            c = boxes.centres[:, None, start:start+cfg.ray_chunk]
            half = boxes.half_extents[:, None, start:start+cfg.ray_chunk]
            yaw = boxes.yaw[:, None, start:start+cfg.ray_chunk]
            co, si = yaw.cos(), yaw.sin()
            delta = o[:, :, None]-c
            local_o = torch.stack((co*delta[..., 0]+si*delta[..., 1], -si*delta[..., 0]+co*delta[..., 1], delta[..., 2]), -1)
            dr = d[:, :, None]
            local_d = torch.stack((co*dr[..., 0]+si*dr[..., 1], -si*dr[..., 0]+co*dr[..., 1], dr[..., 2].expand_as(local_o[..., 2])), -1)
            entry, exit = ray_box_interval(local_o, local_d, -half, half, cfg.intersection_epsilon)
            t = entry.clamp_min(0)
            t = torch.where(exit >= t, t, torch.inf)
            dynamic = torch.minimum(dynamic, t.amin(-1))
    if capsules is not None:
        if capsules.starts.ndim != 3 or capsules.starts.shape[0] != o.shape[0] or capsules.starts.shape[-1] != 3 or capsules.ends.shape != capsules.starts.shape or capsules.radii.shape != capsules.starts.shape[:2]:
            raise ValueError('capsule shapes')
        if not bool(torch.isfinite(capsules.starts).all() & torch.isfinite(capsules.ends).all() & torch.isfinite(capsules.radii).all() & (capsules.radii > 0).all()):
            raise ValueError('invalid capsules')
        from .map_types import DynamicCapsules
        for start in range(0, capsules.starts.shape[1], cfg.ray_chunk):
            batch = DynamicCapsules(capsules.starts[:, start:start+cfg.ray_chunk], capsules.ends[:, start:start+cfg.ray_chunk], capsules.radii[:, start:start+cfg.ray_chunk])
            dynamic = torch.minimum(dynamic, capsule_intersections(o, d, batch, eps=cfg.intersection_epsilon).amin(-1))
    # Static/self wins exact ties. It is unsafe to carve beyond a self return.
    use_dynamic = dynamic < dist
    nearest = torch.minimum(dynamic, dist)
    is_self = self_hit.bool() & ~use_dynamic & (dist <= cfg.ray_max_range_m)
    occupied = (nearest <= cfg.ray_max_range_m) & torch.isfinite(nearest) & ~is_self
    return ResolvedRays(o, d, nearest.clamp_max(cfg.ray_max_range_m), occupied, is_self)
