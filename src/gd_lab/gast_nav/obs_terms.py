"""Observed-only sensor bridge. No terrain metadata/oracle accepted by actor map API.

Isaac imports are lazy so ray geometry, support fitting and map conversion can be
verified without loading a simulator. All map coordinates are sample body-yaw.
"""
import math
import torch
from .map_types import RayBatch, DynamicCapsules, WIDE_GRID
from .map_config import load_config
from .ray_geometry import capsule_intersections
from .support import grow_support, make_seed_mask
from .obstacle_map import ObstacleMap
from .near_clearance import build_near, extended_near_grid
from .map_inputs import network_inputs


def sensor_pattern(settings, *, device='cpu'):
    """Combined front hemispheres and two rear hemispheres behind the hips."""
    na, ne = settings['azimuth_count'], settings['elevation_count']
    # Cell centres avoid duplicated +/- pi boundary rays.
    az = (torch.arange(na, device=device)+.5)*math.pi/na-math.pi/2
    el = torch.linspace(*settings['elevation_deg'], ne, device=device)*math.pi/180
    e, a = torch.meshgrid(el, az, indexing='ij')
    direction = torch.stack((e.cos()*a.cos(), e.cos()*a.sin(), e.sin()), -1).reshape(-1, 3)
    starts, directions = [], []
    for rear, key in ((False, 'front_mounts_m'), (True, 'rear_mounts_m')):
        for mount in settings[key]:
            starts.append(torch.tensor(mount, device=device).expand_as(direction).clone())
            d = direction.clone()
            if rear: d[:, :2] *= -1
            directions.append(d)
    return torch.cat(starts), torch.cat(directions)


def randomize_sensor_pattern(sensor, env_ids, settings, *, generator):
    """Episode-local mount and angle jitter; never modifies another env's rays."""
    ids = torch.as_tensor(env_ids, device=sensor.ray_starts.device, dtype=torch.long)
    base_o, base_d = sensor_pattern(settings, device=ids.device)
    mounts = len(settings['front_mounts_m'])+len(settings['rear_mounts_m'])
    per_mount = len(base_o)//mounts
    jitter = (torch.rand((len(ids), mounts, 3), device=ids.device, generator=generator)*2-1)*settings['mount_jitter_m']
    angles = (torch.rand((len(ids), mounts, 1), device=ids.device, generator=generator)*2-1)*settings['angle_jitter_rad']
    angles = angles.repeat_interleave(per_mount, 1)
    d = base_d[None].expand(len(ids), -1, -1).clone()
    x, y = d[..., :1].clone(), d[..., 1:2].clone()
    d[..., :1], d[..., 1:2] = angles.cos()*x-angles.sin()*y, angles.sin()*x+angles.cos()*y
    sensor.ray_starts[ids] = base_o+jitter.repeat_interleave(per_mount, 1)
    sensor.ray_directions[ids] = d


def isaac_sensor_cfg(settings, *, prim_path, mesh_prim_path, update_period):
    """Static mesh RayCaster. Dynamic cylinders are intersected analytically later.
    'base' alignment rotates BOTH origins and directions by sensor quaternion.
    """
    from isaaclab.sensors import RayCasterCfg
    from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
    from isaaclab.utils import configclass
    @configclass
    class NavPatternCfg(PatternBaseCfg):
        func = staticmethod(lambda cfg, device: sensor_pattern(cfg.settings, device=device))
        settings: dict = None
    return RayCasterCfg(prim_path=prim_path, mesh_prim_paths=[mesh_prim_path],
                        ray_alignment='base', pattern_cfg=NavPatternCfg(settings=settings),
                        max_distance=settings['max_range_m'], update_period=update_period)


def world_to_sample(points, pose, *, vectors=False):
    p = points if vectors else points-pose[:, None, :3]
    c, s = pose[:, 3].cos()[:, None], pose[:, 3].sin()[:, None]
    return torch.stack((c*p[..., 0]+s*p[..., 1], -s*p[..., 0]+c*p[..., 1], p[..., 2]), -1)


def capsules_to_sample(capsules, pose):
    return DynamicCapsules(world_to_sample(capsules.starts, pose), world_to_sample(capsules.ends, pose), capsules.radii)


def static_sensor_rays(sensor, pose, *, self_capsules=None, config=None):
    """Extract actual RayCaster base-frame pattern and pose, including randomized mounts.
    Self capsules are supplied in world coordinates from robot joint geometry.
    """
    from isaaclab.utils.math import quat_apply
    data = sensor.data
    q = data.quat_w[:, None].expand(-1, sensor.ray_starts.shape[1], -1)
    ow = quat_apply(q, sensor.ray_starts)+data.pos_w[:, None]
    dw = quat_apply(q, sensor.ray_directions)
    distance = (data.ray_hits_w-ow).norm(dim=-1)
    distance = torch.where(torch.isfinite(distance), distance, torch.inf)
    rays = RayBatch(world_to_sample(ow, pose), world_to_sample(dw, pose, vectors=True), distance,
                    torch.zeros_like(distance, dtype=torch.bool))
    return mask_self_returns(rays, capsules_to_sample(self_capsules, pose), config=config) if self_capsules is not None else rays


def mask_self_returns(rays, self_capsules, *, config=None):
    cfg = config or load_config()
    direction = torch.nn.functional.normalize(rays.directions, dim=-1)
    hit = capsule_intersections(rays.origins, direction, self_capsules, eps=cfg.intersection_epsilon).amin(-1)
    is_self = hit <= rays.distances
    return RayBatch(rays.origins, direction, torch.minimum(hit, rays.distances), rays.self_hit | is_self)


def fit_observed_surface(points, valid, grid, settings):
    """Grid observed returns and fit local planes; no interpolation across missing cells.

    A cell is observed only when it contains an actual return. Local covariance
    gives an upward normal; rank-deficient fits (one point/line) stay unknown.
    Neighbours are measured hits within a configured Euclidean radius.
    """
    n, r, _ = points.shape
    finite = valid.bool() & torch.isfinite(points).all(-1)
    safe = torch.where(finite[..., None], points, 0.)
    ix = torch.floor((safe[..., 0]-grid.x0)/grid.res+.5).long()
    iy = torch.floor((safe[..., 1]-grid.y0)/grid.res+.5).long()
    inside = finite & (ix>=0) & (ix<grid.nx) & (iy>=0) & (iy<grid.ny)
    index = iy.clamp(0,grid.ny-1)*grid.nx+ix.clamp(0,grid.nx-1)
    heights = points.new_full((n,grid.nx*grid.ny), torch.inf)
    heights.scatter_reduce_(1,index,torch.where(inside,safe[...,2],torch.inf),reduce='amin',include_self=True)
    observed = torch.isfinite(heights)
    xy = grid.xy(device=points.device,dtype=points.dtype).flatten(0,1)
    normals = points.new_full((n,len(xy),3),torch.nan)
    for first in range(0,len(xy),settings['fit_chunk']):
        last=min(first+settings['fit_chunk'],len(xy))
        centre=torch.cat((xy[None,first:last].expand(n,-1,-1),heights[:,first:last,None]),-1)
        nearby=((safe[:,None]-centre[:,:,None]).square().sum(-1) <= settings['support_normal_radius_m']**2) & finite[:,None]
        count=nearby.sum(-1)
        weight=nearby.to(points.dtype)
        mean=torch.einsum('ncr,nrd->ncd',weight,safe)/count.clamp_min(1)[...,None]
        delta=safe[:,None]-mean[:,:,None]
        covariance=torch.einsum('ncr,ncri,ncrj->ncij',weight,delta,delta)/count.clamp_min(1)[...,None,None]
        eigval,eigvec=torch.linalg.eigh(covariance)
        normal=eigvec[...,0]
        normal=torch.where(normal[...,2:3]<0,-normal,normal)
        good=(count>=settings['normal_min_points']) & (eigval[...,1]>settings['support_min_scatter']) & observed[:,first:last]
        normals[:,first:last]=torch.where(good[...,None],normal,torch.nan)
    shape=(n,grid.ny,grid.nx)
    return torch.where(observed,heights,torch.nan).reshape(shape), observed.reshape(shape), normals.reshape(*shape,3)


def observed_support(points, valid, contact_points, contact_valid, segment, grid, settings, *, config=None):
    height, observed, normals=fit_observed_surface(points,valid,grid,settings)
    near=points[...,:2].norm(dim=-1)<=settings['support_body_radius_m']
    seeds=make_seed_mask(contact_points,contact_valid,points,valid,near,grid=grid)
    # Measured contact z, never root height minus a presumed standing height.
    reference=torch.where(contact_valid,contact_points[...,2],torch.nan).nanmedian(dim=-1).values
    reference_valid=torch.isfinite(reference)
    support,code,connected=grow_support(height,observed,normals,seeds,segment_type=segment,
                                       current_support_height=torch.nan_to_num(reference),config=config)
    # Without current support evidence even flat priors cannot manufacture floor.
    support=torch.where(reference_valid[:,None,None],support,torch.nan)
    return support


class ObservationPipeline:
    def __init__(self,num_envs,settings,*,device='cpu',map_config=None):
        self.settings=settings
        self.config=map_config or load_config()
        self.mapper=ObstacleMap(num_envs,device=device,config=self.config)
        self.near=None

    def reset(self,env_ids):
        self.mapper.reset(env_ids)
        if self.near is not None:
            self.near.unknown[env_ids]=True
            self.near.age_s[env_ids]=torch.inf

    def capture(self,*,rays,terrain_points,terrain_valid,contact_points,contact_valid,segment,pose,time_ms,capsules=None):
        # Static first returns supply support evidence; self returns never do.
        hit=rays.origins+rays.directions*torch.nan_to_num(rays.distances,posinf=0.)[...,None]
        valid=torch.isfinite(rays.distances) & ~rays.self_hit & (rays.distances<=self.config.ray_max_range_m)
        if capsules is not None:
            dynamic=capsule_intersections(rays.origins,torch.nn.functional.normalize(rays.directions,dim=-1),capsules,
                                          eps=self.config.intersection_epsilon).amin(-1)
            valid &= rays.distances<=dynamic
        points=torch.cat((hit,terrain_points),1);valid=torch.cat((valid,terrain_valid),1)
        args=(points,valid,contact_points,contact_valid,segment)
        wide_support=observed_support(*args,WIDE_GRID,self.settings,config=self.config)
        fine_support=observed_support(*args,extended_near_grid()[0],self.settings,config=self.config)
        self.mapper.update(rays,wide_support,pose,time_ms,capsules=capsules)
        self.near=build_near(rays,fine_support,capsules=capsules,config=self.config)
        return self.inputs(pose,time_ms)

    def inputs(self,pose,time_ms):
        if self.near is None:raise RuntimeError('capture required before inference')
        return network_inputs(self.mapper.history,self.near,pose,time_ms,config=self.config)


def critic_privileged(*,sdf,feasible,observed,acceleration,controller):
    """Separate critic-only tensor: SDF samples, two feasibility axes, a_train, controller onehot."""
    from . import schema
    return torch.cat((sdf.flatten(1),feasible[:,None].float(),observed[:,None].float(),acceleration,
                      torch.nn.functional.one_hot(controller.long(),len(schema.CONTROLLERS)).float()),-1)
