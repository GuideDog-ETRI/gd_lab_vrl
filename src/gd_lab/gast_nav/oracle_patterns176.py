"""Pattern/extrinsic adapter: consume actual calibrated simulator patterns.

No guessed OS1 beam table, no D4xx, no point inflation or full-map GT oracle.
The caller supplies real OS1/Mid directions; missing calibration is an error.
"""
import math
import torch
from .oracle_occgrid176 import OracleRays176,_validate_rays


def load_gdm_profile176(rbq_yaml,sensor_yaml,*,effective_parameters):
    """Resolved launch values explicit: YAML core near=.2 differs from gdm .4.

    Required effective keys: ouster_min_range_m, ouster_max_range_m,
    livox_min_range, livox_max_range, livox_user_sector_deg,
    livox_vertical_fov_deg. Native OS1 altitudes must come from sensor metadata.
    Returns profile + canonical TF hierarchy; never silently invent metadata.
    """
    import yaml
    with open(rbq_yaml) as f:setup=yaml.safe_load(f)['/trav_map']['ros__parameters']
    with open(sensor_yaml) as f:sensors=yaml.safe_load(f)
    keys=('ouster_min_range_m','ouster_max_range_m','livox_min_range','livox_max_range','livox_user_sector_deg','livox_vertical_fov_deg')
    if any(k not in effective_parameters for k in keys):raise ValueError('resolved gdm sensor parameters required')
    profile={k:effective_parameters[k] for k in keys}
    profile.update(ouster_ring_range=setup['ouster_ring_range'],ouster_exclude_azimuth_deg=setup['input_exclude_azimuth_deg'],self_hits=sensors.get('self_hits',{}),transforms=sensors['transforms'])
    return profile


def mounted_rays176(local_directions,sources,rings,mount_index,rotation,translation,
                    profile,*,screen_door=False,rear_near_m=.001,rear_far_m=6.,hip_x_m=-.31):
    """Native directions [R,3], source/ring/mount [R]; R/t [N,C,3,3]/[N,C,3].

    R/t are true calibrated sensor-local -> current sample body-yaw transforms.
    Front coverage is sensor-local: Ouster clockwise exclusion + native rings;
    Mid rear/user sector and native elevation; screen door additionally body x>0.
    Rear is unspecified coverage, clip rays at hip boundary; no hardware FoV.
    """
    if local_directions.ndim!=2 or local_directions.shape[1]!=3:raise ValueError('calibrated native directions required')
    count=len(local_directions);n=rotation.shape[0]
    if sources.shape!=(count,) or rings.shape!=(count,) or mount_index.shape!=(count,) or sources.dtype!=torch.int64 or rings.dtype!=torch.int64 or mount_index.dtype!=torch.int64:raise ValueError('pattern labels')
    if rotation.ndim!=4 or rotation.shape[-2:]!=(3,3) or translation.shape!=rotation.shape[:2]+(3,):raise ValueError('mount transforms')
    if bool(((mount_index<0)|(mount_index>=rotation.shape[1])).any()):raise ValueError('mount index')
    if not bool(torch.isfinite(local_directions).all()&torch.isfinite(rotation).all()&torch.isfinite(translation).all()):raise ValueError('finite pattern/mount')
    eye=torch.eye(3,device=rotation.device,dtype=rotation.dtype)
    if not torch.allclose(rotation.transpose(-1,-2)@rotation,eye.expand_as(rotation),atol=1e-5) or not bool((torch.linalg.det(rotation)>0).all()):raise ValueError('mount SO3')
    local=local_directions/local_directions.norm(dim=-1,keepdim=True)
    bearing=torch.rad2deg(torch.atan2(local[:,1],local[:,0]));clockwise=(-bearing)%360
    elev=torch.rad2deg(torch.atan2(local[:,2],local[:,:2].norm(dim=-1)))
    valid=torch.ones(count,device=local.device,dtype=torch.bool)
    ouster=sources==0;mid=sources==1;rear=sources==2
    lo,hi=profile['ouster_ring_range'];valid &= ~ouster|((rings>=lo)&(rings<=hi))
    a,b=profile['ouster_exclude_azimuth_deg']
    excluded=((clockwise>=a)&(clockwise<=b)) if a<b else ((clockwise>=a)|(clockwise<=b))
    if a==b:excluded=torch.zeros_like(excluded)
    valid &= ~ouster|~excluded
    user=profile['livox_user_sector_deg']
    vl,vh=profile['livox_vertical_fov_deg']
    valid &= ~mid|((bearing.abs()<=180-user/2)&(elev>=vl)&(elev<=vh))
    direction=torch.einsum('nbij,bj->nbi',rotation[:,mount_index],local)
    origin=translation[:,mount_index]
    near=torch.where(ouster,local.new_tensor(profile['ouster_min_range_m']),torch.where(mid,local.new_tensor(profile['livox_min_range']),local.new_tensor(rear_near_m)))[None].expand(n,-1).clone()
    far=torch.where(ouster,local.new_tensor(profile['ouster_max_range_m']),torch.where(mid,local.new_tensor(profile['livox_max_range']),local.new_tensor(rear_far_m)))[None].expand(n,-1).clone()
    available=valid[None].expand(n,-1).clone()
    if screen_door:
        # Mid screen-door rule is body forward halfspace x>0; crossing rays
        # terminate there, rather than only filtering distant endpoint points.
        dx=direction[...,0];gap=origin[...,0]
        cap=torch.where(dx<0,gap/(-dx).clamp_min(1e-12),torch.inf)
        far=torch.where(mid[None],torch.minimum(far,cap),far)
        available &= ~mid[None]|(gap>=0)
    dx=direction[...,0]
    cap=torch.where(dx>0,(hip_x_m-origin[...,0])/dx.clamp_min(1e-12),torch.inf)
    far=torch.where(rear[None],torch.minimum(far,cap),far)
    available &= ~rear[None]|(origin[...,0]<=hip_x_m)
    available &= far>near
    # Invalid rays retain well-formed positive interval metadata, carve nothing.
    far=torch.where(available,far,near+1.)
    r=_validate_rays(OracleRays176(origin,direction,near,far,available,sources[None].expand(n,-1)))
    return r,local[None].expand(n,-1,-1)


def apply_self_sectors176(returns,local_directions,profile):
    """Classify measured first returns; NEVER remove hit and expose background."""
    from dataclasses import replace
    elev=torch.rad2deg(torch.atan2(local_directions[...,2],local_directions[...,:2].norm(dim=-1)))
    bearing=torch.rad2deg(torch.atan2(local_directions[...,1],local_directions[...,0]))
    self_hit=returns.self_hit.clone()
    for sector in profile.get('self_hits',{}).get('livox_frame',[]):
        angle=(bearing-sector['bearing_deg']+180)%360-180
        selected=(returns.rays.source==1)&(angle.abs()<=sector['half_angle_deg'])&(elev>=sector['elev_min_deg'])&(elev<=sector['elev_max_deg'])&(returns.distance_m<=sector['max_range'])
        self_hit |= selected&torch.isfinite(returns.distance_m)
    return replace(returns,self_hit=self_hit)


def rear_cube_pattern176(divisions=32,*,device='cpu',dtype=torch.float32):
    """Numeric rear oracle quadrature, no sensor kind/FoV. Use actual rear
    coverage bound in mounted_rays176. Finite beam spacing must be validated
    against minimum object size/phase before claiming perfect detection.
    """
    if not isinstance(divisions,int) or divisions<2:raise ValueError('divisions')
    axis=(torch.arange(divisions,device=device,dtype=dtype)+.5)*2/divisions-1
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');u=xx.flatten();v=yy.flatten();one=torch.ones_like(u)
    return torch.cat([torch.stack(d,-1) for d in ((one,u,v),(-one,u,v),(u,one,v),(u,-one,v),(u,v,one),(u,v,-one))],0)
