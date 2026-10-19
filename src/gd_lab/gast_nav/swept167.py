"""Candidate trajectory collision union, pending validated trajectory source.

No command-speed proxy, enclosing hull, or subtraction of current body space.
Caller supplies candidate-through-stop link poses and a verified bound on
collision-point travel between samples, plus projected pose/support error.
This API is deliberately NOT connected to the runtime supervisor yet.
"""
import math
import torch


def trajectory_required(footprint, grid, body_positions, body_quaternions, sample_pose,
                        *, projected_error_m, between_sample_motion_bound_m):
    if body_positions.ndim!=4 or body_positions.shape[1]<2:
        raise ValueError('candidate-through-stop trajectory requires N,T,links,3')
    if body_positions.shape[-1]!=3 or body_quaternions.shape!=(*body_positions.shape[:-1],4):
        raise ValueError('trajectory shape')
    for value in (projected_error_m,between_sample_motion_bound_m):
        if not math.isfinite(value) or value<0:raise ValueError('verified nonnegative error/motion bound required')
    if not torch.isfinite(body_positions).all() or not torch.isfinite(body_quaternions).all():
        raise ValueError('nonfinite trajectory')
    xy=grid.xy(device=body_positions.device,dtype=body_positions.dtype)
    # Cell centre coverage includes any intersecting cell; inter-sample bound
    # covers collision motion not represented by the supplied sampled poses.
    margin=projected_error_m+between_sample_motion_bound_m+grid.res/math.sqrt(2)
    required=torch.zeros((len(body_positions),grid.ny,grid.nx),device=xy.device,dtype=torch.bool)
    for t in range(body_positions.shape[1]):
        required |= footprint.mask(xy,body_positions[:,t],body_quaternions[:,t],sample_pose,margin)
    return required


def candidate_stop_profile(value, mode, cfg, *, dt=.01, turn_deceleration=None):
    """Scalar commanded velocity through delay+one tick and linear ramp to zero."""
    if mode not in ('NAV','TURN') or not math.isfinite(value) or dt<=0:raise ValueError('candidate')
    if mode=='NAV' and value<0:raise ValueError('reverse NAV has no envelope')
    decel=cfg['brake_deceleration_mps2'] if mode=='NAV' else turn_deceleration
    if decel is None or decel<=0:raise ValueError('TURN deceleration must be explicit')
    hold=math.ceil(cfg['command_delay_s']/dt)+cfg['extra_control_ticks']
    ramp=math.ceil(abs(value)/(decel*dt))
    speeds=[value]*hold+[math.copysign(max(0.,abs(value)-(i+1)*decel*dt),value) for i in range(ramp)]
    return torch.tensor(speeds,dtype=torch.float64)


def p2_projected_inflation(profile, cfg, *, dt=.01, radius_m, angular_speed_max=0.):
    """99% marginal P2 error inflation, NOT a hard or joint pathwise bound.

    Matches UPDATE157 dimensionless slip noise sampled at 50Hz. Surface error
    is supplied separately as a vertical extrusion, not silently added to XY.
    """
    if radius_m<0 or not torch.isfinite(profile).all():raise ValueError('profile')
    duration=len(profile)*dt
    variance=cfg['rw_xy_m_sqrt_s']**2*duration+cfg['k_slip']**2*.02*float(profile.square().sum())*dt+cfg['white_xy_m']**2
    xy=math.sqrt(-2*math.log(.01))*math.sqrt(variance)
    xy+=float(profile.abs().max())*cfg['pose_delay_s'] if len(profile) else 0.
    yaw=2.575829*math.radians(math.sqrt(cfg['rw_yaw_deg_sqrt_s']**2*duration+cfg['white_yaw_deg']**2))
    yaw+=abs(angular_speed_max)*cfg['pose_delay_s']
    return dict(horizontal_m=xy+radius_m*yaw,vertical_m=cfg['support_error_m'],duration_s=duration,
                statistical_contract='99% marginal XY-radius/yaw, not simultaneous pathwise guarantee')
