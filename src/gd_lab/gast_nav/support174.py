"""Measured contact positions and conservative historical support (no D4xx)."""
import torch
from .temporal169 import current_support169
from .free_evidence167 import pose_budget
from .packets import age_codes,age_upper_s

def actual_contacts174(env,foot_names,pose,world_to_sample):
    """PhysX contact POINT averages, not link centres. No contact -> NaN/invalid."""
    pts=[]
    for foot in foot_names:
        value=env.scene['ground_contact174_'+foot].data.contact_pos_w
        if value is None:raise ValueError('actual contact points must be enabled')
        # One sensor body and one ground filter. Reject inconsistent data shape.
        if value.shape[1]==1 and value.shape[-1]==3:
            valid=torch.isfinite(value[:,0]).all(-1);count=valid.sum(1)
            point=torch.nan_to_num(value[:,0]).sum(1)/count.clamp_min(1)[:,None]
            pts.append(torch.where((count>0)[:,None],point,torch.nan))
        else:raise ValueError('one foot / one ground filter required')
    p=torch.stack(pts,1);valid=torch.isfinite(p).all(-1)
    if getattr(env,'_contact174_force_gate',False):
        if not hasattr(env,'_contact_weight174'):
            env._contact_weight174=env.scene['robot'].root_physx_view.get_masses().sum(-1).to(p.device)*9.81
        for i,foot in enumerate(foot_names):
            history=env.scene['ground_contact174_'+foot].data.net_forces_w_history
            if getattr(getattr(env,'nav',None),'cfg',{}).get('pipeline179',False):
                from .pipeline179 import stable_contact179
                valid[:,i] &= stable_contact179(history[:,:,0],env._contact_weight174,env.cfg.sim.dt)
            else:
                valid[:,i] &= (history[:,:,0].norm(dim=-1)>.1*env._contact_weight174[:,None]).all(-1)

    return world_to_sample(p,pose),valid

def historical_support174(measurement,grid,config):
    """Newest valid source wins; unknown current support may use valid old support.

    P2 horizontal uncertainty erodes observed-support regions, rather than
    filling gaps with a nominal warp. Changed newer measured height blocks older
    heights. The returned source age is never refreshed to query time.
    """
    m=measurement;n=len(m.pose);out=torch.full((n,grid.ny,grid.nx),torch.nan,device=m.pose.device)
    ages=torch.full_like(out,torch.inf);reference=torch.full_like(out,torch.nan)
    for i,cap in enumerate(m.captures):
        h=current_support169(cap,m.pose,m.now_ms,m.epoch,grid,config)
        raw=((m.now_ms-cap.time_ms).double()/1000).clamp_min(0)
        upper=age_upper_s(age_codes(raw)).to(h.dtype)
        same=(cap.time_ms==m.now_ms)&(cap.pose==m.pose).all(-1)&(cap.epoch==m.epoch)&(m.travel_m[:,i]==0)
        p,y=pose_budget(upper,m.travel_m[:,i],m.linear_speed_bound_mps,m.yaw_rate_bound_radps,m.settings)
        if m.settings.same_sample_pose_cancellation:
            p=torch.where(same,0.,p);y=torch.where(same,0.,y)
        maxr=grid.xy(device=h.device,dtype=h.dtype).norm(dim=-1).max()+grid.res
        radius=torch.ceil((p+2*torch.sin(y.clamp(max=3.14159265)/2)*maxr)/grid.res).long()
        eligible=(cap.epoch==m.epoch)&(upper<=config.ttl_s)
        if m.settings.historical_obstacle_speed_bound_mps is None:eligible &= same
        finite=torch.isfinite(h)
        # Height reference comparison rejects reused support across measured steps.
        conflict=torch.isfinite(reference)&finite&((reference-h).abs()>config.overlap_epsilon)
        valid=finite&~conflict&eligible[:,None,None]
        # All cells touched by possible xy displacement need agreeing heights.
        for r in torch.unique(radius).tolist():
            ids=radius==r
            if r>=max(grid.nx,grid.ny):valid[ids]=False;continue
            if r:
                hh=h[ids,None];known=finite[ids,None]
                low=-torch.nn.functional.max_pool2d(torch.where(known,-hh,torch.inf),2*r+1,1,r)
                high=torch.nn.functional.max_pool2d(torch.where(known,hh,torch.inf),2*r+1,1,r)
                full=torch.nn.functional.avg_pool2d(known.float(),2*r+1,1,r,count_include_pad=True)==1
                valid[ids] &= (full&((high-low)<=config.overlap_epsilon))[:,0]
        take=~torch.isfinite(out)&valid
        out=torch.where(take,h,out);ages=torch.where(take,upper[:,None,None],ages)
        reference=torch.where(~torch.isfinite(reference)&finite,h,reference)
    return out,ages
