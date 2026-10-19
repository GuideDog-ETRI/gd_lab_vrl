"""UPDATE177 analytic command-to-stop sweep, independent of measured pose domains.

Conservative sampled rotations of projected collision boxes. Between-sample
angular motion is enclosed with a Lipschitz pad; translation is an extrusion
of each angular box. This encloses combinations of turn and translation, rather
than claiming a measured gait or a deployment stop guarantee.
"""
import math
import torch
from .swept174 import FutureSweep174


def stopping_sweep177(centres,half,velocity,omega,*,control_dt=.01,delay_s=.04,
                      deceleration=.5,angular_deceleration=1.,gait_margin=.04,
                      turn_radius=.76,reverse_overshoot_deg=3.,samples=17):
    if centres.shape!=half.shape or centres.ndim!=3 or centres.shape[-1]!=3:raise ValueError('collision boxes')
    n,q,_=centres.shape
    if velocity.shape!=(n,2) or omega.shape!=(n,):raise ValueError('command shape')
    if not all(math.isfinite(v) for v in (control_dt,delay_s,deceleration,angular_deceleration,gait_margin,turn_radius,reverse_overshoot_deg)):raise ValueError('finite settings')
    if control_dt<=0 or delay_s<0 or deceleration<=0 or angular_deceleration<=0 or gait_margin<0 or turn_radius<=0 or reverse_overshoot_deg<0 or samples<2:raise ValueError('settings')
    if not bool(torch.isfinite(centres).all()&torch.isfinite(half).all()&(half>=0).all()&torch.isfinite(velocity).all()&torch.isfinite(omega).all()):raise ValueError('finite motion/geometry')
    latency=delay_s+control_dt;speed=velocity.norm(dim=-1)
    shift=velocity*(latency+speed/(2*deceleration))[:,None]
    theta=omega*(latency+omega.abs()/(2*angular_deceleration))
    overshoot=(omega!=0).to(omega.dtype)*math.radians(reverse_overshoot_deg)
    lo=torch.minimum(theta,torch.zeros_like(theta))-overshoot
    hi=torch.maximum(theta,torch.zeros_like(theta))+overshoot
    angles=lo[:,None]+(hi-lo)[:,None]*torch.linspace(0,1,samples,device=centres.device,dtype=centres.dtype)
    c,s=angles.cos()[:,:,None],angles.sin()[:,:,None]
    cx,cy=centres[:,None,:,0],centres[:,None,:,1]
    hx,hy=half[:,None,:,0]+gait_margin,half[:,None,:,1]+gait_margin
    x=c*cx-s*cy;y=s*cx+c*cy
    ex=c.abs()*hx+s.abs()*hy;ey=s.abs()*hx+c.abs()*hy
    # Bounding radius cannot be less than actual current collision extent.
    radius=torch.maximum((centres[...,:2].norm(dim=-1)+(half[...,:2]+gait_margin).norm(dim=-1)).amax(-1),centres.new_full((n,),turn_radius))
    pad=radius*(hi-lo)/(2*(samples-1))
    out_c=torch.stack((x+shift[:,None,None,0]/2,y+shift[:,None,None,1]/2,centres[:,None,:,2].expand(-1,samples,-1)),-1)
    out_h=torch.stack((ex+shift[:,None,None,0].abs()/2+pad[:,None,None],ey+shift[:,None,None,1].abs()/2+pad[:,None,None],half[:,None,:,2].expand(-1,samples,-1)),-1)
    out_c=out_c.reshape(n,-1,3);out_h=out_h.reshape(n,-1,3)
    required=torch.ones(out_c.shape[:2],device=centres.device,dtype=torch.bool)
    sweep=FutureSweep174(out_c,out_h,required,torch.ones(n,device=centres.device,dtype=torch.bool),centres.new_zeros(n,0,2),torch.zeros(n,0,device=centres.device,dtype=torch.bool))
    return sweep,dict(stop_distance=speed*latency+speed.square()/(2*deceleration),stop_angle=theta,radius=radius,angular_pad=pad)


class AnalyticSweep177:
    def __init__(self,settings):self.settings=dict(settings);self.last={}
    def build(self,nav,support,grid,*,projection_only=True):
        from gd_lab.core.camera_geometry import camera_rotation_matrix
        from .free_evidence167 import _rotation
        from .obs_terms import world_to_sample
        d=nav.robot.data;pose=nav.pose();rot=_rotation(-pose[:,3])[:,None]@camera_rotation_matrix(d.body_quat_w)
        pos=world_to_sample(d.body_pos_w,pose);centres=[];half=[]
        for index,kind,offset,axis,size in nav.footprint.parts:
            centres.append(pos[:,index]+torch.einsum('nij,j->ni',rot[:,index],offset))
            half.append((rot[:,index]@axis).abs()@size if kind=='box' else torch.ones_like(pos[:,index])*size)
        cmds=nav.control.commands.commands
        # Requested candidate, not the previous supervisor-zeroed command.
        v=pose.new_tensor([[c.v_cap,0.] if c.requested_mode=='NAV' else [0.,0.] for c in cmds])
        omega=pose.new_tensor([c.turn_rate if c.requested_mode=='TURN' else 0. for c in cmds])
        sweep,self.last=stopping_sweep177(torch.stack(centres,1),torch.stack(half,1),v,omega,control_dt=nav.env.step_dt,**self.settings)
        return sweep
