"""D-5 measured-motion stop model. Unvalidated command/scene domains close.

Next-swing extrapolation is a diagnostic model, not a learned gait guarantee.
All recorded post-stop collision boxes must fit for a domain to be admitted.
"""
import json
from pathlib import Path
from dataclasses import replace
import torch
from .analytic_sweep177 import stopping_sweep177


def measured_sweep179(centres,half,velocity,omega,relative_link_velocity,*,lateral_drift_m=.05,**settings):
    if relative_link_velocity.shape!=centres.shape or not torch.isfinite(relative_link_velocity).all():
        raise ValueError('measured link velocity required')
    if lateral_drift_m<0:raise ValueError('negative residual drift')
    horizon=settings.get('delay_s',.04)+settings.get('control_dt',.01)+torch.maximum(
        velocity.norm(dim=-1)/settings.get('deceleration',.5),omega.abs()/settings.get('angular_deceleration',1.))
    # Include each moving leg's next displacement through the stop horizon.
    delta=relative_link_velocity*horizon[:,None,None]
    next_c=centres+delta/2
    next_h=half+delta.abs()/2
    next_h=next_h.clone();next_h[...,1]+=lateral_drift_m
    result,stats=stopping_sweep177(next_c,next_h,velocity,omega,**settings)
    stats.update(stop_horizon_s=horizon,next_swing_displacement=delta)
    return result,stats


class MeasuredSweep179:
    def __init__(self,settings):
        self.settings=dict(settings);self.last={}
        self.settings.pop('current_body_margin_m',None)
        self.gate=self.settings.pop('sweep_qualification_gate',True)
        scope=self.settings.pop('scope','deployment')
        if not self.gate and scope not in ('diagnostic','training'):raise ValueError('qualification bypass only diagnostic/training')
        calibration=self.settings.pop('calibration_file',None)
        self.calibration=None
        if calibration:
            path=Path(calibration)
            if not path.is_absolute():path=Path(__file__).resolve().parents[3]/path
            self.calibration=json.loads(path.read_text())
        file=Path(self.settings.pop('qualification_file'))
        if not file.is_absolute():file=Path(__file__).resolve().parents[3]/file
        audit=json.loads(file.read_text()) if file.exists() else {}
        if self.calibration is not None and self.calibration['settings']!=self.settings:raise ValueError('calibration settings mismatch')
        self.domains=audit.get('domains',[]) if audit.get('settings')==self.settings else []

    def build(self,nav,support,grid,*,projection_only=True):
        from gd_lab.core.camera_geometry import camera_rotation_matrix
        from .free_evidence167 import _rotation
        from .obs_terms import world_to_sample
        d=nav.robot.data;pose=nav.pose();yaw=_rotation(-pose[:,3]);rot=yaw[:,None]@camera_rotation_matrix(d.body_quat_w)
        pos=world_to_sample(d.body_pos_w,pose);centres=[];half=[];relative=[]
        rootv=torch.einsum('nij,nj->ni',yaw,d.root_lin_vel_w)
        rootw=torch.einsum('nij,nj->ni',yaw,d.root_ang_vel_w)
        for index,kind,offset,axis,size in nav.footprint.parts:
            off=torch.einsum('nij,j->ni',rot[:,index],offset)
            centre=pos[:,index]+off
            vel=torch.einsum('nij,nj->ni',yaw,d.body_lin_vel_w[:,index])+torch.cross(torch.einsum('nij,nj->ni',yaw,d.body_ang_vel_w[:,index]),off,dim=-1)
            rel=vel-rootv-torch.cross(rootw,centre,dim=-1)
            if nav.robot.body_names[index]=='trunk':rel=torch.zeros_like(rel)
            centres.append(centre);relative.append(rel)
            half.append((rot[:,index]@axis).abs()@size if kind=='box' else torch.ones_like(centre)*size)
        c=torch.stack(centres,1);h=torch.stack(half,1)
        result,self.last=measured_sweep179(c,h,rootv[:,:2],rootw[:,2],torch.stack(relative,1),control_dt=nav.env.step_dt,**self.settings)
        # No nearest-domain fallback. Domains are admitted only after every
        # trial in that command/scene group passes the empirical audit.
        qualified=torch.zeros(len(pose),device=pose.device,dtype=torch.bool)
        metadata=nav.index.lookup(range(len(pose)))
        for i,(meta,cmd) in enumerate(zip(metadata,nav.control.commands.commands)):
            value=cmd.v_cap if cmd.requested_mode=='NAV' else cmd.turn_rate
            for row in self.domains:
                if row['qualified'] and row['kind']==meta.kind and row['mode']==cmd.requested_mode and abs(row['value']-value)<1e-5:
                    v=rootv[i,:2];w=rootw[i,2]
                    qualified[i]=((v>=v.new_tensor(row['velocity_min'])).all()&(v<=v.new_tensor(row['velocity_max'])).all()&(w>=row['omega_min'])&(w<=row['omega_max']))
        self.last['qualified']=qualified
        if self.calibration is not None:
            # Stair variants share a runtime family: use worst variant p95.
            # Unmeasured command combinations use the largest calibrated padding.
            padding=[]
            for meta,cmd in zip(metadata,nav.control.commands.commands):
                value=cmd.v_cap if cmd.requested_mode=='NAV' else cmd.turn_rate
                matches=[r['margin_m'] for r in self.calibration['domains'] if r['kind'].split('/')[0]==meta.kind and r['mode']==cmd.requested_mode and abs(r['value']-value)<1e-5]
                padding.append(max(matches) if matches else self.calibration['padding_m'])
            padding=result.half.new_tensor(padding)
            half=result.half.clone();half[...,:2]+=padding[:,None,None]
            self.last['calibrated_padding_m']=padding
            result=replace(result,half=half)
        self.last['qualification_gate']=self.gate
        return replace(result,qualified=qualified if self.gate else result.qualified)
