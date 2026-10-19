"""Reward-only GT stair-end detector (D-8, 2026-10-19 user rule).

Stair stepping seen, then flat ground ahead at the level of the four feet -> stop
(landing or final floor alike, like a guide dog). Body pitch is not required.

Never used to construct actor features or override commands. Thresholds are
initial training assumptions, not deployment validation. A continuously high
GO/heartbeat is not a new GO. The shared release input is an accepted new
request (rising GO or a new manoeuvre), independent of source AUTO/GDM/USER.
"""
import torch

class StairEnd184:
    def __init__(self,n,device,cfg):
        self.cfg=cfg
        self.last_z=torch.zeros(n,4,device=device)
        self.contact_seen=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.step_seen=torch.zeros(n,device=device,dtype=torch.bool)
        self.pitch_seen=self.step_seen.clone();self.wait=self.step_seen.clone()
        self.flat_time=torch.zeros(n,device=device);self.event=self.step_seen.clone()
        self.initialized=self.step_seen.clone();self.previous_go=self.step_seen.clone()
        self.last_request=torch.zeros(n,device=device,dtype=torch.long)
        self.total_events=torch.zeros_like(self.last_request)
    def reset(self,ids):
        for name in ('last_z','contact_seen','step_seen','pitch_seen','wait','flat_time','event','initialized','previous_go','last_request','total_events'):
            getattr(self,name)[ids]=0
    def tick(self,*,foot_z,contacts,pitch,roll,stair_surface,nav,go,request_id,dt,ahead_flat=None):
        # Contact-height changes only: swing-foot arcs on flat ground/ramp
        # cannot arm the label. GT stair_surface is reward-only topology.
        finite=torch.isfinite(foot_z).all(-1)&torch.isfinite(pitch)&torch.isfinite(roll)
        jump=((foot_z-self.last_z).abs()>=self.cfg['step_height_m']) & contacts & self.contact_seen
        self.step_seen |= jump.any(-1)&stair_surface&nav&finite
        self.pitch_seen |= (pitch.abs()>=self.cfg['step_pitch_rad'])&stair_surface&nav&finite
        self.last_z=torch.where(contacts,foot_z,self.last_z)
        self.contact_seen |= contacts
        flat=contacts.all(-1)&finite&((foot_z.amax(-1)-foot_z.amin(-1))<=self.cfg['same_height_m'])
        flat &= (pitch.abs()<=self.cfg['level_angle_rad'])&(roll.abs()<=self.cfg['level_angle_rad'])&nav
        self.flat_time=torch.where(flat,self.flat_time+dt,0.)
        # Once detected, re-arm only after genuinely new stair stepping.
        if ahead_flat is None:
            self.event=self.step_seen&self.pitch_seen&~self.wait&(self.flat_time>=self.cfg['confirm_s'])
        else:
            # New rule: flat ground ahead at the current four-foot level ends the stair.
            self.event=self.step_seen&ahead_flat&~self.wait&(self.flat_time>=self.cfg['confirm_s'])
        fresh_go=go & (~self.previous_go | (request_id!=self.last_request)) & self.initialized
        self.wait=(self.wait&~fresh_go)|self.event # detection tick cannot release itself
        self.step_seen &= ~self.event;self.pitch_seen &= ~self.event
        self.total_events+=self.event.long()
        self.wait &= nav
        self.previous_go=go.clone();self.last_request=request_id.clone();self.initialized[:]=True
        return self.wait
