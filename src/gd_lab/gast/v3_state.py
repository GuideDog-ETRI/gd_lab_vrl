"""Host synchronized-packet state; integer ticks avoid 300ms float boundaries."""
import torch
from gd_lab.gast.student_v3 import relative_pose
class BeliefBank:
    def __init__(self,student,n,device):
        self.student=student;self.hidden,self.belief=student.init_hidden(n,device)
        self.episode=torch.zeros(n,dtype=torch.long,device=device)
        self.last=torch.full_like(self.episode,-1);self.delivery=self.last.clone();self.start=torch.zeros_like(self.last)
        self.ready=torch.zeros(n,dtype=torch.bool,device=device)
        self.gate=torch.zeros(n,1,device=device);self.previous_action=torch.zeros(n,12,device=device)
        self.capture_pose=torch.zeros(n,7,device=device);self.capture_pose[:,3]=1
        self.tick_pose=self.capture_pose.clone();self.first=torch.ones_like(self.ready)
        self.last_tick=-1;self.counters={}
    def reset(self,done,step):
        self.hidden=torch.where(done[:,None],torch.zeros_like(self.hidden),self.hidden)
        self.belief=torch.where(done[:,None],torch.zeros_like(self.belief),self.belief)
        self.previous_action=torch.where(done[:,None],torch.zeros_like(self.previous_action),self.previous_action)
        self.gate=torch.where(done[:,None],torch.zeros_like(self.gate),self.gate)
        self.episode+=done.long();self.ready &= ~done;self.first |= done
        self.last=torch.where(done,-1,self.last);self.delivery=torch.where(done,-1,self.delivery)
        self.start=torch.where(done,step,self.start)
    def valid(self,capture,delivery,epochs,step,frames,pose):
        # Future delivery belongs in transport queue, never invoke encode with it.
        valid=(epochs==self.episode)&(capture>=self.start)&(capture>self.last)&(delivery>=self.delivery)
        valid &= (capture<=delivery)&(delivery<=step)
        valid &= torch.isfinite(frames).flatten(1).all(-1)&torch.isfinite(pose).all(-1)
        self.counters['rejected']=self.counters.get('rejected',0)+int((~valid).sum())
        return valid
    def publish(self,frames,pose,capture,delivery,epochs,step):
        if frames.shape[1:]!=(4,2,45,80):raise ValueError('4 views x depth/IR required')
        mask=self.valid(capture,delivery,epochs,step,frames,pose)
        ids=mask.nonzero().flatten()
        if not len(ids):return ids,None,None
        previous=torch.where(self.ready[ids,None],self.capture_pose[ids],pose[ids])
        rel=relative_pose(pose[ids],previous)
        age=self.gate.new_full((len(ids),1),(delivery-capture)*.01)
        yes=torch.ones(len(ids),device=frames.device,dtype=torch.bool)
        memory,raw,spatial,quality=self.student.encode(frames[ids],rel,age,yes,~yes,self.hidden[ids])
        self.hidden=self.hidden.index_copy(0,ids,memory);self.gate=self.gate.index_copy(0,ids,quality.detach())
        self.capture_pose[ids]=pose[ids].detach();self.last[ids]=capture;self.delivery[ids]=delivery;self.ready[ids]=True
        return ids,raw,spatial
    def inputs(self,step):
        capture_age=step-self.last;delivery_age=step-self.delivery
        fresh=self.ready & (capture_age>=0)&(capture_age<30)
        age=torch.stack((capture_age,delivery_age),-1).to(self.hidden.dtype)*.01
        age=torch.where(self.ready[:,None],age,torch.ones_like(age)).clamp(0,1)
        return age,fresh
    def effective(self,step):return self.gate*self.inputs(step)[1][:,None]
    def tick(self,proprio,pose,step):
        if step<=self.last_tick:raise ValueError('duplicate/reversed tick: halt; clock reversal requires new bank epoch')
        age,fresh=self.inputs(step)
        motion=relative_pose(pose,torch.where(self.first[:,None],pose,self.tick_pose))[:,:3]
        dt=proprio.new_full((len(proprio),1),.01 if self.last_tick<0 else (step-self.last_tick)*.01)
        dt=torch.where(self.first[:,None],torch.full_like(dt,.01),dt)
        # Host already cleared state at reset. A new-epoch packet may be used immediately.
        raw,self.belief=self.student.tick(proprio,self.previous_action,dt,age,fresh,motion,
            torch.zeros_like(self.first),self.hidden,self.belief)
        self.tick_pose=pose.detach().clone();self.first.zero_();self.last_tick=step
        return raw
    def detach(self):
        self.hidden=self.hidden.detach();self.belief=self.belief.detach();self.previous_action=self.previous_action.detach()
