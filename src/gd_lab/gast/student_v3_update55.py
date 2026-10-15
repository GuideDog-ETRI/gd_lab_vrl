"""Optional event-conditioned single GRU; no training labels in graph ABI."""
import torch
from torch import nn
from gd_lab.gast.student_v3 import BeliefStudent, relative_pose
from gd_lab.gast.v3_state import BeliefBank

class AsyncBeliefStudent(BeliefStudent):
    architecture = 'gast_belief_v3_async55'
    def __init__(self, visual=None, size='small',predict_correct=False):
        super().__init__(visual,size)
        self.predict_correct=predict_correct
        if predict_correct:
            self.architecture='gast_belief_v3_predict_correct55'
            with torch.random.fork_rng(devices=[]):
                self.predictor=nn.GRUCell(230+12+3+1,self.belief_dim)
        old=self.belief
        # Add new_packet and flattened SE(3) rotation/translation (3x4).
        with torch.random.fork_rng(devices=[]):
            self.belief=nn.GRUCell(old.input_size+13,old.hidden_size)
        with torch.no_grad():
            self.belief.weight_ih[:,:old.input_size].copy_(old.weight_ih)
            self.belief.weight_ih[:,old.input_size:].zero_()
            for name in ('weight_hh','bias_ih','bias_hh'):
                getattr(self.belief,name).copy_(getattr(old,name))
    def tick(self,proprio,previous_action,dt,age,visual_valid,relative_motion,reset,h_visual,h_belief,new_packet,capture_to_now):
        h=torch.where(reset[:,None],torch.zeros_like(h_belief),h_belief)
        previous_action=torch.where(reset[:,None],torch.zeros_like(previous_action),previous_action)
        valid=visual_valid & (age[:,0]>=0) & (age[:,0]<.3)
        event=valid & new_packet
        context=self.visual_projection(h_visual[:,:5984])
        context=torch.where(event[:,None],context,torch.zeros_like(context))
        transform=torch.where(valid[:,None,None],capture_to_now,torch.zeros_like(capture_to_now))
        x=torch.cat((context,proprio,previous_action,age.clamp(0,1),valid[:,None].to(proprio.dtype),
                     relative_motion,dt,event[:,None].to(proprio.dtype),transform.flatten(1)),-1)
        if self.predict_correct:
            predicted=self.predictor(torch.cat((proprio,previous_action,relative_motion,dt),-1),h)
            corrected=self.belief(x,predicted)
            h=torch.where(event[:,None],corrected,predicted)
        else:
            h=self.belief(x,h)
        return self.head(h),h

class AsyncTickGraph(nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,proprio,previous_action,dt,age,visual_valid,relative_motion,reset,h_visual,h_belief,new_packet,capture_to_now):
        return self.student.tick(proprio,previous_action,dt,age,visual_valid,relative_motion,reset,h_visual,h_belief,new_packet,capture_to_now)

def rotation(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(-1,3,3)

def capture_transform(now,now_z,capture,capture_z):
    """Upper 3x4 of T_now^-1 T_capture, full body SE(3), wxyz quaternions."""
    rn=rotation(now[:,3:]).transpose(1,2);rc=rotation(capture[:,3:])
    delta=torch.cat((capture[:,:2]-now[:,:2],(capture_z-now_z)[:,None]),-1)
    return torch.cat((rn@rc,(rn@delta[:,:,None])),-1)

class AsyncBeliefBank(BeliefBank):
    def __init__(self,student,n,device):
        super().__init__(student,n,device)
        self.new_packet=torch.zeros_like(self.ready);self.capture_z=self.gate[:,0].clone()
    def reset(self,done,step):
        super().reset(done,step);self.new_packet &= ~done
    def publish(self,frames,pose,capture,delivery,epochs,step,*,capture_z):
        if step<=self.last_tick:raise ValueError('publish must precede its tick; duplicate/reversed clock')
        if capture_z.shape!=self.capture_z.shape or not torch.isfinite(capture_z).all():raise ValueError('capture root z')
        result=super().publish(frames,pose,capture,delivery,epochs,step)
        ids=result[0];self.new_packet[ids]=True;self.capture_z[ids]=capture_z[ids].detach()
        return result
    def tick(self,proprio,pose,step,*,pose_z):
        if step<=self.last_tick:raise ValueError('duplicate/reversed tick: new bank epoch required')
        if not torch.isfinite(pose).all() or not torch.isfinite(pose_z).all():raise ValueError('nonfinite odometry')
        age,valid=self.inputs(step)
        motion=relative_pose(pose,torch.where(self.first[:,None],pose,self.tick_pose))[:,:3]
        dt=proprio.new_full((len(proprio),1),.01 if self.last_tick<0 else (step-self.last_tick)*.01)
        dt=torch.where(self.first[:,None],torch.full_like(dt,.01),dt)
        transform=capture_transform(pose,pose_z,self.capture_pose,self.capture_z)
        raw,self.belief=self.student.tick(proprio,self.previous_action,dt,age,valid,motion,
            torch.zeros_like(self.first),self.hidden,self.belief,self.new_packet,transform)
        self.new_packet.zero_();self.tick_pose=pose.detach().clone();self.first.zero_();self.last_tick=step
        return raw

def make_student(visual=None,size='small',asynchronous=False,predict_correct=False):
    if predict_correct and not asynchronous:raise ValueError('predict/correct requires explicit async option')
    return AsyncBeliefStudent(visual,size,predict_correct) if asynchronous else BeliefStudent(visual,size)

def make_bank(student,n,device):
    return (AsyncBeliefBank if isinstance(student,AsyncBeliefStudent) else BeliefBank)(student,n,device)

def publish(bank,frames,pose,capture,delivery,epochs,step,root_z):
    kw={'capture_z':root_z} if isinstance(bank,AsyncBeliefBank) else {}
    return bank.publish(frames,pose,capture,delivery,epochs,step,**kw)

def tick(bank,proprio,pose,step,root_z):
    kw={'pose_z':root_z} if isinstance(bank,AsyncBeliefBank) else {}
    return bank.tick(proprio,pose,step,**kw)
