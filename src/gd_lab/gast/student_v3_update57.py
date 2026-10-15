"""Production-only wrappers. Training heads remain in checkpoints, never ONNX.
The cache ABI passes an UNMASKED projection; freshness masking belongs to tick.
"""
import torch
from torch import nn
from gd_lab.gast.student_v3_update55 import AsyncBeliefStudent, capture_transform
from gd_lab.gast.student_v3 import relative_pose

class ProductionEncode(nn.Module):
    def __init__(self, student, cache=False):
        super().__init__(); self.student=student; self.cache=cache
    def forward(self, frames, relative_pose, age, accepted, reset, h_visual):
        h, _, _, quality=self.student.encode(frames,relative_pose,age,accepted,reset,h_visual)
        if self.cache:
            return h, quality, self.student.visual_projection(h[:,:5984])
        return h, quality

def projected_tick(s, proprio, previous_action, dt, age, valid, motion, reset, projection, h_belief, new_packet=None, capture_to_now=None):
    h=torch.where(reset[:,None],torch.zeros_like(h_belief),h_belief)
    previous_action=torch.where(reset[:,None],torch.zeros_like(previous_action),previous_action)
    live=valid & (age[:,0]>=0) & (age[:,0]<.3)
    asynchronous=isinstance(s,AsyncBeliefStudent)
    event=live & new_packet if asynchronous else live
    context=torch.where(event[:,None],projection,torch.zeros_like(projection))
    x=torch.cat((context,proprio,previous_action,age.clamp(0,1),live[:,None].to(proprio.dtype),motion,dt),-1)
    if asynchronous:
        transform=torch.where(live[:,None,None],capture_to_now,torch.zeros_like(capture_to_now))
        x=torch.cat((x,event[:,None].to(proprio.dtype),transform.flatten(1)),-1)
    if getattr(s,'predict_correct',False):
        predicted=s.predictor(torch.cat((proprio,previous_action,motion,dt),-1),h)
        corrected=s.belief(x,predicted)
        h=torch.where(event[:,None],corrected,predicted)
    else: h=s.belief(x,h)
    return s.head(h),h

class CachedTick(nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,proprio,previous_action,dt,age,valid,motion,reset,visual_projection,h_belief):
        return projected_tick(self.student,proprio,previous_action,dt,age,valid,motion,reset,visual_projection,h_belief)
class AsyncCachedTick(CachedTick):
    def forward(self,proprio,previous_action,dt,age,valid,motion,reset,visual_projection,h_belief,new_packet,capture_to_now):
        return projected_tick(self.student,proprio,previous_action,dt,age,valid,motion,reset,visual_projection,h_belief,new_packet,capture_to_now)

class ProjectionCache:
    """Training graph kept until detach; refresh after every optimizer update.
    Owners: encode calls publish, tick reads; reset is an epoch barrier.
    """
    def __init__(self,student,n,device):
        self.student=student;self.value=torch.zeros(n,student.visual_projection[0].out_features,device=device)
        self.valid=torch.zeros(n,dtype=torch.bool,device=device)
    def publish(self,hidden,ids):
        self.value=self.value.index_copy(0,ids,self.student.visual_projection(hidden[ids,:5984]));self.valid[ids]=True
    def invalidate(self,rows):
        self.valid &= ~rows
        self.value=torch.where(rows[:,None],torch.zeros_like(self.value),self.value)
    def detach(self):self.value=self.value.detach()
