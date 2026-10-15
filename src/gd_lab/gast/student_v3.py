"""Explicit-state 4-view GAST encoder and 100 Hz body-conditioned belief.
No privileged labels enter either graph. All pose arguments are relative odometry.
"""
import copy
import torch
from torch import nn
from gd_lab.gast.student import GastStudent

SIZES = {'small': (128, 256), 'large': (512, 896)}

class BeliefStudent(nn.Module):
    architecture = 'gast_belief_v3'
    def __init__(self, visual=None, size='small'):
        super().__init__()
        self.visual = copy.deepcopy(visual) if visual is not None else GastStudent()
        # Original head is replaced; quality feature extractor stays fixed as well.
        del self.visual.head, self.visual.hazard_head
        width, self.belief_dim = SIZES[size]
        self.size = size
        self.visual_projection = nn.Sequential(nn.Linear(187*32, width), nn.ELU())
        self.belief = nn.GRUCell(width+230+12+2+1+3+1, self.belief_dim)
        self.head = nn.Sequential(nn.Linear(self.belief_dim, 64), nn.ELU(), nn.Linear(64,32), nn.Tanh())
        self.capture_head = nn.Sequential(nn.Flatten(), nn.Linear(187*32,64), nn.ELU(), nn.Linear(64,32), nn.Tanh())
        self.visual.requires_grad_(False)
        for name in ('cell_gru','temporal_attention','age_embed','spatial_head'):
            getattr(self.visual,name).requires_grad_(True)
        self.eval()
    def init_hidden(self,n,device):
        return self.visual.init_hidden(n,device), torch.zeros(n,self.belief_dim,device=device)
    def encode(self,frames,relative_pose,age,accepted,reset,h_visual):
        """relative_pose: dx,dy,dyaw in previous capture yaw frame + current
        quaternion in that same frame (wxyz). Output pose slots zero: host owns pose.
        age [B,1] capture->delivery; accepted/reset [B] for synchronized 4-view packet.
        """
        hidden=torch.where(reset[:,None],torch.zeros_like(h_visual),h_visual)
        # Forward copied from v1, with tensor ages and relative coordinates only.
        memory,spatial,quality=self._visual(frames,hidden,relative_pose,age)
        memory=torch.cat((memory[:,:-4],torch.zeros_like(memory[:,-4:-1]),memory[:,-1:]),-1)
        output=torch.where(accepted[:,None],memory,hidden)
        raw_capture=self.capture_head(output[:,:5984].reshape(-1,187,32))
        return output,raw_capture,spatial,quality.detach()
    def _visual(self,frames,hidden,pose,age):
        return self.visual.encode_v3(frames,hidden,pose,age)
    def tick(self,proprio,previous_action,dt,age,fresh,relative_motion,reset,h_visual,h_belief):
        h=torch.where(reset[:,None],torch.zeros_like(h_belief),h_belief)
        previous_action=torch.where(reset[:,None],torch.zeros_like(previous_action),previous_action)
        live=fresh & (age[:,0]>=0) & (age[:,0]<.3)
        context=self.visual_projection(h_visual[:,:5984])
        context=torch.where(live[:,None],context,torch.zeros_like(context))
        x=torch.cat((context,proprio,previous_action,age.clamp(0,1),live[:,None].to(proprio.dtype),relative_motion,dt),-1)
        h=self.belief(x,h)
        return self.head(h),h

class EncodeGraph(nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,frames,relative_pose,age,accepted,reset,h_visual):
        return self.student.encode(frames,relative_pose,age,accepted,reset,h_visual)
class TickGraph(nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,proprio,previous_action,dt,age,fresh,relative_motion,reset,h_visual,h_belief):
        return self.student.tick(proprio,previous_action,dt,age,fresh,relative_motion,reset,h_visual,h_belief)

def relative_pose(current,previous):
    """Host adapter for simulator pose_xyyaw; replace source with odometry on robot."""
    yaw=previous[:,2];delta=current[:,:2]-previous[:,:2]
    dx=yaw.cos()*delta[:,0]+yaw.sin()*delta[:,1]
    dy=-yaw.sin()*delta[:,0]+yaw.cos()*delta[:,1]
    dyaw=torch.atan2((current[:,2]-yaw).sin(),(current[:,2]-yaw).cos())
    w,x,y,z=current[:,3:7].unbind(-1);c=(yaw/2).cos();s=(yaw/2).sin()
    q=torch.stack((c*w+s*z,c*x+s*y,c*y-s*x,c*z-s*w),-1)
    return torch.cat((torch.stack((dx,dy,dyaw),-1),q),-1)
