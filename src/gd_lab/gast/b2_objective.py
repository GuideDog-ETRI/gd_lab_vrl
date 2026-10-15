"""UPDATE31 frozen-actor student-content training; simulator-independent core."""
import torch
from torch.nn import functional as F
from gd_lab.gast.geometry import targets

MODES=('memory_head','all_but_quality')
GROUPS=('flat','prep','other')

def configure_student(student,mode):
    if mode not in MODES:raise ValueError(mode)
    prefixes=('cell_gru.','temporal_attention.','age_embed.','head.')
    for name,p in student.named_parameters():
        p.requires_grad_(name.startswith(prefixes) if mode=='memory_head' else not name.startswith('quality.'))
        p.grad=None
    student.eval() # fixed inference behavior; autograd remains enabled
    return [p for p in student.parameters() if p.requires_grad]

def groups(distance,flat,active):
    prep=active & torch.isfinite(distance) & (distance>=.3) & (distance<=1.)
    # One group per row; flat cannot override preparation.
    return torch.where(prep,1,torch.where(flat,0,2)).long()

class GroupMean:
    def __init__(self,mode):
        if mode not in ('equal3','prep4'):raise ValueError(mode)
        self.mode=mode;self.sums={};self.counts={k:0 for k in GROUPS}
    def add(self,values,labels,valid=None):
        valid=torch.ones_like(labels,dtype=torch.bool) if valid is None else valid
        for i,name in enumerate(GROUPS):
            mask=valid & (labels==i);n=int(mask.sum())
            if n:
                value=values[mask].sum()
                self.sums[name]=self.sums.get(name,0)+value;self.counts[name]+=n
    def value(self,zero):
        if self.mode=='equal3':
            v=[self.sums[k]/self.counts[k] for k in GROUPS if self.counts[k]]
            return sum(v)/len(v) if v else zero
        # prep4 merges flat and other BEFORE averaging their rows.
        n=self.counts['flat']+self.counts['other'];num=zero;den=0
        if self.counts['prep']:num=num+4*self.sums['prep']/self.counts['prep'];den+=4
        if n:num=num+(self.sums.get('flat',0)+self.sums.get('other',0))/n;den+=1
        return num/den if den else zero

class WindowLoss:
    def __init__(self,mode):
        self.action=GroupMean(mode);self.raw=GroupMean(mode)
        self.spatial_sum=None;self.spatial_weight=0.;self.spatial_rows=0
    def spatial(self,pred,clean,terrain):
        y,mask=targets(clean.detach(),terrain.detach())
        loss=torch.cat((F.smooth_l1_loss(pred[...,:1],y[...,:1],reduction='none'),
                        F.binary_cross_entropy_with_logits(pred[...,1:],y[...,1:],reduction='none')),-1)
        weight=torch.ones_like(loss);weight[...,2:5]=1+4*y[...,2:5]
        weight=weight*mask
        value=(loss*weight).sum()
        self.spatial_sum=value if self.spatial_sum is None else self.spatial_sum+value
        self.spatial_weight+=float(weight.sum());self.spatial_rows+=len(pred)
    def value(self,zero):
        a=self.action.value(zero);r=self.raw.value(zero)
        s=self.spatial_sum/max(1.,self.spatial_weight) if self.spatial_sum is not None else zero
        return a+.1*r+s,{'action':float(a.detach()),'raw':float(r.detach()),'spatial':float(s.detach()),
           'action_n':dict(self.action.counts),'raw_n':dict(self.raw.counts),'spatial_rows':self.spatial_rows}

def action_rows(actor,base,raw,teacher_raw,effective_gate):
    gate=effective_gate.detach()
    actual=actor(torch.cat((base.detach(),raw*gate),-1))
    with torch.no_grad():expected=actor(torch.cat((base.detach(),teacher_raw.detach()*gate),-1))
    return (actual-expected).square().mean(-1),actual

class PacketBank:
    def __init__(self,student,n,device):
        self.hidden=student.init_hidden(n,device);self.raw=torch.zeros(n,32,device=device)
        self.gate=torch.zeros(n,1,device=device);self.stamp=torch.full((n,),-100.,device=device)
        self.ready=torch.zeros(n,dtype=torch.bool,device=device)
        self.episode=torch.zeros(n,dtype=torch.long,device=device)
        self.last=torch.full((n,),-1,dtype=torch.long,device=device)
    def valid(self,capture,captured):return (captured==self.episode)&(capture>self.last)
    def reset(self,done):
        self.hidden=torch.where(done[:,None],torch.zeros_like(self.hidden),self.hidden)
        self.raw=torch.where(done[:,None],torch.zeros_like(self.raw),self.raw)
        self.gate=torch.where(done[:,None],torch.zeros_like(self.gate),self.gate)
        self.ready=self.ready & ~done;self.episode=self.episode+done.long()
        self.stamp=torch.where(done,torch.full_like(self.stamp,-100),self.stamp)
    def publish(self,ids,memory,raw,gate,stamp,capture):
        self.hidden=self.hidden.index_copy(0,ids,memory);self.raw=self.raw.index_copy(0,ids,raw)
        self.gate=self.gate.index_copy(0,ids,gate.detach());self.stamp[ids]=stamp.detach()
        self.last[ids]=capture;self.ready[ids]=True
    def effective(self,now):
        age=now-self.stamp;fresh=self.ready & (age>=0) & (age<.3)
        return self.gate*fresh[:,None]
    def detach(self):self.hidden=self.hidden.detach();self.raw=self.raw.detach();self.gate=self.gate.detach()
