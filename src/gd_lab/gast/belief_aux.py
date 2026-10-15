"""UPDATE54 training-only belief map. Never attach this module to the student.

Grid is yaw-aligned 11(y)x17(x), spacing .1 m. Height is metres in the
scanner_z - ground_z - .5 convention. Support is local levelness, NOT safety.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from gd_lab.gast.b2_objective import GroupMean
from gd_lab.students.gavd.model import spatial_targets


@torch.no_grad()
def aligned_targets(terrain, capture_pose, capture_z, current_pose, current_z, eligible):
    """Inverse resample delivered capture into current yaw grid, shift height datum.

    Bilinear height is used only if ALL contributing cells have visible,
    finite, unsaturated height. Out-of-footprint visibility is not a negative.
    Saturated capture heights cannot be inverted and never supervise geometry.
    Targets depend only on delivered capture labels and current odometry/datum.
    """
    terrain, capture_pose, capture_z, current_pose, current_z = (
        v.detach() for v in (terrain, capture_pose, capture_z, current_pose, current_z))
    n = len(terrain)
    good = eligible.bool() & torch.isfinite(capture_pose).all(-1) & torch.isfinite(current_pose).all(-1)
    good &= torch.isfinite(capture_z) & torch.isfinite(current_z)
    cp = torch.nan_to_num(capture_pose); now = torch.nan_to_num(current_pose)
    ys, xs = torch.meshgrid(torch.linspace(-.5,.5,11,device=terrain.device,dtype=terrain.dtype),
                           torch.linspace(-.8,.8,17,device=terrain.device,dtype=terrain.dtype), indexing='ij')
    x, y = xs[None], ys[None]
    c, s = now[:,2,None,None].cos(), now[:,2,None,None].sin()
    wx = c*x-s*y+now[:,0,None,None]-cp[:,0,None,None]
    wy = s*x+c*y+now[:,1,None,None]-cp[:,1,None,None]
    c, s = cp[:,2,None,None].cos(), cp[:,2,None,None].sin()
    grid = torch.stack(((c*wx+s*wy)/.8,(-s*wx+c*wy)/.5),-1)
    inside = (grid.abs() <= 1+1e-6).all(-1) & good[:,None,None]
    grid = grid.clamp(-2,2)
    h = terrain[:,:187].reshape(n,11,17)/5
    visible = (terrain[:,187:].reshape(n,11,17) > .5)
    valid = visible & torch.isfinite(h) & (h.abs() < 1)
    def sample(v):
        return F.grid_sample(v[:,None].to(terrain.dtype),grid,mode='bilinear',padding_mode='zeros',align_corners=True)[:,0]
    valid_now = (sample(valid) >= 1-1e-6) & inside
    # A visibility label describes availability from this delivered capture,
    # not the field of view of a hypothetical current image.
    visible_now = (sample(visible) >= 1-1e-6) & inside
    hnow = sample(torch.where(valid,h,0)) + torch.nan_to_num(current_z-capture_z)[:,None,None]
    valid_now &= torch.isfinite(hnow) & (hnow.abs() < 1)
    packed = torch.cat((torch.where(valid_now,hnow,0).flatten(1)*5,valid_now.flatten(1).float()),-1)
    target, mask = spatial_targets(packed)
    target[...,1] = visible_now.flatten(1).float()
    mask[...,1] = inside.flatten(1)
    return target, mask


def row_losses(prediction, target, mask):
    target, mask = target.detach(), mask.detach()
    safe = torch.where(mask,target,torch.zeros_like(target))
    error = torch.cat((F.smooth_l1_loss(prediction[...,:1],safe[...,:1],reduction='none'),
                       F.binary_cross_entropy_with_logits(prediction[...,1:],safe[...,1:],reduction='none')),-1)
    weight = mask.to(error.dtype).clone()
    weight[...,2:4] *= 1+4*safe[...,2:4]
    denominators = weight.sum(1).clamp_min(1)
    rows = ((torch.where(mask,error,torch.zeros_like(error))*weight).sum(1)/denominators).mean(-1)
    return rows, denominators


class BeliefAux:
    """Separate head/checkpoint/optimizer parameters; lambda=0 is a true bypass."""
    def __init__(self, belief_dim, coefficient=0., device='cpu'):
        self.coefficient = float(coefficient)
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('v3_aux_lambda must be finite and nonnegative')
        self.head = None
        if self.coefficient:
            # Auxiliary initialization must not consume the rollout RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.head = nn.Sequential(nn.Linear(belief_dim,128),nn.ELU(),nn.Linear(128,187*5)).to(device)
        self.reset()
    def reset(self):
        self.group = GroupMean('prep4')
        self.edge_coefficients = {'prep':[], 'nonprep':[]}
        self.cells = [0]*5
    def parameters(self):return [] if self.head is None else list(self.head.parameters())
    def state_dict(self):return None if self.head is None else self.head.state_dict()
    def add(self, belief, target, mask, groups):
        if self.head is None:return
        rows, den = row_losses(self.head(belief).reshape(-1,187,5),target,mask)
        usable = mask.any(-1).any(-1)
        self.group.add(rows,groups,usable)
        self.cells = [a+int(b) for a,b in zip(self.cells,mask.sum((0,1)))]
        for prep, name in ((True,'prep'),(False,'nonprep')):
            selected = usable & ((groups==1) if prep else (groups!=1))
            pos = mask[selected,:,2:4] & (target[selected,:,2:4] > .5)
            # Before group mean and lambda: edge-positive / per-channel weighted
            # denominator / five channels. Final coefficients logged in value().
            coeff = (5/den[selected,2:4]/5)[:,None,:].expand(-1,187,-1)[pos]
            if coeff.numel():self.edge_coefficients[name].extend([float(coeff.min()),float(coeff.max())])
    def value(self, base_loss):
        if self.head is None:return base_loss, {'aux_lambda':0.,'aux_loss':0.}
        aux = self.group.value(base_loss.new_zeros(()))
        np = self.group.counts['prep']; nn = self.group.counts['flat']+self.group.counts['other']
        norm = (4 if np else 0)+(1 if nn else 0)
        effective = {}
        for name,count,factor in (('prep',np,4),('nonprep',nn,1)):
            values = self.edge_coefficients[name]
            scale = self.coefficient*factor/(max(1,norm)*max(1,count))
            effective[name] = [min(values)*scale,max(values)*scale] if values else None
        return base_loss+self.coefficient*aux, {'aux_lambda':self.coefficient,'aux_loss':float(aux.detach()),
            'aux_group_n':dict(self.group.counts),'aux_valid_cells':self.cells,
            'aux_positive_edge_effective_coefficient_minmax':effective,
            'aux_nominal_prep_positive_edge_multiplier':20,'aux_nominal_nonprep_positive_edge_multiplier':5}


class LookaheadAux(BeliefAux):
    """UPDATE55: one training-only module for UPDATE54 map and lookahead.

    Forward-cell height/support reuse the SAME 187x5 map head. Unknown
    support remains masked, never relabeled as zero. No distance regressor.
    The action-conditioned transition is never part of BeliefStudent.
    """
    horizons=(10,20,40)  # 100 Hz: .1/.2/.4 seconds
    def __init__(self,belief_dim,coefficient=0.,front_coefficient=0.,future_coefficient=0.,device='cpu'):
        for c in (coefficient,front_coefficient,future_coefficient):
            if not math.isfinite(c) or c<0:raise ValueError('aux coefficients must be finite and nonnegative')
        super().__init__(belief_dim,max(coefficient,front_coefficient),device)
        self.coefficient=float(coefficient);self.front_coefficient=float(front_coefficient)
        self.future_coefficient=float(future_coefficient);self.future=None
        if future_coefficient:
            with torch.random.fork_rng(devices=[]):
                self.future=nn.ModuleDict({'initial':nn.Linear(belief_dim,64),
                    'transition':nn.GRUCell(12,64),'readout':nn.Linear(64,32)}).to(device)
        self.reset()
    def reset(self):
        super().reset();self.front=GroupMean('prep4');self.sequence=[];self.front_cells=[0,0]
    def parameters(self):
        return super().parameters()+([] if self.future is None else list(self.future.parameters()))
    def state_dict(self):
        if self.head is None and self.future is None:return None
        return {'map':super().state_dict(),'future':None if self.future is None else self.future.state_dict()}
    def load_state_dict(self,state):
        if state is None:return
        # UPDATE54 map-only checkpoints remain usable.
        if 'map' not in state:state={'map':state,'future':None}
        if self.head is not None and state['map'] is not None:self.head.load_state_dict(state['map'],strict=True)
        if self.future is not None and state['future'] is not None:self.future.load_state_dict(state['future'],strict=True)
    def add(self,belief,target,mask,groups):
        if self.coefficient:super().add(belief,target,mask,groups)
        if not self.front_coefficient:return
        # x>0, all lateral cells; height and local support only.
        select=torch.arange(187,device=belief.device)%17>8
        pred=self.head(belief).reshape(-1,187,5)[:,select][:,:,[0,4]]
        valid=mask[:,select][:,:,[0,4]].detach()
        safe=torch.where(valid,target[:,select][:,:,[0,4]].detach(),0)
        err=torch.cat((F.smooth_l1_loss(pred[...,:1],safe[...,:1],reduction='none'),
                       F.binary_cross_entropy_with_logits(pred[...,1:],safe[...,1:],reduction='none')),-1)
        rows=(torch.where(valid,err,0).sum(1)/valid.sum(1).clamp_min(1)).mean(-1)
        self.front.add(rows,groups,valid.flatten(1).any(-1))
        self.front_cells=[a+int(b) for a,b in zip(self.front_cells,valid.sum((0,1)))]
    def add_transition(self,belief,record):
        if self.future is None:return
        self.sequence.append((belief,record))
    def future_loss(self,zero):
        if self.future is None or not self.sequence:return zero,{}
        device=zero.device
        hs=torch.stack([v[0] for v in self.sequence]);records=[v[1] for v in self.sequence]
        def stack(k):return torch.stack([r[k].detach().to(device) for r in records])
        actions=stack('executed_action');targets=stack('teacher_execution_raw')
        epochs=stack('epoch');done=stack('done').bool();groups=stack('group')
        steps=torch.tensor([r['step'] for r in records],device=device)
        total=zero;counts={}
        for horizon in self.horizons:
            length=len(records)-horizon
            if length<=0:counts[str(horizon)]=0;continue
            valid=torch.ones_like(epochs[:length],dtype=torch.bool)
            for offset in range(horizon):
                valid &= ~done[offset:offset+length]
                valid &= epochs[offset+1:offset+1+length]==epochs[:length]
                valid &= (steps[offset+1:offset+1+length]==steps[:length]+offset+1)[:,None]
            # Optional loader loss origin mask: future suffix supplies labels only.
            valid &= torch.tensor([r.get('loss_origin',True) for r in records[:length]],device=device)[:,None]
            valid &= torch.isfinite(targets[horizon:]).all(-1)
            for offset in range(horizon):valid &= torch.isfinite(actions[offset:offset+length]).all(-1)
            counts[str(horizon)]=int(valid.sum())
            if not valid.any():continue
            latent=self.future['initial'](hs[:length][valid])
            for offset in range(horizon):
                latent=self.future['transition'](actions[offset:offset+length][valid].detach(),latent)
            prediction=self.future['readout'](latent)
            error=(prediction-targets[horizon:][valid].detach()).square().mean(-1)
            group=GroupMean('prep4');group.add(error,groups[:length][valid]);total=total+group.value(zero)
        return total/len(self.horizons),counts
    def value(self,base_loss):
        # Preserve exact original tensor and RNG when every coefficient is zero.
        loss,stats=super().value(base_loss) if self.coefficient else (base_loss,{'aux_lambda':0.,'aux_loss':0.})
        if self.front_coefficient:
            front=self.front.value(base_loss.new_zeros(()));loss=loss+self.front_coefficient*front
            stats.update(front_loss=float(front.detach()),front_valid_cells=self.front_cells)
        if self.future_coefficient:
            future,counts=self.future_loss(base_loss.new_zeros(()));loss=loss+self.future_coefficient*future
            stats.update(future_loss=float(future.detach()),future_valid_pairs=counts)
        return loss,stats
