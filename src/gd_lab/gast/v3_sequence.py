"""CPU-stored observation sequences. No saved model hidden is ever consumed.

Burn-in is truncated from zero with current v3 weights. Prefix length is
explicit; it is not claimed to reproduce an infinite episode history.
"""
from collections import deque
import copy
from gd_lab.gast.belief_aux import aligned_targets
from contextlib import nullcontext
import math
import torch
from gd_lab.gast.student_v3_update55 import make_bank,publish,tick
from gd_lab.gast.b2_objective import action_rows
from gd_lab.gast.b3_objective import WindowLoss

CATEGORIES=('flat','stairs','force','landing','recovery')

def slice_row(value,row,n):
    if torch.is_tensor(value):
        value=value.detach().cpu()
        return value[row:row+1].clone() if value.ndim and len(value)==n else value.clone()
    if isinstance(value,dict):
        return {k:({0:slice_row(v[row],row,n)} if row in v else {}) if k=='terminal' and isinstance(v,dict)
                else slice_row(v,row,n) for k,v in value.items() if k not in ('h_visual','h_belief')}
    if isinstance(value,tuple):return tuple(slice_row(v,row,n) for v in value)
    if isinstance(value,list):
        if len(value)==n and all(isinstance(v,str) for v in value):return [value[row]]
        return [slice_row(v,row,n) for v in value]
    return value

def categories(record):
    labels=record['labels'];phase=labels.get('phase',['unknown'])[0];family=labels.get('family',['unknown'])[0]
    result=set()
    if phase=='flat' or family=='v3_flat':result.add('flat')
    if phase=='stairs' or 'stairs' in family:result.add('stairs')
    if phase in ('landing','recovery'):result.add(phase)
    for terminal in record.get('terminal',{}).values():
        if terminal.get('phase') in ('landing','recovery'):result.add(terminal['phase'])
    if any(torch.is_tensor(v) and bool((v!=0).any()) for v in labels.get('force_world',{}).values()):result.add('force')
    return result

class SequenceLoader:
    """Contiguous train origins plus detached future suffix; never pad across files.
    length counts control ticks. BPTT in the online runner counts captures.
    """
    def __init__(self,length=16,burn_in=16,future=40):
        if length not in (16,32,64) or burn_in<1 or future<0:raise ValueError('sequence length/burn-in/future')
        self.length=length;self.burn_in=burn_in;self.future=future
    @property
    def span(self):return self.burn_in+self.length+self.future
    def split(self,records):
        if len(records)!=self.span:raise ValueError('incomplete sequence')
        if any(b['step']!=a['step']+1 for a,b in zip(records,records[1:])):raise ValueError('noncontiguous sequence')
        return records[:self.burn_in],records[self.burn_in:]
    def load(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        records=saved['ticks'];n=len(records[0]['epoch']) if records else 0
        for start in range(0,len(records)-self.span+1,self.length):
            for row in range(n):
                seq=[slice_row(r,row,n) for r in records[start:start+self.span]]
                if not all('tick_packets' in r and 'actor_base' in r for r in seq):raise ValueError('UPDATE55 sequence recording required')
                self.split(seq);yield seq

class SequenceReplay:
    def __init__(self,length=16,burn_in=16,capacity=8):
        if capacity<1:raise ValueError('replay capacity')
        self.loader=SequenceLoader(length,burn_in,future=0)
        self.capacity=capacity;self.buckets={k:deque(maxlen=capacity) for k in CATEGORIES}
        self.history=None;self.added=0
    def append(self,record):
        n=len(record['epoch'])
        if self.history is None:self.history=[deque(maxlen=self.loader.span) for _ in range(n)]
        for row in range(n):
            history=self.history[row];history.append(slice_row(record,row,n))
            if len(history)<self.loader.span:continue
            # Place each observed category in its own bounded queue. Common
            # flat samples cannot evict rare stair/force/landing/recovery samples.
            tags=categories(history[-1])
            if not tags:continue
            seq=tuple(history)
            for tag in tags:self.buckets[tag].append(seq)
            self.added+=1
    def samples(self):
        # One sequence PER populated category, rotating without global RNG use.
        selected=[]
        for tag,bucket in self.buckets.items():
            if bucket:
                selected.append((tag,bucket[0]));bucket.rotate(-1)
        return selected
    def coverage(self):return {tag:len(q) for tag,q in self.buckets.items()}

def to_device(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(to_device(v,device) for v in value)
    return value

def replay_loss(model,actor,records,burn_in,auxiliary=None,train_length=None,aux_only=False):
    """Burn in BOTH visual and belief from v3's observations, then B3 loss.
    Recorded 34624 action is a fixed target. No critic/EWC/V-trace.
    """
    if not 0<burn_in<len(records):raise ValueError('burn-in needs nonempty training suffix')
    if any(b['step']!=a['step']+1 for a,b in zip(records,records[1:])):raise ValueError('noncontiguous replay')
    device=next(model.parameters()).device;n=len(records[0]['epoch'])
    bank=make_bank(model,n,device);bank.episode.copy_(records[0]['epoch'].to(device))
    window=WindowLoss('prep4')
    aux=copy.copy(auxiliary) if auxiliary is not None else None
    if aux is not None:aux.reset()
    train_length=len(records)-burn_in if train_length is None else train_length
    if train_length<1 or burn_in+train_length>len(records):raise ValueError('training sequence length')
    for index,stored in enumerate(records):
        record=to_device(stored,device);step=record['step'];train=burn_in<=index<burn_in+train_length
        received={}
        with nullcontext() if train else torch.no_grad():
            if not torch.equal(bank.episode,record['epoch']):raise ValueError('replay reset/epoch boundary mismatch')
            for packet in record['tick_packets']:
                frames,epochs,extra,groups,contract=packet['payload']
                ids,raw,spatial=publish(bank,frames,extra[3],packet['capture_step'],packet['delivery_step'],epochs,step,contract['root_z'])
                if train and aux is not None and aux.head is not None:
                    for row in ids.tolist():received[row]=(extra[2][row],extra[3][row],contract['scanner_z'][row],epochs[row])
                if train and not aux_only and len(ids):
                    window.raw.add((raw-extra[6][ids].detach()).square().mean(-1),groups[ids])
                    window.spatial(spatial,extra[1][ids],extra[2][ids])
            bank.previous_action=record['previous_action'].detach()
            raw=tick(bank,record['proprio'],record['pose'],step,record['root_z'])
            if received:
                rows=torch.tensor(sorted(received),device=device)
                terrain,cap_pose,cap_z,epoch=zip(*(received[row] for row in sorted(received)))
                target,mask=aligned_targets(torch.stack(terrain),torch.stack(cap_pose),torch.stack(cap_z),record['pose'][rows],record['scanner_z'][rows],
                    (torch.stack(epoch)==bank.episode[rows]) & bank.inputs(step)[1][rows])
                aux.add(bank.belief[rows],target,mask,record['group'][rows])
            if aux is not None and index>=burn_in:
                aux.add_transition(bank.belief,dict(record,loss_origin=train))
            if train and not aux_only:
                gate=bank.effective(step).detach()
                values,action=action_rows(actor,record['actor_base'].detach(),raw,record['teacher_execution_raw'].detach(),gate)
                window.action.add(values,record['group'],gate[:,0]>0)
                window.preserve((action-record['baseline_action'].detach()).square().mean(-1),record['group'],torch.ones(n,device=device,dtype=torch.bool))
            bank.reset(record['done'].reshape(-1).bool(),step+1)
        if index==burn_in-1:bank.detach()
    loss,stats=window.value(next(model.parameters()).new_zeros(()))
    if aux is not None:
        loss,aux_stats=aux.value(loss);stats.update(aux_stats)
    stats.update(burn_in_ticks=burn_in,train_ticks=train_length,hidden_source='current_v3_observation_burn_in_from_zero')
    return loss,stats
