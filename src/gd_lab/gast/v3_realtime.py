"""Optional deterministic training host, milliseconds, fixed physics substeps.
No threads/GPU/Pilot imports. Callback advance(dt_seconds, applied_action) must
advance physics exactly one substep; observe() returns available proprio/pose/z.
All timestamps are retained. No fabricated catch-up GRU ticks.
"""
from dataclasses import dataclass
from collections import deque
import random
import math
import torch
from gd_lab.gast.student_v3 import relative_pose
from gd_lab.gast.student_v3_update55 import make_bank, AsyncBeliefBank, AsyncBeliefStudent, capture_transform, publish
from gd_lab.gast.student_v3_update57 import ProjectionCache, projected_tick

@dataclass(frozen=True)
class Timing:
    jitter_ms: int=0
    long_interval_ms: int=20
    long_probability: float=0.
    action_delay_ticks: int=0
    burst_probability: float=0.
    burst_packets: int=3
    vision_delay_ms: int=0
    long_vision_ms: int=1200
    long_vision_probability: float=0.
    encode_ms: int=0
    proprio_delay_ms: int=0
    odometry_std_m: float=0.
    timestamp_offset_ms: float=0.
    timestamp_drift_ppm: float=0.
    seed: int=57
    def __post_init__(self):
        if self.jitter_ms not in (0,1,2) or self.long_interval_ms not in (20,30):raise ValueError('tick interval sweep')
        if self.action_delay_ticks not in (0,1,2):raise ValueError('action delay sweep')
        for k in ('long_probability','burst_probability','long_vision_probability'):
            if not 0<=getattr(self,k)<=1:raise ValueError(k)
        for k in ('burst_packets','vision_delay_ms','long_vision_ms','encode_ms','proprio_delay_ms'):
            if not isinstance(getattr(self,k),int) or getattr(self,k)<0:raise ValueError(k)
        if self.burst_packets<1:raise ValueError('burst_packets')
        for k in ('odometry_std_m','timestamp_offset_ms','timestamp_drift_ppm'):
            if not math.isfinite(getattr(self,k)):raise ValueError(k)
        if self.odometry_std_m<0:raise ValueError('odometry_std_m')

@dataclass(frozen=True)
class Packet:
    sequence: int
    capture_ms: float
    arrival_ms: float
    observed_capture_ms: float
    frames: torch.Tensor
    pose: torch.Tensor
    root_z: torch.Tensor
    epochs: torch.Tensor
    labels: object=None  # consumed only by training callback, never student input

class LatestEncoder:
    """One in-flight and one newest complete packet. Completed work is not starved.
    Packet ownership transfers on submit: producers must not mutate its tensors.
    Epoch validity is checked by consumer on completion (including partial resets).
    """
    def __init__(self,duration_ms=0):
        self.duration_ms=duration_ms;self.flight=None;self.waiting=None
        self.last_sequence=-1;self.dropped=0;self.rejected=0;self.max_occupancy=0
    def submit(self,packet,now):
        if packet.frames.shape[1:]!=(4,2,45,80):raise ValueError('complete synchronized four-view packet required')
        if packet.arrival_ms>now or packet.sequence<=self.last_sequence:
            self.rejected+=1;return False
        self.last_sequence=packet.sequence
        if self.waiting is not None:self.dropped+=1
        self.waiting=packet
        self.max_occupancy=max(self.max_occupancy,int(self.flight is not None)+1)
        return True
    def poll(self,now,complete):
        if self.flight is not None and self.flight[1]<=now:
            packet,finished=self.flight;self.flight=None
            complete(packet,finished)
        if self.flight is None and self.waiting is not None:
            packet=self.waiting;self.waiting=None
            self.flight=(packet,now+self.duration_ms)
            if self.duration_ms==0:
                self.flight=None;complete(packet,now)
        self.max_occupancy=max(self.max_occupancy,int(self.flight is not None)+int(self.waiting is not None))

class RealtimeBank:
    """Same 51/55 model weights; fractional tick units only at the host boundary."""
    def __init__(self,student,n,device='cpu',projection_cache=False):
        self.bank=make_bank(student,n,device);self.student=student
        # Legacy bank stores ticks in int64. Real-time host retains fractional ticks.
        for key in ('last','delivery','start'):setattr(self.bank,key,getattr(self.bank,key).double())
        self.cache=ProjectionCache(student,n,device) if projection_cache else None
    def reset(self,done,now_ms):
        self.bank.reset(done,now_ms/10)
        if self.cache:self.cache.invalidate(done)
    def complete(self,packet,now_ms):
        b=self.bank
        result=publish(b,packet.frames,packet.pose,packet.observed_capture_ms/10,now_ms/10,packet.epochs,now_ms/10,packet.root_z)
        if self.cache and len(result[0]):self.cache.publish(b.hidden,result[0])
        return result
    def tick(self,proprio,pose,z,now_ms):
        b=self.bank;step=now_ms/10
        if not self.cache:
            kw={'pose_z':z} if isinstance(b,AsyncBeliefBank) else {}
            return b.tick(proprio,pose,step,**kw)
        if step<=b.last_tick:raise ValueError('reversed/duplicate tick')
        if isinstance(b,AsyncBeliefBank) and (not torch.isfinite(pose).all() or not torch.isfinite(z).all()):raise ValueError('nonfinite odometry')
        age,valid=b.inputs(step);valid &= self.cache.valid
        motion=relative_pose(pose,torch.where(b.first[:,None],pose,b.tick_pose))[:,:3]
        dt=proprio.new_full((len(proprio),1),.01 if b.last_tick<0 else (step-b.last_tick)*.01)
        dt=torch.where(b.first[:,None],torch.full_like(dt,.01),dt)
        extra=(b.new_packet,capture_transform(pose,z,b.capture_pose,b.capture_z)) if isinstance(b,AsyncBeliefBank) else ()
        raw,b.belief=projected_tick(self.student,proprio,b.previous_action,dt,age,valid,motion,torch.zeros_like(b.first),self.cache.value,b.belief,*extra)
        if isinstance(b,AsyncBeliefBank):b.new_packet.zero_()
        b.tick_pose=pose.detach().clone();b.first.zero_();b.last_tick=step
        return raw
    def detach(self,weights_changed=False):
        self.bank.detach()
        if self.cache:
            self.cache.detach()
            if weights_changed:
                # Held feature must be reprojected with new weights at TBPTT boundary.
                ids=self.bank.ready.nonzero().flatten();self.cache.publish(self.bank.hidden,ids)

class TrainingHost:
    """Fixed 1 ms physics; independently scheduled captures, completions and ticks.
    actor(raw, gate, observed_proprio) must use only available observations.
    on_tick(record) is the training-only hook for execution-time teacher/loss.
    on_encode(packet, result) owns capture supervision. Model weights may only be
    updated at a detached TBPTT barrier, then call state.detach(weights_changed=True).
    """
    def __init__(self,student,n,timing=Timing(),projection_cache=False,clip_actions=None):
        self.config=timing;self.state=RealtimeBank(student,n,projection_cache=projection_cache)
        self.queue=LatestEncoder(timing.encode_ms);self.rng=random.Random(timing.seed)
        self.odom_rng=random.Random(timing.seed+1)
        self.now_ms=0;self.next_tick=0;self.tick_index=0;self.next_capture=0;self.sequence=0
        self.transit=[];self.actions=deque();self.applied=torch.zeros(n,12)
        self.observations=deque(maxlen=timing.proprio_delay_ms+1)
        self.burst_left=0;self.dropped=0;self.clip_actions=clip_actions;self.last_release=None;self.release_offset=0
    def reset(self,done):
        self.state.reset(done,self.now_ms)
        self.applied=torch.where(done[:,None],torch.zeros_like(self.applied),self.applied)
        # Queued actions carry per-row epochs and cannot cross this barrier.
        self.observations.clear()
    def _interval(self):
        c=self.config
        if c.long_probability and self.rng.random()<c.long_probability:
            self.release_offset=0  # re-anchor nominal schedule after skipped releases
            return c.long_interval_ms
        offset=self.rng.randint(-c.jitter_ms,c.jitter_ms) if c.jitter_ms else 0
        interval=10+offset-self.release_offset
        self.release_offset=offset
        return interval
    def run_until(self,end_ms,*,advance,observe,capture,actor,on_tick=lambda r:None,on_encode=lambda p,r:None,capture_period_ms=80):
        if not isinstance(end_ms,int) or end_ms<self.now_ms or capture_period_ms<1:raise ValueError('monotonic integer milliseconds required')
        c=self.config;b=self.state.bank
        while self.now_ms<end_ms:
            now=self.now_ms
            proprio,pose,z=observe()
            pose=pose.detach().clone();z=z.detach().clone()
            if c.odometry_std_m:
                noise=pose.new_tensor([[self.odom_rng.gauss(0,c.odometry_std_m) for _ in range(2)] for _ in range(len(pose))]);pose[:,:2]+=noise
            self.observations.append((now,proprio.detach().clone()))
            if now==self.next_capture:
                frames,labels=capture();self.sequence+=1
                if self.burst_left==0 and c.burst_probability and self.rng.random()<c.burst_probability:self.burst_left=c.burst_packets
                if self.burst_left:self.burst_left-=1;self.dropped+=1
                else:
                    delay=c.vision_delay_ms
                    if c.long_vision_probability and self.rng.random()<c.long_vision_probability:delay+=c.long_vision_ms
                    observed=now+c.timestamp_offset_ms+now*c.timestamp_drift_ppm/1e6
                    self.transit.append(Packet(self.sequence,now,now+delay,observed,frames.detach().clone(),pose.clone(),z.clone(),b.episode.clone(),labels))
                self.next_capture+=capture_period_ms
            due=[p for p in self.transit if p.arrival_ms<=now]
            self.transit=[p for p in self.transit if p.arrival_ms>now]
            # Finish the current encode before processing arrivals; never cancel it
            # just because a newer packet exists. Submission retains only latest.
            self.queue.poll(now,lambda p,t:on_encode(p,self.state.complete(p,t)))
            for p in sorted(due,key=lambda p:(p.arrival_ms,p.sequence)):self.queue.submit(p,now)
            self.queue.poll(now,lambda p,t:on_encode(p,self.state.complete(p,t)))
            if now==self.next_tick:
                # previous_action is what physics actually used in the preceding
                # interval. A command becoming due now applies AFTER this decision.
                b.previous_action=self.applied.clone()
                sample_ms,observed_proprio=self.observations[0]
                raw=self.state.tick(observed_proprio,pose,z,now)
                gate=b.effective(now/10).detach()
                action=actor(raw,gate,observed_proprio)
                if not torch.isfinite(action).all():raise ValueError('nonfinite action')
                command=action.detach().clone()
                if self.clip_actions is not None:command=command.clamp(-self.clip_actions,self.clip_actions)
                self.actions.append((self.tick_index+c.action_delay_ticks,command,b.episode.clone()))
                while self.actions and self.actions[0][0]<=self.tick_index:
                    _,value,epochs=self.actions.popleft()
                    self.applied=torch.where((epochs==b.episode)[:,None],value,self.applied)
                on_tick(dict(now_ms=now,release_jitter_ms=self.release_offset,dt_ms=10 if self.last_release is None else now-self.last_release,
                    proprio_sample_ms=sample_ms,previous_action=b.previous_action.clone(),applied_action=self.applied.clone(),
                    action=action,raw=raw,gate=gate,age=b.inputs(now/10)[0],epoch=b.episode.clone(),h_belief=b.belief))
                self.last_release=now;self.tick_index+=1;self.next_tick=now+self._interval()
            done=advance(.001,self.applied)
            self.now_ms+=1
            if done is not None and done.any():self.reset(done.bool())

class _CachedBankMixin:
    def __init__(self,student,n,device):
        super().__init__(student,n,device);self.cache=ProjectionCache(student,n,device)
    def publish(self,*args,**kwargs):
        result=super().publish(*args,**kwargs)
        if len(result[0]):self.cache.publish(self.hidden,result[0])
        return result
    def reset(self,done,step):
        super().reset(done,step);self.cache.invalidate(done)
    def tick(self,proprio,pose,step,*,pose_z=None):
        from types import SimpleNamespace
        facade=SimpleNamespace(bank=self,student=self.student,cache=self.cache)
        return RealtimeBank.tick(facade,proprio,pose,pose_z,step*10)
    def detach(self):
        super().detach();self.cache.detach()
        # Runner calls detach after optimizer.step; current projection weights win.
        self.cache.publish(self.hidden,self.ready.nonzero().flatten())

def make_training_bank(student,n,device,cache=False):
    if not cache:return make_bank(student,n,device)
    from gd_lab.gast.v3_state import BeliefBank
    cls=AsyncBeliefBank if isinstance(student,AsyncBeliefStudent) else BeliefBank
    return type('Cached'+cls.__name__,(_CachedBankMixin,cls),{})(student,n,device)

class RunnerEncoder:
    """10ms legacy host adapter; full 1ms timing is provided by TrainingHost."""
    def __init__(self,delay_ticks=0):self.queue=LatestEncoder(delay_ticks);self.original={}
    def receive(self,packets,step):
        import copy
        completed=[]
        def commit(p,finished):
            original=self.original.pop(p.sequence)
            # Keep capture and original arrival, distinguish compute completion.
            item=copy.copy(original);item.arrival_step=original.delivery_step;item.delivery_step=finished
            completed.append(item)
        self.queue.poll(step,commit)
        for item in packets:
            frames,epochs,extra,_,contract=item.payload
            p=Packet(item.capture_step,item.capture_step,item.delivery_step,item.capture_step,frames,extra[3],contract['root_z'],epochs)
            waiting=self.queue.waiting
            if self.queue.submit(p,step):
                if waiting is not None:self.original.pop(waiting.sequence,None)
                self.original[p.sequence]=item
        self.queue.poll(step,commit)
        return completed

class AppliedActions:
    def __init__(self,n,device,delay=0):
        if delay not in (0,1,2):raise ValueError('action delay')
        self.delay=delay;self.pending=deque();self.value=torch.zeros(n,12,device=device);self.index=0
    def apply(self,command,epochs):
        self.pending.append((self.index+self.delay,command,epochs.clone()))
        while self.pending and self.pending[0][0]<=self.index:
            _,value,epoch=self.pending.popleft()
            self.value=torch.where((epoch==epochs)[:,None],value,self.value)
        self.index+=1;return self.value.clone()
    def reset(self,done):self.value=torch.where(done[:,None],torch.zeros_like(self.value),self.value)
