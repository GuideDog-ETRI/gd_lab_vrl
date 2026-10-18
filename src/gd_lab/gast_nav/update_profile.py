"""Opt-in one-update inclusive CUDA-event and CPU wall timings."""
import time
from collections import defaultdict
import torch

class UpdateProfile:
    def __init__(self, algorithm):
        self.alg=algorithm;self.restore=[];self.events=[];self.wall=defaultdict(float)
    def wrap(self,obj,name,label):
        old=getattr(obj,name)
        def timed(*a,**kw):
            start=time.perf_counter();x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record()
            try:return old(*a,**kw)
            finally:y.record();self.events.append((label,x,y));self.wall[label]+=time.perf_counter()-start
        self.restore.append((obj,name,old));setattr(obj,name,timed)
    def __enter__(self):
        from . import map_inputs
        if self.alg.capture_pool:self.wrap(self.alg.capture_pool,'unpack','map_unpack_inclusive')
        self.wrap(map_inputs,'warp_categorical','categorical_warp')
        self.wrap(map_inputs,'_warp_near','near_warp_inclusive')
        p=self.alg.policy
        for name in ('nav_wide_encoder','nav_near_encoder','nav_critic_wide','nav_critic_near'):
            self.wrap(getattr(p,name),'forward','nav_cnn')
        self.wrap(p.terrain_encoder,'forward','gast_terrain_encoder')
        self.wrap(p.cenet,'forward','cenet_forward')
        self.wrap(torch.autograd,'backward','backward')
        return self
    def __exit__(self,*exc):
        for obj,name,old in reversed(self.restore):setattr(obj,name,old)
        torch.cuda.synchronize()
        gpu=defaultdict(float)
        for key,x,y in self.events:gpu[key]+=x.elapsed_time(y)
        self.result={key:{'wall_ms':self.wall[key]*1000,'gpu_ms':gpu[key],
            'calls':sum(k==key for k,_,_ in self.events)} for key in set(self.wall)|{'mirror','cenet_update'}}
