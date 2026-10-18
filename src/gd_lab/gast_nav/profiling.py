"""Inference-only UPDATE162 profiling hooks; inclusive wall/CUDA timings."""
import functools,time
from collections import defaultdict
import torch

class PipelineProfiler:
    def __init__(self):self.rows=defaultdict(list);self.originals=[]
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            before=time.perf_counter();start.record()
            result=original(*args,**kwargs)
            end.record();self.rows[label].append((1000*(time.perf_counter()-before),start,end))
            return result
        setattr(obj,name,measured);self.originals.append((obj,name,original))
    def install(self):
        from . import obstacle_map,near_clearance,support,obs_terms,map_inputs,env,ray_geometry
        from .batched_oracle import BatchedOracle
        self.wrap(BatchedOracle,"evaluate","oracle.evaluate")
        for module,name in [(obstacle_map,'ray_evidence'),(near_clearance,'ray_evidence'),(near_clearance,'clearance_from_state'),
            (obs_terms,'fit_observed_surface'),(obs_terms,'grow_support'),(obs_terms,'observed_support'),(obs_terms,'build_near'),
            (support,'_grow'),(map_inputs,'warp_categorical'),(map_inputs,'_warp_near'),(obs_terms,'network_inputs'),
            (env,'turn_feasible'),(env,'passage_feasible'),(env,'route_hint'),(env,'project_route'),
            (obs_terms,'capsule_intersections'),(obstacle_map,'resolve_first_hits'),(near_clearance,'resolve_first_hits')]:
            self.wrap(module,name,module.__name__.rsplit('.',1)[-1]+'.'+name)
        for cls,name in [(obs_terms.ObservationPipeline,'capture'),(obs_terms.ObservationPipeline,'inputs'),
            (obstacle_map.ObstacleMap,'update'),(env.NavState,'capture'),(env.NavState,'prepare'),(env.NavState,'compute_rewards')]:
            self.wrap(cls,name,cls.__name__+'.'+name)
    def flush(self):
        torch.cuda.synchronize();result={}
        for key,rows in self.rows.items():
            result[key]=dict(calls=len(rows),wall_ms=sum(x[0] for x in rows),gpu_ms=sum(a.elapsed_time(b) for _,a,b in rows))
        self.rows.clear();return result
    def close(self):
        for obj,name,value in reversed(self.originals):setattr(obj,name,value)
