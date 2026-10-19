"""One-rollout inclusive timings and explicit CUDA host-transfer call census."""
import cProfile,json,pstats,time
from collections import defaultdict
import torch
from .profiling import PipelineProfiler

class RolloutProfile(PipelineProfiler):
    def start(self,env,alg):
        self.cpu=cProfile.Profile();self.transfers=defaultdict(lambda:[0,0.])
        from . import obs_terms
        from . import env as env_module
        for obj,name,label in [(env.sim,'step','physics'),(env.nav,'capture','capture_gate'),
            (env.nav.pipeline,'capture','map_capture'),(env.nav,'prepare','prepare_inclusive'),
            (env.nav,'compute_rewards','rewards_inclusive'),(env.nav.oracle,'evaluate','oracle'),
            (env.observation_manager,'compute','observation_assembly'),(alg.policy,'act','policy_actor'),
            (alg.policy,'evaluate','policy_critic'),(alg.capture_pool,'references','packed_store'),
            (env_module,'static_sensor_rays','obstacle_rays'),(env.scene['nav_obstacles'],'_update_buffers_impl','obstacle_raycast'),(obs_terms,'build_near','near_clearance')]:
            self.wrap(obj,name,label)
        # Python-dispatched methods only; C++ internal transfers are not counted.
        for name in ('item','tolist','cpu','numpy','__bool__','__float__','__int__'):
            old=getattr(torch.Tensor,name)
            def counted(t,*a,_old=old,_name=name,**kw):
                if not t.is_cuda:return _old(t,*a,**kw)
                start=time.perf_counter()
                try:return _old(t,*a,**kw)
                finally:self.transfers[_name][0]+=1;self.transfers[_name][1]+=time.perf_counter()-start
            setattr(torch.Tensor,name,counted);self.originals.append((torch.Tensor,name,old))
        self.cpu.enable()
    def finish(self,log,rank):
        self.cpu.disable();self.close()
        self.cpu.dump_stats(str(log/f'rank{rank}_rollout_cpu.prof'))
        with (log/f'rank{rank}_rollout_cpu.txt').open('w') as f:pstats.Stats(self.cpu,stream=f).sort_stats('cumulative').print_stats(100)
        result=self.flush();result['python_cuda_transfers']={k:{'calls':v[0],'wall_ms':v[1]*1000} for k,v in self.transfers.items()}
        (log/f'rank{rank}_rollout_profile.json').write_text(json.dumps(result,indent=2))
