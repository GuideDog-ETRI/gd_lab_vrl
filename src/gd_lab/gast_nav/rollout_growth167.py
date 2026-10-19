"""Opt-in bounded host timing; no CUDA events, profiler, timer thread or sync.

Inclusive wall times include existing device waits. Never interpret as kernel
exclusive times. Counters are cleared every iteration; close restores wrappers.
"""
import functools
import time

class RolloutGrowthProbe:
    def __init__(self):
        self.rows={};self.originals=[]

    def wrap(self,obj,name,label):
        old=getattr(obj,name)
        @functools.wraps(old)
        def timed(*args,**kwargs):
            before=time.perf_counter()
            try:return old(*args,**kwargs)
            finally:
                row=self.rows.setdefault(label,[0,0.])
                row[0]+=1;row[1]+=time.perf_counter()-before
        self.originals.append((obj,name,old));setattr(obj,name,timed)

    def install(self,env,alg,runner_module):
        for obj,name,label in (
            (env,'step','env_step'),(env.nav,'prepare','nav_prepare'),
            (env.nav,'reset','nav_reset'),(env.nav,'physics_step','nav_physics_hook'),
            (env.nav,'compute_rewards','nav_rewards'),
            (env.nav.pipeline,'capture','map_capture'),
            (env.observation_manager,'compute','observation_assembly'),
            (env.scene,'update','scene_update'),
            (env.nav.fixed_policy,'act_inference','fixed_hold_policy'),
            (alg,'act_selected','policy_selected'),(alg,'process_env_step','storage'),
            (runner_module,'any_rank','rank_guard'),(runner_module,'raise_if_any','rank_error_guard')):
            self.wrap(obj,name,label)

    def flush(self):
        out={k:{'calls':v[0],'wall_s':v[1]} for k,v in self.rows.items()}
        self.rows.clear()
        return out

    def close(self):
        for obj,name,old in reversed(self.originals):setattr(obj,name,old)
        self.originals.clear()


def retained_counts(env,alg):
    """CPU container cardinalities only; no tensor copies or CUDA reads."""
    control=env.nav.control
    return dict(map_frames=len(env.nav.pipeline.mapper.history),
                landing_consumed=sum(len(f.consumed) for f in control.fsm),
                landing_queue=sum(len(f.queue) for f in control.fsm),
                turn_states=sum(len(s.turn_states) for s in control.supervisors),
                packed_records=len(alg.capture_pool.records),
                normalization_snapshots=len(alg.nav_normalization_obs))
