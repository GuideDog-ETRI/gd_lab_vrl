"""UPDATE162 bounded training loop; existing collective/frozen-state checks reused."""
import json
import os
import signal
import time
from pathlib import Path
from collections import Counter,defaultdict
import torch
import torch.distributed as dist
from .runner_support import TopK
from .trainable import assert_frozen
from gd_lab.rl.distributed_sync import (any_rank,raise_if_any,assert_synchronized_state,install_collective_optimizer_step)
from gd_lab.rl.terrain_resume import capture_family_level_means,restore_family_start_levels

class PhaseTimers:
    def __init__(self):self.wall=defaultdict(float);self.events=[]
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        def call(*args,**kw):
            start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            try:return original(*args,**kw)
            finally:
                b.record();self.wall[label]+=time.perf_counter()-start;self.events.append((label,a,b))
        setattr(obj,name,call)
    def flush(self):
        torch.cuda.synchronize()
        gpu=defaultdict(float)
        for key,a,b in self.events:gpu[key]+=a.elapsed_time(b)
        out={**{k+'_wall_ms':v*1000 for k,v in self.wall.items()},**{k+'_gpu_ms':v for k,v in gpu.items()}}
        self.wall.clear();self.events.clear();return out


def validate_pilot(args,settings,world):
    if not args.pilot:return
    if os.environ.get('GAST_NAV_PILOT')!='1':raise ValueError('GAST_NAV_PILOT=1 required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2':raise ValueError('pilot visible GPUs must be exactly 0,1,2')
    if not args.distributed or world!=settings['world_size']:raise ValueError('pilot requires exactly three ranks')
    if args.stage!='A0':raise ValueError('pilot is A0 only')
    expected=settings['smoke_envs_per_rank']*world if args.pilot_smoke else settings['total_envs']
    if args.total_envs!=expected:raise ValueError('pilot env count outside authorization')
    limit=settings['smoke_iterations'] if args.pilot_smoke else settings['max_iterations']
    if not 1<=args.max_iterations<=limit:raise ValueError('pilot iteration count outside authorization')


def run_training(*,env,policy,obs,args,root,migration,pilot_settings):
    from tensordict import TensorDict
    from .ppo import GastNavPPO
    import yaml
    rank=int(os.environ.get('RANK','0'));world=dist.get_world_size() if dist.is_initialized() else 1
    cfg=env.cfg.nav_settings;train=cfg['training'];phase=env.cfg.nav_stage_settings
    runtime=yaml.safe_load((root/'configs/gast_nav/runtime.yaml').read_text())
    policy.train()
    if world>1:
        for value in policy.state_dict().values():dist.broadcast(value,src=0)
    alg=GastNavPPO(policy,nav_config=runtime['ppo'],trainable_manifest=root/phase['trainable'],device=args.device,
        schedule='fixed',learning_rate=train['learning_rate'],num_learning_epochs=train['epochs'],num_mini_batches=train['minibatches'])
    install_collective_optimizer_step(alg.optimizer,device=args.device,context='Nav PPO optimizer')
    optimizers=[('ppo',alg.optimizer),('cenet',policy.cenet.optimizer)]
    log=root/train['log_root']/args.run_id;log.mkdir(parents=True,exist_ok=True)
    grid=env.scene.terrain.cfg.terrain_generator.nav_scene_grid
    families={k:sorted({c for (_,c),m in grid.items() if m.kind==k}) for k in cfg['scene_mix']}
    start_iteration=0
    top=TopK(log/'best_top5',train['top_k'])
    if args.resume:
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved['stage']!=args.stage:raise ValueError('resume stage differs from checkpoint')
        policy.load_state_dict(saved['model_state_dict'],strict=True)
        # Reapply manifest after restored state so immutable checksums match resume.
        alg.configure_trainable(root/phase['trainable'])
        alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        policy.cenet.optimizer.load_state_dict(saved['cenet_optimizer_state_dict'])
        install_collective_optimizer_step(alg.optimizer,device=args.device,context='resumed Nav PPO optimizer')
        optimizers=[('ppo',alg.optimizer),('cenet',policy.cenet.optimizer)]
        start_iteration=saved['iteration']
        restore_family_start_levels(env.scene.terrain,env.scene.env_origins,families,saved.get('family_levels',{}))
        raw,_=env.reset();obs=TensorDict(raw,batch_size=[env.num_envs],device=args.device)
        top.records=[(s,i,Path(p)) for s,i,p in saved.get('top5',[]) if Path(p).is_file() and Path(p).parent.resolve()==top.directory.resolve()]
    optimizer_steps=0
    original_step=alg.optimizer.step
    def counted_step(*a,**kw):
        nonlocal optimizer_steps
        result=original_step(*a,**kw);optimizer_steps+=1;return result
    alg.optimizer.step=counted_step
    assert_synchronized_state(policy,optimizers,context='startup/resume')
    horizon=pilot_settings['smoke_horizon'] if args.pilot_smoke else train['horizon']
    alg.init_storage('rl',env.num_envs,horizon,obs,[12])
    timing=PhaseTimers();timing.wrap(env.nav,'capture','map');timing.wrap(env.sim,'step','physics')
    stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    previous=signal.signal(signal.SIGUSR1,request_stop)
    slow_count=0
    metrics_path=log/f'rank{rank}_metrics.jsonl'
    def checkpoint(iteration,name):
        if rank!=0:return
        state={'model_state_dict':policy.state_dict(),'optimizer_state_dict':alg.optimizer.state_dict(),
               'cenet_optimizer_state_dict':policy.cenet.optimizer.state_dict(),'iteration':iteration,'stage':args.stage,
               'migration':migration,'family_levels':capture_family_level_means(env.scene.terrain,families),
               'top5':[(s,i,str(p)) for s,i,p in top.records],
               'resume_contract':'weights/optimizers/normalizers + family curriculum; reset episodes, not bitwise trajectory resume'}
        path=log/name;temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)
    iteration=start_iteration;tick=-1
    try:
        for iteration in range(start_iteration,args.max_iterations):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(args.device)
            begin=time.perf_counter();reward_sum=0.;controllers=Counter();rewards=defaultdict(float);valid_rows=0
            terminations=0;timeouts=0;person0=env.nav.person_contacts.summary()['person_contact']
            with torch.no_grad():
                for tick in range(horizon):
                    error=None
                    try:
                        action=alg.act_selected(obs,obs['nav_controller'])
                        raw,reward,terminated,truncated,extras=env.step(action)
                        obs=TensorDict(raw,batch_size=[env.num_envs],device=args.device)
                        if not bool(torch.isfinite(action).all()&torch.isfinite(reward).all()) or not all(bool(torch.isfinite(v).all()) for v in obs.values()):
                            raise FloatingPointError('nonfinite rollout')
                        reward_sum+=float(reward.sum());terminations+=int(terminated.sum());timeouts+=int(truncated.sum())
                        counts=torch.bincount(extras['nav_controller'],minlength=3).tolist()
                        for k,v in enumerate(counts):controllers[k]+=v
                        valid_rows+=counts[0]
                        for k,v in extras['nav_rewards'].items():rewards[k]+=v
                        extras['time_outs']=truncated
                        alg.process_env_step(obs,reward,terminated|truncated,extras)
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                    raise_if_any(error,device=args.device,context=f'rollout {iteration}:{tick}')
                    allocated=torch.cuda.memory_reserved(args.device)
                    capacity=torch.cuda.get_device_properties(args.device).total_memory
                    if any_rank(allocated>capacity*pilot_settings['vram_limit_fraction'],args.device):
                        raise RuntimeError('VRAM high-water limit exceeded during rollout')
                    if any_rank(time.perf_counter()-begin>pilot_settings['iteration_hard_timeout_s'],args.device):
                        checkpoint(iteration,'stopped_timeout.pt');raise RuntimeError('iteration hard timeout during rollout')
                    if any_rank(stop,args.device):
                        checkpoint(iteration,'stopped.pt');return {'status':'stopped','iteration':iteration,'partial_rollout_discarded':True,'optimizer_steps':optimizer_steps}
                alg.compute_returns(obs)
            torch.cuda.synchronize();rollout_s=time.perf_counter()-begin
            update_start=time.perf_counter();metrics=alg.update();torch.cuda.synchronize();update_s=time.perf_counter()-update_start
            check_error=None
            try:assert_frozen(policy,alg.nav_frozen)
            except Exception as exc:check_error=str(exc)
            raise_if_any(check_error,device=args.device,context='frozen checksum')
            fingerprint=assert_synchronized_state(policy,optimizers,context=f'iteration {iteration+1}')
            elapsed=time.perf_counter()-begin
            max_elapsed=torch.tensor(elapsed,device=args.device)
            if world>1:dist.all_reduce(max_elapsed,op=dist.ReduceOp.MAX)
            slow_count=slow_count+1 if float(max_elapsed)>pilot_settings['iteration_limit_s'] else 0
            peak=torch.cuda.max_memory_reserved(args.device);capacity=torch.cuda.get_device_properties(args.device).total_memory
            total=env.num_envs*horizon
            record=dict(iteration=iteration+1,rank=rank,num_envs=env.num_envs,horizon=horizon,
                rollout_s=rollout_s,update_s=update_s,iteration_s=elapsed,step_per_s=total/elapsed,
                optimizer_steps=optimizer_steps,peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),peak_reserved_bytes=peak,
                reward_mean=reward_sum/total,actor_valid_rows=valid_rows,
                controller_fractions={str(k):controllers[k]/total for k in range(3)},rewards={k:v/horizon for k,v in rewards.items()},
                cylinder_skipped=env.nav.cylinder.skip_report(),person_contact=env.nav.person_contacts.summary(),
                person_contact_new=env.nav.person_contacts.summary()['person_contact']-person0,
                terminated=terminations,timeouts=timeouts,nan=False,frozen_checksums_ok=True,ddp_fingerprint=fingerprint,
                **timing.flush(),**metrics)
            with metrics_path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps({'rank':rank,'iteration':iteration+1,'seconds':elapsed,'reward':record['reward_mean'],'actor_rows':valid_rows}),flush=True)
            if any_rank(peak>capacity*pilot_settings['vram_limit_fraction'],args.device):raise RuntimeError('VRAM high-water limit exceeded')
            if slow_count>=pilot_settings['slow_iterations']:
                checkpoint(iteration+1,'stopped_slow.pt')
                raise RuntimeError(f'iteration > {pilot_settings["iteration_limit_s"]} s for {slow_count} consecutive iterations')
            score=torch.tensor([reward_sum,total],device=args.device,dtype=torch.float64)
            if world>1:dist.all_reduce(score)
            if (iteration+1)%args.save_interval==0 or iteration+1==args.max_iterations:
                checkpoint(iteration+1,'latest.pt')
                if rank==0:
                    top.consider(float(score[0]/score[1]),iteration+1,lambda path:checkpoint(iteration+1,str(path.relative_to(log))))
            if any_rank(stop,args.device):checkpoint(iteration+1,'stopped.pt');return {'status':'stopped','iteration':iteration+1,'optimizer_steps':optimizer_steps}
        return {'status':'completed','iteration':args.max_iterations,'optimizer_steps':optimizer_steps}
    except Exception as error:
        # Preserve partial-run diagnostics before app teardown or a late signal.
        try:partial_timing=timing.flush()
        except Exception as timing_error:partial_timing={'timing_error':repr(timing_error)}
        partial=dict(status='failed',iteration=iteration,rollout_tick=tick,
            optimizer_steps=optimizer_steps,error=repr(error),**partial_timing)
        (log/f'rank{rank}_partial.json').write_text(json.dumps(partial,indent=2)+'\n')
        raise
    finally:
        # A second stop request during Isaac close must not restore SIG_DFL
        # and replace the recorded timeout with an unrelated signal exit.
        signal.signal(signal.SIGUSR1,signal.SIG_IGN if previous==signal.SIG_DFL else previous)
