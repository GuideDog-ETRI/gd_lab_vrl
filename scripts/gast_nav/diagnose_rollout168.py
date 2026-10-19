#!/usr/bin/env python3
"""UPDATE168 200-rollout single-process diagnostic; optimizer updates forbidden."""
import argparse,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--envs',type=int,default=16);p.add_argument('--iterations',type=int,default=200);p.add_argument('--horizon',type=int,default=100);p.add_argument('--profile',action='store_true');p.add_argument('--snapshot',action='store_true')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    out=a.output.resolve();assert out.is_relative_to(ROOT/'logs/gast/nav_v1');out.mkdir(parents=True,exist_ok=True)
    a.headless=True;a.device='cuda:0';a.kit_args=' '.join(f'--/app/tokens/{k}={out}/runtime/{k}' for k in ('cache','data','logs'))
    for k in ('cache','data','logs'):(out/'runtime'/k).mkdir(parents=True,exist_ok=True)
    app=None;result=dict(status='started',envs=a.envs,optimizer_steps=0)
    try:
        app=AppLauncher(a).app
        import torch,gymnasium as gym
        from tensordict import TensorDict
        import gd_lab.gast_nav
        from gd_lab.gast_nav.env_cfg import GastNavEnvCfg
        from gd_lab.gast_nav.profiling import PipelineProfiler
        from gd_lab.gast_nav.actor_critic import GastNavActorCritic
        from gd_lab.gast_nav.migration import migrate_checkpoint
        from gd_lab.gast_nav.ppo import GastNavPPO
        from gd_lab.gast_nav.rollout_growth167 import RolloutGrowthProbe,retained_counts
        from gd_lab.gast_nav import pilot_runner
        import yaml,resource,hashlib
        assert os.environ.get('GAST_ROLLOUT_GROWTH_PROBE')=='1'
        assert 1<=a.iterations<=200 and a.horizon==100
        def forbidden(*args,**kwargs):raise RuntimeError('UPDATE168 optimizer step forbidden')
        torch.optim.Adam.step=forbidden
        torch.manual_seed(168)
        from gd_lab.gast.teacher import GastRayIrActorCritic
        profile=PipelineProfiler()
        if a.profile:profile.install()
        if a.snapshot:
            from gd_lab.gast_nav.obs_terms import ObservationPipeline
            original=ObservationPipeline.capture
            def capture(self,**kw):
                if not (out/'capture.pt').exists():torch.save(kw,out/'capture.pt')
                return original(self,**kw)
            ObservationPipeline.capture=capture
        cfg=GastNavEnvCfg(nav_stage='A0');cfg.seed=168;cfg.scene.num_envs=a.envs;cfg.sim.device='cuda:0'
        cfg.scene.terrain.terrain_generator.num_rows=2;cfg.scene.terrain.terrain_generator.num_cols=20;cfg.scene.terrain.max_init_terrain_level=0
        env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
        raw,_=env.reset();obs=TensorDict(raw,batch_size=[a.envs],device='cuda:0')
        policy_args=dict(obs=obs,obs_groups={'policy':['policy'],'critic':['critic']},num_actions=12,
            actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],
            activation='elu',history_length=5,policy_term_dims=[3,3,3,12,12,12,1],velocity_target_slice=(45,48),
            actor_history_steps=4,height_scan_start=110,adaboot_enabled=False)
        fixed=GastRayIrActorCritic(**policy_args).cuda()
        policy=GastNavActorCritic(**policy_args,nav_privileged_dim=obs['nav_privileged'].shape[-1]).cuda()
        ckpt=ROOT/'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
        fixed.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['model_state_dict'],strict=True);fixed.eval();fixed.requires_grad_(False);env.nav.fixed_policy=fixed
        migrate_checkpoint(policy,ckpt,expected_sha256='f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98')
        runtime=yaml.safe_load((ROOT/'configs/gast_nav/runtime.yaml').read_text())
        alg=GastNavPPO(policy,nav_config=runtime['ppo'],device='cuda:0',schedule='fixed')
        alg.map_source=env.nav
        alg.init_storage('rl',a.envs,a.horizon,obs,[12]);alg.capture_pool.clear()
        policy.eval();policy.requires_grad_(False)
        policy.nav_update_old_stats=policy.nav_update_new_stats=False
        def fingerprint():
            digest=hashlib.sha256()
            for k,v in policy.state_dict().items():digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        before_hash=fingerprint()
        probe=RolloutGrowthProbe();probe.install(env,alg,pilot_runner)
        probe.wrap(env.sim,'step','physics')
        probe.wrap(env.scene['nav_obstacles'],'_update_buffers_impl','obstacle_raycast')
        def rss():
            return next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
        result.update(envs=a.envs,horizon=a.horizon,requested_iterations=a.iterations,seed=168,
                      policy='fresh NAV expansion from scaled 3906; frozen stochastic rollout',
                      comparison_limits='no update or distributed collectives; current environment configuration')
        (out/'run_config.json').write_text(json.dumps(result,indent=2))
        with torch.inference_mode():
            for iteration in range(a.iterations):
                probe.flush();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                terminated_n=timeout_n=hold_n=0
                for step in range(a.horizon):
                    action=alg.act_selected(obs,obs['nav_controller'])
                    raw,reward,terminated,truncated,extra=env.step(action)
                    obs=TensorDict(raw,batch_size=[a.envs],device='cuda:0')
                    extra['time_outs']=truncated
                    alg.process_env_step(obs,reward,terminated|truncated,extra)
                    if not bool(torch.isfinite(action).all() & torch.isfinite(reward).all()):raise ValueError('nonfinite')
                    terminated_n+=int(terminated.sum());timeout_n+=int(truncated.sum())
                    hold_n+=int((extra['nav_controller']==1).sum())
                alg.compute_returns(obs)
                torch.cuda.synchronize()
                record=dict(iteration=iteration+1,rollout_s=time.perf_counter()-start,update_s=0.,
                    terminated=terminated_n,timeouts=timeout_n,hold_fraction=hold_n/(a.envs*a.horizon),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),rss_bytes=rss(),
                    retained=retained_counts(env,alg),host=probe.flush(),optimizer_steps=0)
                with (out/'rank0_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                alg.storage.clear();alg.capture_pool.clear();alg.nav_normalization_obs.clear()
                if record['peak_reserved_bytes']>28*2**30:raise RuntimeError('28 GiB reserved diagnostic cap')
        after_hash=fingerprint();assert before_hash==after_hash,'frozen policy changed'
        result.update(status='completed',iterations=a.iterations,policy_sha256_before=before_hash,policy_sha256_after=after_hash)
        probe.close();env.close()
    except Exception as e:result.update(status='failed',error=repr(e));raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2))
        if app is not None:app.close()
if __name__=='__main__':main()
