#!/usr/bin/env python3
"""GAST-Nav launcher: UPDATE162 bounded A0 pilot, explicit future training, inference smoke."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

def main():
    import faulthandler, signal, time
    diagnostic_id=os.environ.get('GAST_RUN_ID','diagnostic')
    if not __import__('re').fullmatch(r'[a-zA-Z0-9_-]+',diagnostic_id):raise ValueError('run id')
    diagnostic_dir=ROOT/'logs/gast/nav_v1'/diagnostic_id
    diagnostic_dir.mkdir(parents=True,exist_ok=True)
    stack_file=(diagnostic_dir/('rank'+os.environ.get('RANK','0')+'_stacks.txt')).open('ab',buffering=0)
    faulthandler.enable(file=stack_file,all_threads=True)
    # CPython's background traceback walker can race frame/code destruction.
    # Keep native fault/SIGUSR2 handlers; periodic dumps are opt-in diagnostics.
    if os.environ.get('GAST_NAV_PERIODIC_STACKS')=='1':
        faulthandler.dump_traceback_later(120,repeat=True,file=stack_file)
    faulthandler.register(signal.SIGUSR2,file=stack_file,all_threads=True)
    def stamp(event):
        row={'event':event,'rank':os.environ.get('RANK','0'),'time':time.time()}
        with (diagnostic_dir/('rank'+os.environ.get('RANK','0')+'_startup.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    stamp('python_start')
    if os.environ.get('GAST_NCCL_PREFLIGHT_ONLY') == '1':
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local)
        stamp('pure_init_before');dist.init_process_group('nccl',timeout=timedelta(seconds=120));stamp('pure_init_after')
        value=torch.tensor([local+1.],device=f'cuda:{local}')
        stamp('pure_collective_before');dist.all_reduce(value);torch.cuda.synchronize();stamp('pure_collective_after')
        assert value.item()==world*(world+1)/2
        dist.destroy_process_group()
        return
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true');mode.add_argument('--train',action='store_true');mode.add_argument('--pilot',action='store_true')
    parser.add_argument('--stage',choices=['A0','A','B','C'],default=None)
    parser.add_argument('--distributed',action='store_true');parser.add_argument('--total_envs',type=int,default=4096)
    parser.add_argument('--pilot-smoke',action='store_true')
    parser.add_argument('--max_iterations',type=int)
    parser.add_argument('--save_interval',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--easy-c184',action='store_true')
    parser.add_argument('--run_id',default=os.environ.get('GAST_RUN_ID'))
    parser.add_argument('--output',type=Path,default=ROOT/'tests/gast_nav/update160b_smoke')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    import yaml,re
    pilot_settings=yaml.safe_load((ROOT/'configs/gast_nav/pilot.yaml').read_text())
    args.stage=args.stage or ('A0' if args.pilot else 'B')
    if args.pilot_smoke and not args.pilot:parser.error('--pilot-smoke requires --pilot')
    if args.pilot_smoke:args.total_envs=pilot_settings['smoke_envs_per_rank']*int(os.environ.get('WORLD_SIZE',pilot_settings['world_size']))
    args.max_iterations=args.max_iterations or (pilot_settings['smoke_iterations'] if args.pilot_smoke else pilot_settings['max_iterations'] if args.pilot else yaml.safe_load((ROOT/f'configs/gast_nav/stage_{args.stage}.yaml').read_text())['max_iterations'])
    args.save_interval=args.save_interval or pilot_settings['save_interval']
    if args.run_id is None:args.run_id='nav_'+__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.run_id):parser.error('invalid run id')
    if args.smoke and args.distributed:parser.error('smoke is one GPU only')
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if args.train and os.environ.get('GAST_NAV_TRAINING_AUTHORIZED')!='1':
        parser.error('training disabled; a separate explicit authorization is required')
    from gd_lab.gast_nav.pilot_runner import validate_pilot
    validate_pilot(args,pilot_settings,world)
    args.device=f'cuda:{local}';args.headless=True
    if args.smoke:
        args.output=args.output.resolve()
        if not args.output.is_relative_to(ROOT/'tests/gast_nav'):parser.error('smoke writes must stay in tests/gast_nav')
        args.output.mkdir(parents=True,exist_ok=True)
        # Do not rerun after any simulator launch without renewed permission.
        with (args.output/'GPU_ATTEMPT_STARTED').open('x') as f:f.write('one authorized UPDATE160B attempt\n')
    runtime=args.output/'runtime' if args.smoke else ROOT/'logs/gast/nav_v1'/args.run_id/'runtime'/str(rank)
    for key in ('cache','data','logs'): (runtime/key).mkdir(parents=True,exist_ok=True)
    args.kit_args=(args.kit_args or '')+' '+ ' '.join(f'--/app/tokens/{key}={runtime/key}' for key in ('cache','data','logs'))
    result={'status':'started','optimizer_steps':0 if args.smoke else None,'rollout_steps':0,'stage':args.stage}
    app=None
    try:
        import torch
        import torch.distributed as dist
        if args.distributed and not args.smoke:
            from datetime import timedelta
            stamp('init_process_group_before')
            torch.cuda.set_device(local)
            dist.init_process_group('nccl',timeout=timedelta(seconds=pilot_settings['collective_timeout_s']))
            stamp('init_process_group_after')
            stamp('first_collective_before')
            probe=torch.tensor([rank+1.],device=args.device)
            dist.all_reduce(probe);torch.cuda.synchronize()
            assert probe.item()==world*(world+1)/2
            stamp('first_collective_after')
        stamp('app_before');launcher=AppLauncher(args);app=launcher.app;stamp('app_after')
        # Kit may install native signal handlers during startup. Restore our Python trace.
        faulthandler.enable(file=stack_file,all_threads=True)
        import torch
        import torch.distributed as dist
        import gymnasium as gym
        from tensordict import TensorDict
        import gd_lab.gast_nav
        from gd_lab.gast_nav.env_cfg import GastNavEnvCfg
        from gd_lab.gast_nav.actor_critic import GastNavActorCritic
        from gd_lab.gast.teacher import GastRayIrActorCritic
        from gd_lab.gast_nav.migration import migrate_checkpoint
        from gd_lab.gast_nav.trainable import apply_manifest
        from gd_lab.gast_nav.runner_support import rank_env_count,TopK
        import yaml
        if args.smoke:
            def forbidden(*a,**kw):raise RuntimeError('optimizer step forbidden during smoke')
            pending=[torch.optim.Optimizer];seen=set()
            while pending:
                cls=pending.pop()
                if cls in seen:continue
                seen.add(cls);pending.extend(cls.__subclasses__());cls.step=forbidden
        cfg=GastNavEnvCfg(nav_stage=args.stage)
        cfg.scene.num_envs=16 if args.smoke else rank_env_count(args.total_envs,world,rank)
        cfg.sim.device=args.device;cfg.nav_smoke=args.smoke
        if args.smoke or args.pilot_smoke:
            cfg.scene.terrain.terrain_generator.num_rows=2
            cfg.scene.terrain.terrain_generator.num_cols=20
            cfg.scene.terrain.max_init_terrain_level=0
        if args.easy_c184:
            if not args.train or world!=1 or args.total_envs!=256 or args.max_iterations>20:
                raise ValueError('UPDATE184 local smoke scope')
            from gd_lab.gast_nav.fixture184 import configure184
            cfg=configure184(cfg,diagnostic_dir,envs=256)
            # The full-pixel static oracle is diagnostic, not the old 120s pilot pipeline.
            pilot_settings=dict(pilot_settings,iteration_hard_timeout_s=1800.,iteration_limit_s=1800.)
        stamp('env_before')
        env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
        stamp('env_after')
        if args.easy_c184:
            env._contact174_force_gate=True
            (diagnostic_dir/'resolved_config.json').write_text(json.dumps(cfg.nav_settings,indent=2))
        if args.smoke:
            terrain=env.scene.terrain;grid=terrain.cfg.terrain_generator.nav_scene_grid
            # Deterministic scene coverage without replacing actual mesh or metadata.
            chosen=[next((r,c) for (r,c),m in grid.items() if m.kind==kind) for kind in cfg.nav_settings['scene_mix']]
            for i in range(env.num_envs):
                r,c=chosen[i%len(chosen)];terrain.terrain_levels[i]=r;terrain.terrain_types[i]=c
            terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        raw,_=env.reset();obs=TensorDict(raw,batch_size=[env.num_envs],device=args.device)
        policy_args=dict(obs=obs,obs_groups={'policy':['policy'],'critic':['critic']},num_actions=12,
            actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],
            activation='elu',history_length=5,policy_term_dims=[3,3,3,12,12,12,1],velocity_target_slice=(45,48),
            actor_history_steps=4,height_scan_start=110,adaboot_enabled=False)
        fixed=GastRayIrActorCritic(**policy_args).to(args.device)
        policy=GastNavActorCritic(**policy_args,nav_privileged_dim=obs['nav_privileged'].shape[-1]).to(args.device)
        checkpoint=ROOT/'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
        sha='f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98'
        stamp('migration_before')
        migration=migrate_checkpoint(policy,checkpoint,expected_sha256=sha)
        stamp('migration_after')
        fixed.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=False)['model_state_dict'],strict=True)
        fixed.eval();fixed.requires_grad_(False);env.nav.fixed_policy=fixed
        phase=cfg.nav_stage_settings
        _,manifest=apply_manifest(policy,ROOT/phase['trainable'],learning_rate=cfg.nav_settings['training']['learning_rate'])
        result['migration']=migration;result['manifest_stage']=manifest['stage']
        if args.smoke:
            policy.eval();kinds=sorted({m.kind for m in env.nav.index.lookup(range(env.num_envs))})
            assert all(k in kinds for k in cfg.nav_settings['scene_mix'])
            result['scene_kinds']=kinds;result['observation_shapes']={k:list(v.shape) for k,v in obs.items()}
            controllers=set();force_error=0.
            with torch.inference_mode():
                for step in range(200):
                    env.nav.force_fault=step in cfg.nav_settings['smoke']['force_fault_steps']
                    if step==180:
                        # Same single app, final 20 ticks exercise C cylinders.
                        env.nav.stage=copy_stage=dict(env.nav.stage,moving_cylinder=True)
                        raw,_=env.reset();obs=TensorDict(raw,batch_size=[env.num_envs],device=args.device)
                    action=policy.act_inference(obs)
                    raw,reward,terminated,truncated,extras=env.step(action)
                    obs=TensorDict(raw,batch_size=[env.num_envs],device=args.device)
                    assert all(bool(torch.isfinite(v).all()) for v in obs.values())
                    assert bool(torch.isfinite(reward).all())
                    controllers.update(extras['nav_controller'].tolist())
                    mass=env.nav.robot.root_physx_view.get_masses().to(args.device)
                    # Active reset may clear buffers after last physics write; check nonreset rows.
                    keep=~(terminated|truncated)
                    expected=-mass[...,None]*env.nav.forces.last_acceleration[:,None]+env.nav.forces.last_handle
                    if keep.any():force_error=max(force_error,float((env.nav.forces.last_force[keep]-expected[keep]).abs().max()))
                    result['rollout_steps']=step+1
                    if step%20==0:print(json.dumps({'step':step,'controllers':sorted(controllers)}),flush=True)
            result.update(status='passed',controllers=sorted(controllers),force_sum_max_error=force_error,
                force_write_count=env.nav.forces.write_count,rewards=extras['nav_rewards'],cylinder_skips=env.nav.cylinder.skip_report(),
                cylinder_enabled=int(env.nav.cylinder.enabled.sum()),cylinder_C_steps=20)
            assert 0 in controllers and 1 in controllers
            assert force_error<1e-4
        elif os.environ.get('GAST_NAV_REPLAY_OUT'):
            # RL replay recording (tools/rl_replay); no training, no checkpoint writes.
            from gd_lab.gast_nav.replay_nav import record_nav_replay
            result.update(record_nav_replay(env=env,policy=policy,obs=obs,args=args,root=ROOT))
        else:
            from gd_lab.gast_nav.pilot_runner import run_training
            # All ranks share the run's one literal 'core' path (core_uses_pid=0).
            import resource
            resource.setrlimit(resource.RLIMIT_CORE,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
            os.chdir(diagnostic_dir)
            stamp('core_limit:'+str(resource.getrlimit(resource.RLIMIT_CORE)))
            faulthandler.enable(file=stack_file,all_threads=True)
            faulthandler.register(signal.SIGUSR2,file=stack_file,all_threads=True)
            stamp('runner_before')
            result.update(run_training(env=env,policy=policy,obs=obs,args=args,root=ROOT,
                                       migration=migration,pilot_settings=pilot_settings))
        env.close()
    except Exception as error:
        result['status']='failed';result['error']=repr(error)
        raise
    finally:
        if args.smoke:
            (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        else:
            destination=ROOT/'logs/gast/nav_v1'/args.run_id
            destination.mkdir(parents=True,exist_ok=True)
            (destination/f'rank{rank}_result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        if app is not None:app.close()

if __name__=='__main__':main()
