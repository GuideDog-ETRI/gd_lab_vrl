#!/usr/bin/env python3
"""GAST-Nav launcher. UPDATE160B execution is --smoke only, never --train."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true');mode.add_argument('--train',action='store_true')
    parser.add_argument('--stage',choices=['A0','A','B','C'],default='B')
    parser.add_argument('--distributed',action='store_true');parser.add_argument('--total_envs',type=int,default=4096)
    parser.add_argument('--output',type=Path,default=ROOT/'tests/gast_nav/update160b_smoke')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.smoke and args.distributed:parser.error('smoke is one GPU only')
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if args.train and os.environ.get('GAST_NAV_TRAINING_AUTHORIZED')!='1':
        parser.error('training disabled; a separate explicit authorization is required')
    args.device=f'cuda:{local}';args.headless=True
    if args.smoke:
        args.output=args.output.resolve()
        if not args.output.is_relative_to(ROOT/'tests/gast_nav'):parser.error('smoke writes must stay in tests/gast_nav')
        args.output.mkdir(parents=True,exist_ok=True)
        # Do not rerun after any simulator launch without renewed permission.
        with (args.output/'GPU_ATTEMPT_STARTED').open('x') as f:f.write('one authorized UPDATE160B attempt\n')
    runtime=args.output/'runtime' if args.smoke else ROOT/'logs/gast/nav_v1'/args.stage/'runtime'/str(rank)
    for key in ('cache','data','logs'): (runtime/key).mkdir(parents=True,exist_ok=True)
    args.kit_args=(args.kit_args or '')+' '+ ' '.join(f'--/app/tokens/{key}={runtime/key}' for key in ('cache','data','logs'))
    result={'status':'started','optimizer_steps':0,'rollout_steps':0,'stage':args.stage}
    app=None
    try:
        launcher=AppLauncher(args);app=launcher.app
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
        elif args.distributed:
            torch.cuda.set_device(local);dist.init_process_group('nccl')
        cfg=GastNavEnvCfg(nav_stage=args.stage)
        cfg.scene.num_envs=16 if args.smoke else rank_env_count(args.total_envs,world,rank)
        cfg.sim.device=args.device;cfg.nav_smoke=args.smoke
        if args.smoke:
            cfg.scene.terrain.terrain_generator.num_rows=2
            cfg.scene.terrain.terrain_generator.num_cols=20
            cfg.scene.terrain.max_init_terrain_level=0
        env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
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
        migration=migrate_checkpoint(policy,checkpoint,expected_sha256=sha)
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
        else:
            from gd_lab.gast_nav.ppo import GastNavPPO
            runtime_cfg=yaml.safe_load((ROOT/'configs/gast_nav/runtime.yaml').read_text())
            train=cfg.nav_settings['training'];policy.train()
            if world>1:
                for value in policy.state_dict().values():dist.broadcast(value,src=0)
            alg=GastNavPPO(policy,nav_config=runtime_cfg['ppo'],trainable_manifest=ROOT/phase['trainable'],device=args.device,
                schedule='fixed',learning_rate=train['learning_rate'],num_learning_epochs=train['epochs'],num_mini_batches=train['minibatches'])
            alg.init_storage('rl',env.num_envs,train['horizon'],obs,[12])
            log=ROOT/train['log_root']/args.stage;top=TopK(log/'best_top5',train['top_k'])
            for iteration in range(phase['max_iterations']):
                score=torch.zeros(2,device=args.device)
                with torch.no_grad():
                    for _ in range(train['horizon']):
                        action=alg.act_selected(obs,obs['nav_controller'])
                        raw,reward,terminated,truncated,extras=env.step(action)
                        obs=TensorDict(raw,batch_size=[env.num_envs],device=args.device)
                        extras['time_outs']=truncated
                        alg.process_env_step(obs,reward,terminated|truncated,extras)
                        score+=torch.stack((reward.sum(),reward.new_tensor(reward.numel())))
                    alg.compute_returns(obs)
                metrics=alg.update()
                if world>1:dist.all_reduce(score)
                if rank==0:
                    log.mkdir(parents=True,exist_ok=True)
                    with (log/'metrics.jsonl').open('a') as f:f.write(json.dumps({'iteration':iteration,'mean_reward':float(score[0]/score[1]),**metrics})+'\n')
                    if (iteration+1)%train['checkpoint_interval']==0:
                        def save(path):torch.save({'model_state_dict':policy.state_dict(),'optimizer_state_dict':alg.optimizer.state_dict(),
                                                  'iteration':iteration+1,'migration':migration},path)
                        top.consider(float(score[0]/score[1]),iteration+1,save)
        env.close()
    except Exception as error:
        result['status']='failed';result['error']=repr(error)
        raise
    finally:
        if args.smoke:
            (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        if app is not None:app.close()

if __name__=='__main__':main()
