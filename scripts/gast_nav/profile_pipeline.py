#!/usr/bin/env python3
"""UPDATE162 local GPU0 inference scaling; no optimizer steps or training."""
import argparse,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--envs',type=int,default=16);p.add_argument('--steps',type=int,default=100);p.add_argument('--profile',action='store_true');p.add_argument('--snapshot',action='store_true')
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
        cfg=GastNavEnvCfg(nav_stage='A0');cfg.scene.num_envs=a.envs;cfg.sim.device='cuda:0'
        cfg.scene.terrain.terrain_generator.num_rows=2;cfg.scene.terrain.terrain_generator.num_cols=20;cfg.scene.terrain.max_init_terrain_level=0
        env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
        raw,_=env.reset();obs=TensorDict(raw,batch_size=[a.envs],device='cuda:0')
        fixed=GastRayIrActorCritic(obs=obs,obs_groups={'policy':['policy'],'critic':['critic']},num_actions=12,
            actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],
            activation='elu',history_length=5,policy_term_dims=[3,3,3,12,12,12,1],velocity_target_slice=(45,48),
            actor_history_steps=4,height_scan_start=110,adaboot_enabled=False).cuda()
        ckpt=ROOT/'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
        fixed.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['model_state_dict'],strict=True);fixed.eval();fixed.requires_grad_(False);env.nav.fixed_policy=fixed
        if a.profile:
            profile.wrap(env.sim,'step','physics.step')
            profile.wrap(fixed,'act_inference','policy.act_inference')
        result['initialization_profile']=profile.flush();samples=[]
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode():
            for step in range(a.steps):
                torch.cuda.synchronize();before=time.perf_counter()
                action=fixed.act_inference(obs);raw,_,_,_,_=env.step(action);obs=TensorDict(raw,batch_size=[a.envs],device='cuda:0')
                torch.cuda.synchronize();samples.append(time.perf_counter()-before)
                if step%10==0:print(json.dumps(dict(step=step,wall_s=samples[-1])),flush=True)
                if time.perf_counter()-start>180:break
        result.update(status='completed' if len(samples)==a.steps else 'bounded_stop',steps=len(samples),step_seconds=samples,wall_s=sum(samples),profile=profile.flush(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        profile.close();env.close()
    except Exception as e:result.update(status='failed',error=repr(e));raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2))
        if app is not None:app.close()
if __name__=='__main__':main()
