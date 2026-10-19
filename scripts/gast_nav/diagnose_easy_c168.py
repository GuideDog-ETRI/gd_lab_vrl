#!/usr/bin/env python3
"""UPDATE168 reproducible easy-C integration diagnostic, frozen 3906 only."""
import argparse,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--envs',type=int,default=64);p.add_argument('--steps',type=int,default=600);p.add_argument('--seed',type=int,default=168);p.add_argument('--profile',action='store_true');p.add_argument('--snapshot',action='store_true');p.add_argument('--dense',action='store_true');p.add_argument('--depth-scale',type=int,default=1);p.add_argument('--rule',choices=['off','r1','r2','both'],default='off');p.add_argument('--cert-ttl',type=float,default=2.);p.add_argument('--easy-c',action='store_true');p.add_argument('--pending167',action='store_true');p.add_argument('--envelope167',action='store_true');p.add_argument('--measured167',action='store_true')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert a.envs==64 and a.steps==600, 'UPDATE168 C gate requires 64 env, 600 ticks'
    a.easy_c=True;a.pending167=True;a.envelope167=True;a.measured167=True
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
        from gd_lab.gast_nav.supervisor import HoldReasonMetrics
        metrics=HoldReasonMetrics();windows={};reward_sums={}
        profile=PipelineProfiler()
        if a.profile:profile.install()
        if a.snapshot:
            from gd_lab.gast_nav.obs_terms import ObservationPipeline
            original=ObservationPipeline.capture
            capture_count=[0]
            def capture(self,**kw):
                if capture_count[0] in (0,10,30,40,60):torch.save(kw,out/f'capture_{capture_count[0]}.pt')
                capture_count[0]+=1
                return original(self,**kw)
            ObservationPipeline.capture=capture
        os.environ['GAST_NAV_HOLD_METRICS']='1'
        cfg=GastNavEnvCfg(nav_stage='A0');cfg.seed=a.seed;torch.manual_seed(a.seed);cfg.scene.num_envs=a.envs;cfg.sim.device='cuda:0'
        cfg.scene.terrain.terrain_generator.num_rows=2;cfg.scene.terrain.terrain_generator.num_cols=20;cfg.scene.terrain.max_init_terrain_level=0
        if a.easy_c:
            import yaml
            from gd_lab.gast_nav.scene_config import load_scene_config
            from gd_lab.gast_nav.curriculum167 import easy_scene_settings
            limits=yaml.safe_load((ROOT/'configs/gast_nav/curriculum_C.yaml').read_text())['scene_limits']
            scene_file=out/'easy_scenes.yaml';scene_file.write_text(yaml.safe_dump(easy_scene_settings(load_scene_config(),limits)))
            for kind,terrain in cfg.scene.terrain.terrain_generator.sub_terrains.items():
                if kind.startswith('nav_'):terrain.settings_path=str(scene_file)
            cfg.nav_settings['curriculum_C_limits']=limits
            cfg.scene.terrain.terrain_generator.difficulty_range=(0.,0.)
            result['curriculum_C_limits']=limits
        if a.pending167:
            import yaml
            c=yaml.safe_load((ROOT/'configs/gast_nav/curriculum_C.yaml').read_text())
            cfg.nav_settings['rewards'].update(c['reward_overrides'])
            cfg.nav_settings['free_evidence_backend']='pending167'
            cfg.nav_settings['diagnostics167']=True
            cfg.nav_settings['sensors']['lower_depth']['diagnostic_unoccluded']=True
            result['free_evidence_backend']='pending167'
        if a.measured167:
            assert a.pending167, 'use the same C rewards/diagnostic fixture'
            cfg.nav_settings['free_evidence_backend']='producer168'
            result['free_evidence_backend']='producer168'
            result['producer_status']='producer168 live primitive current-capture adapter; C-only aperture Claude-approved'
            result['seed']=a.seed
            result['volume_valid_producer']='continuous block first-hit producer168; no historical reuse'
        if a.envelope167:
            cfg.nav_settings['swept_envelope_file']=str(ROOT/'logs/gast/nav_v1/update167_envelope/envelopes.pt')
        if a.dense:
            from gd_lab.gast_nav.obs_terms import isaac_sensor_cfg
            cfg.nav_settings['sensors'].update(azimuth_count=48,elevation_count=32)
            cfg.scene.nav_obstacles=isaac_sensor_cfg(cfg.nav_settings['sensors'],prim_path='{ENV_REGEX_NS}/Robot/trunk',mesh_prim_path='/World/ground',update_period=cfg.nav_settings['capture_period_s'])
        cfg.nav_settings['observability']['r1_static_history']=a.rule in ('r1','both')
        cfg.nav_settings['observability']['r2_unknown_components']=a.rule in ('r2','both')
        cfg.nav_settings['observability']['certification_ttl_s']=a.cert_ttl
        result['certification_settings']=dict(cfg.nav_settings['observability'])
        cfg.nav_settings['sensors']['lower_depth']['sampling_scale']=a.depth_scale
        result['sensor_settings']=cfg.nav_settings['sensors'];result['simulation_turn_cap']=cfg.nav_settings['commands']['simulation_turn_cap_radps']
        (out/'resolved_config.json').write_text(json.dumps(cfg.nav_settings,indent=2))
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
                action=fixed.act_inference(obs);raw,_,_,_,extra=env.step(action);metrics.add(extra['nav_decision_rows']);windows.setdefault(str(step//100),HoldReasonMetrics()).add(extra['nav_decision_rows']);obs=TensorDict(raw,batch_size=[a.envs],device='cuda:0')
                for k,v in extra['nav_rewards'].items():reward_sums[k]=reward_sums.get(k,0.)+v
                if a.pending167 and step==599:
                    torch.save(dict(actual=env.nav.diagnostic_rays[0],unoccluded=env.nav.diagnostic_rays[1],support=env.nav.pipeline.near_support_height,required=env.nav.required_sweep,rows=env.nav.decision_rows),out/'rays167.pt')
                if step in (100,299,400,599):
                    torch.save(dict(near_state=env.nav.pipeline.near_state,near_support=env.nav.pipeline.near_support_known,
                        wide_state=env.nav.pipeline.mapper.history[0].state,wide_support=env.nav.pipeline.mapper.history[0].support,
                        required_sweep=env.nav.required_sweep,body_near=env.nav.body_near,body_wide=env.nav.body_wide,rows=env.nav.decision_rows,margins=[c.c.margin for c in env.nav.control.commands.commands],
                        near=env.nav.near,wide=env.nav.wide),out/f'evidence_{step}.pt')
                torch.cuda.synchronize();samples.append(time.perf_counter()-before)
                if step%10==0:
                    with (out/'producer_metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**env.nav.pipeline.certified_producer.stats))+'\n')
                if step%10==0:print(json.dumps(dict(step=step,wall_s=samples[-1])),flush=True)
                if time.perf_counter()-start>3600:break
        result['windows']={k:v.snapshot() for k,v in windows.items()};result['reason_groups']=metrics.snapshot();result['reward_means']={k:v/len(samples) for k,v in reward_sums.items()}
        result.update(status='completed' if len(samples)==a.steps else 'bounded_stop',steps=len(samples),step_seconds=samples,wall_s=sum(samples),profile=profile.flush(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        gate_rows=sum([result['windows'].get(str(i),[]) for i in (3,4,5)],[])
        total=sum(r['ticks'] for r in gate_rows);nav=[r for r in gate_rows if r['kind']!='legacy']
        result['gate']=dict(hold_fraction=sum(r['hold'] for r in gate_rows)/max(total,1),
            observed=sum(r['sums']['observed'] for r in nav),nav_ticks=sum(r['ticks'] for r in nav),
            progress=result['reward_means'].get('progress',0),ready=False)
        import xml.etree.ElementTree as ET
        junit=ROOT/'logs/gast/nav_v1/update168_producer/cpu_final.xml'
        cases=list(ET.parse(junit).getroot().iter('testcase')) if junit.exists() else []
        verified=all(any(c.get('name','').startswith(f'test_counterexample{i}_') and not any(c.find(k) is not None for k in ('failure','error','skipped')) for c in cases) for i in range(1,7))
        verified &= bool(cases) and not any(c.find(k) is not None for c in cases for k in ('failure','error'))
        result['gate']['counterexamples_1_6_passed']=verified
        result['gate']['ready']=(result['status']=='completed' and total==19200 and
            result['gate']['hold_fraction']<=.30 and result['gate']['progress']>0 and verified)

        profile.close();env.close()
    except Exception as e:result.update(status='failed',error=repr(e));raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2))
        if app is not None:app.close()
if __name__=='__main__':main()
