#!/usr/bin/env python3
"""UPDATE179 D-5 easy-C integration diagnostic, frozen 3906 only."""
import argparse,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--memory-ttl',type=float,choices=[.5,1.,2.],default=.5);p.add_argument('--legacy-check',action='store_true');p.add_argument('--envelope174',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--envs',type=int,default=64);p.add_argument('--steps',type=int,default=600);p.add_argument('--seed',type=int,default=169);p.add_argument('--profile',action='store_true');p.add_argument('--snapshot',action='store_true');p.add_argument('--dense',action='store_true');p.add_argument('--depth-scale',type=int,default=1);p.add_argument('--rule',choices=['off','r1','r2','both'],default='off');p.add_argument('--cert-ttl',type=float,default=.5);p.add_argument('--easy-c',action='store_true');p.add_argument('--pending167',action='store_true');p.add_argument('--envelope167',action='store_true');p.add_argument('--measured167',action='store_true')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert a.envs==64 and a.steps in (2,100,200,300,600), 'UPDATE174 requires 64 env; 2-tick smoke cannot pass gate'
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
        quick_rows=[];quick_rewards={};metrics=HoldReasonMetrics();windows={};reward_sums={};post3_rewards={};eligible179=[];excluded179={}
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
            cfg.nav_settings['free_evidence_backend']='coverage174'
            cfg.nav_settings['supervisor176']=yaml.safe_load((ROOT/'configs/gast_nav/supervisor177.yaml').read_text())
            cfg.nav_settings['sensors']['lower_depth']['enabled']=False
            result['free_evidence_backend']='coverage174'
            result['producer_status']='coverage174 + temporal169; no D4xx, no aperture, no mesh'
            cfg.nav_settings['temporal169_config']='configs/gast_nav/temporal169_training_c.yaml'
            result['seed']=a.seed
            result['volume_valid_producer']='coverage174 scene-only first-hit; no D4xx support'
        # UPDATE167 successful-subset 2D envelope is not a qualified 3D future trajectory.
        cfg.nav_settings.pop('swept_envelope_file',None)
        if a.envelope174:cfg.nav_settings['diagnostic_envelope174']=str(a.envelope174.resolve())
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
        if a.legacy_check:
            from dataclasses import replace
            import gd_lab.gast_nav.fixture174 as fixtures
            original_fixture=fixtures.fixture174
            def legacy_fixture(kind,settings_path,seed=174):
                f=original_fixture('nav_corridor',settings_path,seed)
                return replace(f,meta=replace(f.meta,kind='legacy'))
            fixtures.fixture174=legacy_fixture
        from gd_lab.gast_nav.fixture174 import configure_fixture174
        configure_fixture174(cfg.scene.terrain,a.envs,scene_file,seed=a.seed)
        cfg.scene.terrain.terrain_generator.coverage174_mixed=True
        from isaaclab.sensors import ContactSensorCfg
        import math
        cfg.scene.lazy_sensor_update=False
        for foot in ('FL_foot','FR_foot','RL_foot','RR_foot'):
            setattr(cfg.scene,'ground_contact174_'+foot,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+foot,filter_prim_paths_expr=[f'/World/ground/solids/box_{i}' for i in range(cfg.scene.terrain.terrain_generator.expected_contact_boxes174)],track_contact_points=True,history_length=math.ceil(.05/cfg.sim.dt)+1,max_contact_data_count_per_prim=32,update_period=cfg.sim.dt))
        for name in list(vars(cfg.curriculum)):
            if not name.startswith('_'):setattr(cfg.curriculum,name,None)
        # Primitive-only flat C has no legacy gap/stair rewards or metadata.
        # Retain the requested C nav reward, not fabricated legacy geometry.
        for name in list(vars(cfg.rewards)):
            if not name.startswith('_') and name!='nav_total':setattr(cfg.rewards,name,None)
        cfg.nav_stage_settings['moving_cylinder']=False
        cfg.scene.nav_cylinder.spawn.collision_props.collision_enabled=False
        cfg.nav_settings['primitive_c169']=True
        assert not cfg.nav_settings['observability']['r1_static_history'] and not cfg.nav_settings['observability']['r2_unknown_components']
        (out/'resolved_config_pre177.json').write_text(json.dumps(cfg.nav_settings,indent=2))
        from gd_lab.gast_nav.e1_terrain177 import E1Terrain177
        from isaaclab.managers import ObservationTermCfg
        cfg.observations.terrain.camera_visible=ObservationTermCfg(func=E1Terrain177)
        cfg.nav_settings['terrain_fusion177']=True
        cfg.nav_settings['pipeline179']=True
        cfg.nav_settings['pipeline180']=True
        cfg.nav_settings['current_body_margin_m']=yaml.safe_load((ROOT/'configs/gast_nav/pipeline179.yaml').read_text())['current_body_margin_m']
        cfg.nav_settings['front_depth183']=yaml.safe_load((ROOT/'configs/gast_nav/front_depth183.yaml').read_text())
        cfg.nav_settings['front_depth183'].update(enabled=True,obstacle_ttl_s=a.memory_ttl,support_ttl_s=a.memory_ttl)
        cfg.nav_settings['supervisor176'].update(obstacle_ttl_s=a.memory_ttl,support_ttl_s=a.memory_ttl)
        env=gym.make('Gd-GastNavV1-Rbq10-Dreamwaq-v0',cfg=cfg).unwrapped
        (out/'camera183_profile.json').write_text(json.dumps(env.nav.pipeline.certified_producer.camera.manifest,indent=2))
        (out/'oracle177_profile.json').write_text(json.dumps(env._oracle177_profile,indent=2))
        (out/'resolved_config.json').write_text(json.dumps(cfg.nav_settings,indent=2))
        env._contact174_force_gate=True
        result['live_inventory_audit']=env.nav.pipeline.certified_producer.audit169
        (out/'sensor_profile176.json').write_text(json.dumps(env.nav.pipeline.certified_producer.native.manifest,indent=2))
        (out/'live_inventory_audit.json').write_text(json.dumps(result['live_inventory_audit'],indent=2))
        assert not bool(env.nav.cylinder.enabled.any()), 'static C only'
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
                if step>=min(300,a.steps//2):
                    for row in extra['nav_decision_rows']:
                        reason='LEGACY' if row['requested_mode']=='LEGACY' else 'GO_ZERO' if not row['requested_go'] else 'FSM' if row['phase'] in ('PRE_BRAKE','LANDING_ENTERED','DWELL') else None
                        if reason:excluded179[reason]=excluded179.get(reason,0)+1
                        else:eligible179.append(row)
                if 100<=step<200:
                    quick_rows.extend(row for row in extra['nav_decision_rows'] if row['requested_mode']!='LEGACY' and row['requested_go'] and row['phase'] not in ('PRE_BRAKE','LANDING_ENTERED','DWELL'))
                for k,v in extra['nav_rewards'].items():
                    if step<200:quick_rewards[k]=quick_rewards.get(k,0.)+v
                    reward_sums[k]=reward_sums.get(k,0.)+v
                    if step>=300:post3_rewards[k]=post3_rewards.get(k,0.)+v
                if a.pending167 and step==599:
                    torch.save(dict(actual=env.nav.diagnostic_rays[0],unoccluded=env.nav.diagnostic_rays[1],support=env.nav.pipeline.near_support_height,required=env.nav.required_sweep,rows=env.nav.decision_rows),out/'rays167.pt')
                if step==10:torch.save(env.nav.pipeline.certified_producer.support_debug174,out/'support_debug174.pt')
                if step in (0,199,299,599):
                    producer=env.nav.pipeline.certified_producer
                    torch.save(dict(**producer.last_query169,required_near=env.nav.required_sweep,
                        required_wide=env.nav.required_wide,envelope_available=producer.swept_certified,
                        future_sweep_status=producer.swept_status,future_sweep=getattr(env.nav,'future_sweep174',None),
                        swept_diagnostics=producer.swept_diagnostics,
                        route_features179=env.nav.route_features.clone(),body_names=env.nav.robot.body_names,body_pos_w=env.nav.robot.data.body_pos_w,
                        body_quat_w=env.nav.robot.data.body_quat_w,
                        contact_points={foot:env.scene['ground_contact174_'+foot].data.contact_pos_w for foot in ('FL_foot','FR_foot','RL_foot','RR_foot')},
                        producer_stats=dict(producer.stats),expected_wide=env.nav.pipeline.cert_wide_state,
                        expected_near=env.nav.pipeline.cert_near_state),out/f'temporal_probe_{step}.pt')
                if step in (100,199,299,400,599):
                    torch.save(dict(near_state=env.nav.pipeline.near_state,near_support=env.nav.pipeline.near_support_known,
                        wide_state=env.nav.pipeline.mapper.history[0].state,wide_support=env.nav.pipeline.mapper.history[0].support,
                        required_sweep=env.nav.required_sweep,body_near=env.nav.body_near,body_wide=env.nav.body_wide,rows=env.nav.decision_rows,margins=[c.c.margin for c in env.nav.control.commands.commands],
                        near=env.nav.near,wide=env.nav.wide),out/f'evidence_{step}.pt')
                torch.cuda.synchronize();samples.append(time.perf_counter()-before)
                if env.nav.pipeline.certified_producer is not None:
                    with (out/'producer_metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**env.nav.pipeline.certified_producer.stats))+'\n')
                if step==199:
                    qm=HoldReasonMetrics();qm.add(quick_rows)
                    quick=dict(steps=200,eligible_ticks=len(quick_rows),hold_fraction=sum(row['controller']=='HOLD_3906' for row in quick_rows)/max(1,len(quick_rows)),observed_ticks=sum(row['observed'] for row in quick_rows),groups=qm.snapshot(),reward_means={k:v/200 for k,v in quick_rewards.items()})
                    (out/'quick200.json').write_text(json.dumps(quick,indent=2))
                if step%10==0:print(json.dumps(dict(step=step,wall_s=samples[-1])),flush=True)
                if time.perf_counter()-start>3600:break
        selected179=HoldReasonMetrics();selected179.add(eligible179)
        if (out/'quick200.json').exists():result['quick200']=json.loads((out/'quick200.json').read_text())
        result['entry183']=env.nav.pipeline.certified_producer.entry_stats
        result['entry183_history']=env.nav.pipeline.certified_producer.entry_history
        (out/'camera183_profile.json').write_text(json.dumps(env.nav.pipeline.certified_producer.camera.manifest,indent=2))
        result['sweep_calibration180']=json.loads((ROOT/'logs/gast/nav_v1/update180/sweep_calibration.json').read_text())
        result['d5_denominator']=dict(eligible_ticks=len(eligible179),excluded=excluded179,groups=selected179.snapshot(),nonpolicy_fraction=sum(r['controller']!='NAV_POLICY' for r in eligible179)/max(1,len(eligible179)),hold_fraction=sum(r['controller']=='HOLD_3906' for r in eligible179)/max(1,len(eligible179)))
        result['windows']={k:v.snapshot() for k,v in windows.items()};result['reason_groups']=metrics.snapshot();result['reward_means']={k:v/len(samples) for k,v in reward_sums.items()}
        result.update(status='completed' if len(samples)==a.steps else 'bounded_stop',steps=len(samples),step_seconds=samples,wall_s=sum(samples),profile=profile.flush(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        gate_rows=sum([result['windows'].get(str(i),[]) for i in (3,4,5)],[])
        total=sum(r['ticks'] for r in gate_rows);nav=[r for r in gate_rows if r['kind']!='legacy']
        result['gate']=dict(hold_fraction=sum(r['hold'] for r in gate_rows)/max(total,1),
            observed=sum(r['sums']['observed'] for r in nav),nav_ticks=sum(r['ticks'] for r in nav),
            progress=post3_rewards.get('progress',0)/max(len(samples)-300,1),ready=False)
        result['post3_reward_means']={k:v/max(len(samples)-300,1) for k,v in post3_rewards.items()}
        import xml.etree.ElementTree as ET
        junit=ROOT/'logs/gast/nav_v1/update176_producer.xml'
        cases=list(ET.parse(junit).getroot().iter('testcase')) if junit.exists() else []
        native=json.loads((ROOT/'logs/gast/nav_v1/update176_native_audit.json').read_text())
        verified=len(cases)==16 and native['misses']==0 and not any(c.find(k) is not None for c in cases for k in ('failure','error','skipped'))
        result['native_pattern_audit']=dict(trials=native['trials'],misses=native['misses'])
        result['gate']['counterexamples_1_6_passed']=verified
        result['swept_status']=env.nav.pipeline.certified_producer.swept_status
        result['gate']['future_sweep_connected']=getattr(env.nav,'future_sweep174',None) is not None
        result['gate']['legacy_live_checked']=a.legacy_check
        result['legacy_check']=a.legacy_check
        result['gate']['per_kind']={}
        for kind in sorted({r['kind'] for r in gate_rows}):
            rows=[r for r in gate_rows if r['kind']==kind];ticks=sum(r['ticks'] for r in rows)
            result['gate']['per_kind'][kind]=dict(ticks=ticks,hold_fraction=sum(r['hold'] for r in rows)/max(ticks,1),observed=sum(r['sums']['observed'] for r in rows))
        terrain_tests=ROOT/'logs/gast/nav_v1/update177/terrain_tests.xml'
        tc=list(ET.parse(terrain_tests).getroot().iter('testcase')) if terrain_tests.exists() else []
        verified=verified and len(tc)==21 and not any(c.find(k) is not None for c in tc for k in ('failure','error','skipped'))
        result['gate']['terrain_counterexamples_passed']=verified
        new_tests=ROOT/'logs/gast/nav_v1/update177b/terrain177b_tests.xml'
        nc=list(ET.parse(new_tests).getroot().iter('testcase')) if new_tests.exists() else []
        verified=verified and len(nc)==21 and not any(c.find(k) is not None for c in nc for k in ('failure','error','skipped'))
        result['gate']['terrain177b_counterexamples_passed']=verified
        current_cases=list(ET.parse(ROOT/'logs/gast/nav_v1/update183/cpu.xml').getroot().iter('testcase'))
        verified=verified and any('test_terrain_free180' in c.get('classname','') for c in current_cases) and not any(c.find(k) is not None for c in current_cases for k in ('failure','error'))
        result['gate']['update180_tests_passed']=verified
        result['gate']['ready']=(result['status']=='completed' and total==19200 and
            result['d5_denominator']['eligible_ticks']>0 and result['d5_denominator']['hold_fraction']<=.30 and result['gate']['progress']>0 and verified and result['gate']['future_sweep_connected'])

        profile.close();env.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        result.update(status='failed',error=repr(e));raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2))
        if app is not None:app.close()
if __name__=='__main__':main()
