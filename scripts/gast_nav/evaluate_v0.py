#!/usr/bin/env python3
"""Fixed 3906 paired V0. Inference only; local GPU0. No optimizer exists."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('local GPU0 only')
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'logs/gast/nav_v1'):raise ValueError('output scope')
    out.mkdir(parents=True,exist_ok=True)
    args.headless=True;args.device='cuda:0'
    args.kit_args=' '.join(f'--/app/tokens/{k}={out}/runtime/{k}' for k in ('cache','data','logs'))
    for k in ('cache','data','logs'):(out/'runtime'/k).mkdir(parents=True,exist_ok=True)
    result={'status':'started','optimizer_steps':0};app=None;started=time.monotonic()
    try:
        app=AppLauncher(args).app
        import torch,yaml,numpy as np
        from tensordict import TensorDict
        from isaaclab.envs import ManagerBasedRLEnv
        from gd_lab.gast_nav.v0_eval import make_cfg,bindings
        from gd_lab.gast_nav.scene_types import project_route
        from gd_lab.gast_nav.scene_geometry import point_box_distance
        from gd_lab.gast.teacher import GastRayIrActorCritic
        def forbidden(*a,**k):raise RuntimeError('V0 optimizer forbidden')
        pending=[torch.optim.Optimizer];seen=set()
        while pending:
            cls=pending.pop()
            if cls in seen:continue
            seen.add(cls);pending.extend(cls.__subclasses__());cls.step=forbidden
        settings=yaml.safe_load((ROOT/'configs/gast_nav/v0.yaml').read_text());(out/'settings.json').write_text(json.dumps(settings,indent=2))
        cfg=make_cfg(settings);cfg.sim.device='cuda:0';env=ManagerBasedRLEnv(cfg=cfg)
        # Make the mapping explicit and verify one env per planned terrain column.
        t=env.scene.terrain;t.terrain_levels[:]=0;t.terrain_types[:]=torch.arange(env.num_envs,device=env.device)
        t.env_origins[:]=t.terrain_origins[t.terrain_levels,t.terrain_types]
        raw,_=env.reset();obs=TensorDict(raw,batch_size=[env.num_envs],device=env.device)
        fixed=GastRayIrActorCritic(obs=obs,obs_groups={'policy':['policy'],'critic':['critic']},num_actions=12,
            actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],
            activation='elu',history_length=5,policy_term_dims=[3,3,3,12,12,12,1],velocity_target_slice=(45,48),
            actor_history_steps=4,height_scan_start=110,adaboot_enabled=False).to(env.device)
        checkpoint=ROOT/'checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt'
        source_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert source_sha=='f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98'
        fixed.load_state_dict(torch.load(checkpoint,map_location=env.device,weights_only=False)['model_state_dict'],strict=True)
        fixed.eval();fixed.requires_grad_(False)
        def fingerprint():
            sha=hashlib.sha256()
            for name,value in fixed.state_dict().items():sha.update(name.encode());sha.update(value.detach().cpu().numpy().tobytes())
            return sha.hexdigest()
        before=fingerprint();bound=bindings(env);n=env.num_envs;robot=env.scene['robot']
        assert len({(c['kind'],c['seed'],c['structures'],c['mode']) for c,_ in bound})==n
        max_s=np.zeros(n);falls=np.zeros(n,dtype=bool);stalls=falls.copy();complete=falls.copy()
        clearance=np.full(n,np.inf);fall_s=np.full(n,np.nan);trajectory=[];samples=[];max_lateral=np.zeros(n)
        cadence=max(1,round(settings['sample_period_s']/env.step_dt));history=[]
        steps=round(settings['duration_s']/env.step_dt)
        with torch.inference_mode():
            for step in range(steps):
                if step%cadence==0:
                    xy=(robot.data.root_pos_w-env.scene.env_origins)[:,:2].cpu().numpy()
                    tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))).cpu().numpy()
                    projection=np.array([project_route(meta,p) for (_,meta),p in zip(bound,xy)])
                    s=projection[:,0];lateral=projection[:,1]
                    active=~(falls|complete);max_lateral=np.where(active,np.maximum(max_lateral,np.abs(lateral)),max_lateral);max_s=np.where(active,np.maximum(max_s,s),max_s)
                    newly=active & (tilt>settings['fall_tilt_deg']);fall_s[newly]=max_s[newly];falls|=newly
                    complete |= active & ~newly & (np.abs(lateral)<=settings['completion_lateral_tolerance_m']) & np.array([v>=m.route_length-settings['completion_margin_m'] for v,(_,m) in zip(s,bound)])
                    history.append(s.copy());window=round(settings['stall_window_s']/settings['sample_period_s'])
                    if len(history)>window and step*env.step_dt>=settings['stall_grace_s']+settings['stall_window_s']:
                        stalls |= active & ~newly & ~complete & (s-history[-window-1]<settings['stall_progress_m'])
                    body_z=(robot.data.body_pos_w[:,:,2]-env.scene.env_origins[:,2,None]).cpu().numpy()
                    for i,(_,meta) in enumerate(bound):
                        if active[i]:
                            values=[point_box_distance(xy[i],b)-settings['clearance_body_radius_proxy_m'] for b in meta.obstacles if b.kind!='step' and b.center[2]-b.half[2]<=body_z[i].max()+settings['clearance_vertical_padding_m'] and b.center[2]+b.half[2]>=body_z[i].min()-settings['clearance_vertical_padding_m']]
                            if values:clearance[i]=min(clearance[i],min(values))
                    term=env._gast_terrain_term
                    gt=obs.clone();gt['terrain']=term.gt_held[:,:374];gt['gast_history']=term.gt_packed
                    visible=obs.clone();visible['terrain']=term.visible_held[:,:374];visible['gast_history']=term.visible_packed
                    latent_delta=(fixed.terrain_latent(gt)-fixed.terrain_latent(visible)).norm(dim=-1)
                    action_delta=(fixed.act_inference(gt)-fixed.act_inference(visible)).norm(dim=-1)
                    shared=(term.gt_held[:,187:374]>0)&(term.visible_held[:,187:374]>0)
                    for i in np.flatnonzero(active):
                        samples.append(dict(step=step,env=int(i),gt_valid=float(term.gt_held[i,187:374].mean()),visible_valid=float(term.visible_held[i,187:374].mean()),
                            h_abs_delta=float((term.gt_held[i,:187]-term.visible_held[i,:187]).abs().mean()),
                            h_shared_max=float((term.gt_held[i,:187]-term.visible_held[i,:187]).abs().masked_fill(~shared[i],0).max()),
                            ir_abs_delta=float((term.gt_held[i,374:]-term.visible_held[i,374:]).abs().mean()),
                            latent_l2=float(latent_delta[i]),action_l2=float(action_delta[i])))
                    trajectory.append(dict(step=step,progress=max_s.tolist(),falls=falls.astype(int).tolist()))
                action=fixed.act_inference(obs)
                if not torch.isfinite(action).all():raise FloatingPointError('V0 nonfinite action')
                raw,_,terminated,truncated,_=env.step(action)
                if terminated.any() or truncated.any():raise RuntimeError('unexpected reset before metrics; evaluator invalid')
                obs=TensorDict(raw,batch_size=[n],device=env.device)
                if step%100==0:print(json.dumps(dict(step=step,steps=steps,falls=int(falls.sum()),completed=int(complete.sum()))),flush=True)
        after=fingerprint();assert before==after
        rows=[]
        for i,(case,meta) in enumerate(bound):
            rows.append(dict(env=i,**case,progress_m=float(max_s[i]),route_length_m=meta.route_length,fall=bool(falls[i]),stall=bool(stalls[i]),
                max_abs_lateral_m=float(max_lateral[i]),complete=bool(complete[i]),stair_success=bool(complete[i]) if case['kind']=='nav_stair_landing' else None,
                fall_max_route_s_m=None if np.isnan(fall_s[i]) else float(fall_s[i]),
                min_root_disc_clearance_proxy_m=None if not np.isfinite(clearance[i]) else float(clearance[i])))
        comparisons=[]
        for kind in sorted({r['kind'] for r in rows}):
            for mode in ('gt','visible'):
                groups=[[r for r in rows if r['kind']==kind and r['mode']==mode and r['structures']==b] for b in (False,True)]
                fall_delta=100*(np.mean([r['fall'] for r in groups[1]])-np.mean([r['fall'] for r in groups[0]]))
                stall_delta=100*(np.mean([r['stall'] for r in groups[1]])-np.mean([r['stall'] for r in groups[0]]))
                comparisons.append(dict(kind=kind,mode=mode,n_pairs=len(groups[0]),fall_increase_pp=fall_delta,stall_increase_pp=stall_delta,
                    problem=bool(fall_delta>=settings['fall_increase_pp'] or stall_delta>=settings['stall_increase_pp'])))
        result.update(status='completed',source_sha256=source_sha,frozen_sha_before=before,frozen_sha_after=after,
            episodes=rows,comparisons=comparisons,wall_s=time.monotonic()-started,
            limitations=['8 seeds per kind/mode; preliminary thresholds, not confidence bounds',
                'IR is controlled geometric proxy; GT uses counterfactual IR for hidden cells; no rendered images or v25or',
                'clearance is root disc radius .25m against obstacles overlapping body-origin z bounds plus .05m, not swept collision clearance',
                'nominal dynamics, no external force/payload/randomization; LEGACY route-following velocity only'])
        (out/'same_pose.json').write_text(json.dumps(samples));(out/'trajectory.json').write_text(json.dumps(trajectory))
        env.close()
    except Exception as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if app is not None:app.close()

if __name__=='__main__':main()
