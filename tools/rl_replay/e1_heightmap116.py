"""Frozen UPDATE116 rollout. Sensor map never receives clean terrain labels."""
import hashlib,json,os,time,types
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from e1_runner import load_teacher,configure as original_configure,Recorder
from e1_core import LatentSwitch
from runtime import fix_velocity,environment_metadata
from heightmap116 import ElevationMap
from sensor_independent116 import corrupt

FAMILIES={'gap':'platform_gap','pyramid_stairs':'pyramid_stairs','stairs_inv':'pyramid_stairs_inv','nose':'pyramid_stairs_nose','wide':'pyramid_stairs_wide'}
def configure(cfg):
    original_configure(cfg)
    family=FAMILIES[os.environ['U116_FAMILY']]
    if family!='platform_gap':
        gen=cfg.scene.terrain.terrain_generator
        assert family in gen.sub_terrains
        gen.num_cols=20+len(gen.sub_terrains)-1
        for name,sub in gen.sub_terrains.items():sub.proportion=(20 if name==family else 1)/gen.num_cols

class StairRecorder:
    """Center-to-outer-border traversal, arbitrary spawn yaw; pre-reset events."""
    def __init__(self,base,family):
        self.base=base;self.family=family;self.rows=[[] for _ in range(base.num_envs)];self.records=[];self.episodes=[0]*base.num_envs
        self.original=base._reset_idx
        self.trunk=base.scene['contact_forces'].body_names.index('trunk')
        sub=base.cfg.scene.terrain.terrain_generator.sub_terrains[FAMILIES[os.environ['U116_FAMILY']]]
        self.inner=float(sub.platform_width)/2;self.outer=min(base.cfg.scene.terrain.terrain_generator.size)/2-float(sub.border_width)
        def reset(env,ids,*a,**kw):
            self.sample(ids,True)
            return self.original(ids,*a,**kw)
        base._reset_idx=types.MethodType(reset,base)
    def sample(self,ids=None,terminal=False):
        b=self.base;d=b.scene['robot'].data;ids=list(range(b.num_envs)) if ids is None else ids.tolist()
        local=d.root_pos_w-b.scene.env_origins;radius=local[:,:2].abs().amax(-1)
        w,x,y,z=d.root_quat_w.unbind(-1);upright=(2*(w*y-z*x)).abs()<np.sin(.5)
        upright&=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)).abs()<.5
        force=b.scene['contact_forces'].data.net_forces_w[:,self.trunk].norm(dim=-1)
        packed=torch.stack((radius,upright.float(),(force>15).float()),-1).cpu().numpy();enabled=self.family().cpu().numpy()
        timeouts=b.termination_manager.time_outs.cpu().numpy()
        for e in ids:
            if not enabled[e]:continue
            self.rows[e].append(packed[e])
            if terminal:self.finish_one(e,'timeout' if timeouts[e] else 'terminated')
    def finish_one(self,e,reason):
        rows=np.asarray(self.rows[e]);self.rows[e]=[]
        if not len(rows):return
        reached=bool((rows[:,0]>=self.inner).any());passed=bool(((rows[:,0]>=self.outer+.2)&(rows[:,1]>0)).any())
        self.records.append(dict(env_id=e,episode_id=self.episodes[e],reached=reached,passed=passed,termination=reason,body_contact=bool(rows[:,2].any()),max_radius_m=float(rows[:,0].max()),censored=reason=='window_end' and not passed))
        self.episodes[e]+=1
    def finish(self):
        for e in range(len(self.rows)):self.finish_one(e,'window_end')
        return self.records
    def close(self):self.base._reset_idx=self.original

class Diagnostics:
    def __init__(self):self.sums={};self.counts={};self.cpu={};self.corruption={};self.boundary_missing=0
    def add(self,key,values):
        values=np.asarray(values);values=values[np.isfinite(values)]
        self.sums[key]=self.sums.get(key,0.)+float(values.sum());self.counts[key]=self.counts.get(key,0)+int(values.size)
    def cells(self,pred,gt):
        h,v=pred[:,:187],pred[:,187:]>0;gh,gv,gap=gt[:,:187],gt[:,187:374]>0,gt[:,374:561]>0
        pg=v&(h>=4.999);both=v&gv
        self.add('height_mae_clipped_m',np.abs(h-gh)[both]/5)
        self.add('false_valid_unknown_gt',v[~gv]);self.add('false_valid_height_gt_10cm',((np.abs(h-gh)/5)>.1)[both]);self.add('false_gap',pg[gv&~gap]);self.add('gap_recall',pg[gap&gv])
        self.add('coverage',v);self.add('underbody_coverage',v.reshape(-1,11,17)[:,3:8,4:13])
        for a,b in zip(pg.reshape(-1,11,17),gap.reshape(-1,11,17)):
            ea=a&~ndimage.binary_erosion(a);eb=b&~ndimage.binary_erosion(b)
            if not ea.any() or not eb.any():
                if ea.any() or eb.any():self.boundary_missing+=1
                continue
            self.add('gap_boundary_symmetric_m',np.r_[ndimage.distance_transform_edt(~ea)[eb],ndimage.distance_transform_edt(~eb)[ea]]*.1)
    def result(self):
        return dict(metrics={k:dict(mean=self.sums[k]/n if n else None,n=n) for k,n in self.counts.items()},gap_boundary_missing_frames=self.boundary_missing,corruption_counts=self.corruption,wall_batch_ms={key:dict(n=len(v),mean=float(np.mean(v)),p95=float(np.quantile(v,.95))) for key,v in self.cpu.items()})

@torch.inference_mode()
def run(env,teacher,student,attention,transport,args,agent_cfg,env_cfg,teacher_path,iteration):
    from gd_lab.core.camera_transport import delivery_mask
    from gd_lab.gast.observations import TerrainHistory,pose_xyyaw,clean_terrain
    from gd_lab.mdp.terrain_families import terrain_family_gate
    from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
    from gain_control import freeze_gain_resampling
    base=env.unwrapped;n=base.num_envs;device=base.device;dt=base.step_dt
    arm=os.environ['U116_ARM'];noise=os.environ['U116_NOISE'];ir_mode=os.environ['U116_IR'];family_name=os.environ['U116_FAMILY']
    assert arm in ('a','C','B4','h0','h1','c') and n==32 and abs(dt-.01)<1e-9
    student.eval()
    for p in student.parameters():p.requires_grad_(False)
    fix_velocity(base.command_manager.get_term('base_velocity'),args.replay_vx)
    obs,_=env.reset();gain_control=freeze_gain_resampling(base);initial_environment=environment_metadata(base)
    hidden=student.init_hidden(n,device);raw=torch.zeros(n,32,device=device);gate=torch.zeros(n,1,device=device)
    episodes=torch.zeros(n,dtype=torch.long,device=device);last=torch.full_like(episodes,-1)
    switch=LatentSwitch(n,device);geometry=GapTileGeometry(base)
    family=lambda:terrain_family_gate(base,(FAMILIES[family_name],))==0
    rec=Recorder(base,switch,arm,geometry,family) if family_name=='gap' else StairRecorder(base,family)
    maps=[ElevationMap(arm if arm in ('h0','h1') else 'h0') for _ in range(n)];history=TerrainHistory(base);diag=Diagnostics()
    initial=torch.cat((base.scene['robot'].data.root_state_w,base.scene['robot'].data.joint_pos),-1).cpu().numpy()
    initial_hash=hashlib.sha256(initial.tobytes()).hexdigest();initial_terrain={'levels':base.scene.terrain.terrain_levels.cpu().tolist(),'types':base.scene.terrain.terrain_types.cpu().tolist()}
    next_capture=transport.next_capture(base.common_step_counter);base._vrl_camera_capture_step=next_capture
    start=int(base.common_step_counter);steps=round(args.replay_seconds/dt);packet_events=[];max_equivalence=0.
    # Preserve B3 E1's post-step capture/receive/action ordering exactly.
    try:
        for k in range(steps):
            step=int(base.common_step_counter);stamp=step*dt
            good=teacher.terrain_latent(obs)
            if arm in ('a','C','B4'):latent=attention.latent*((stamp-attention.stamp<.3)&attention.ready)[:,None]
            elif arm=='c':latent=good
            else:
                tick=time.perf_counter();poses=pose_xyyaw(base).cpu().numpy();scanner=base.scene['height_scanner'].data.pos_w[:,2].cpu().numpy()
                sampled=[]
                for e,m in enumerate(maps):
                    if m.origin is None:sampled.append(np.zeros(374,np.float32))
                    else:sampled.append(m.sample(poses[e],scanner[e],stamp)[0])
                packed=np.stack(sampled);terrain=torch.as_tensor(packed,device=device)
                view=obs.copy();view['gast_history']=history.update(base,terrain);latent=teacher.terrain_latent(view)
                diag.cpu.setdefault('sample_history_encoder',[]).append((time.perf_counter()-tick)*1000)
                if k%10==0:diag.cells(packed[family().cpu().numpy()],clean_terrain(base).cpu().numpy()[family().cpu().numpy()])
            actions=teacher.act_with_terrain_latent(obs,latent)
            if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action; abort evaluation')
            if arm=='c':max_equivalence=max(max_equivalence,float((actions-teacher.act_inference(obs)).abs().max()))
            if k%10==0:
                slots,_=geometry.for_envs(base);dist=slots[:,1,0]+base.scene.env_origins[:,0]-base.scene['robot'].data.root_pos_w[:,0]
                cosine=torch.nn.functional.cosine_similarity(latent,good,dim=-1).cpu().numpy();distance=dist.cpu().numpy();enabled=family().cpu().numpy()
                for label,lo,hi in [('near',0,.3),('approach',.3,1.5),('far',1.5,float('inf')),('past',-float('inf'),0)]:
                    mask=enabled&(distance>=lo)&(distance<hi) if family_name=='gap' else enabled if label=='far' else np.zeros(n,bool)
                    diag.add('latent_cosine_'+label,cosine[mask])
            rec.sample();obs,_,done,_=env.step(actions)
            reset=done.reshape(-1).bool();hidden[reset]=0;raw[reset]=0;gate[reset]=0
            episodes+=reset.long();attention.reset(reset);switch.reset(reset);history.reset(reset.nonzero().flatten())
            for e in reset.nonzero().flatten().tolist():maps[e].reset()
            step=int(base.common_step_counter)
            if step==next_capture:
                assert (base._vrl_camera_snapshot_steps==step).all()
                tick=time.perf_counter();snap=base._vrl_camera_snapshot
                frames,depth,positions,ros,ks=[v.detach().cpu().numpy().copy() for v in snap]
                # Snapshot rotations are ROS; mapper is OpenGL. q_gl=q_ros*[0,1,0,0].
                quats=np.stack((-ros[...,1],ros[...,0],ros[...,3],-ros[...,2]),-1)
                body_pos=base.scene['robot'].data.root_pos_w.cpu().numpy().copy();body_quat=base.scene['robot'].data.root_quat_w.cpu().numpy().copy()
                validity=np.isfinite(depth)&(depth>=.15)&(depth<5)
                for e in range(n):
                    rng=np.random.default_rng(np.random.SeedSequence([agent_cfg.seed,step,e,116]))
                    depth[e],validity[e],frames[e,:,1],counts=corrupt(depth[e],validity[e],frames[e,:,1],positions[e],quats[e],ks[e],body_pos[e,2]-.5,rng,noise,ir_mode)
                    for key,value in counts.items():diag.corruption[key]=diag.corruption.get(key,0)+value
                frames[:,:,0]=(np.clip(depth,.15,5)-.15)/4.85
                pose=torch.cat((pose_xyyaw(base),base.scene['robot'].data.root_quat_w),-1).clone()
                capture_stamp=torch.full((n,),step*dt,device=device)
                transport.capture(step,(torch.as_tensor(frames,device=device),episodes.clone(),pose,capture_stamp,depth,validity,positions,quats,ks,body_pos,body_quat))
                packet_events.append(['capture',step,transport.dropped]);next_capture=transport.next_capture(step);base._vrl_camera_capture_step=next_capture
                diag.cpu.setdefault('sensor_preprocess',[]).append((time.perf_counter()-tick)*1000)
            for packet in transport.receive(step):
                frames,captured,pose,capture_stamp,depth,validity,positions,quats,ks,body_pos,body_quat=packet.payload
                valid=delivery_mask(packet.capture_step,captured,episodes,last);ids=valid.nonzero().flatten();packet_events.append(['receive',packet.capture_step,packet.delivery_step,step])
                if not len(ids):continue
                if arm in ('a','C','B4'):
                    output,memory,_,_,quality=student.encode(frames[ids].clamp_min(0),hidden[ids],pose[ids],(step-packet.capture_step)*dt,torch.ones(len(ids),device=device))
                    hidden[ids]=memory;attention.latent[ids]=output;attention.ready[ids]=True;attention.stamp[ids]=capture_stamp[ids]
                if arm in ('h0','h1'):
                    tick=time.perf_counter()
                    for e in ids.tolist():maps[e].integrate(depth[e],validity[e],frames[e,:,1].cpu().numpy(),positions[e],quats[e],ks[e],body_pos[e],body_quat[e],packet.capture_step*dt)
                    diag.cpu.setdefault('map_integrate',[]).append((time.perf_counter()-tick)*1000)
                last[ids]=packet.capture_step
            if k==0 or (k+1)%100==0:print(f'[UPDATE116] {arm}/{family_name}/{noise}/{ir_mode} step {k+1}/{steps}',flush=True)
        assert environment_metadata(base)==initial_environment
        records=rec.metrics.finish() if family_name=='gap' else rec.finish()
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result=dict(schema='update116_v1',arm=arm,condition=args.e1_condition,seed=agent_cfg.seed,noise=noise,ir=ir_mode,family=family_name,num_envs=n,seconds=steps*dt,student_iteration=iteration,student_sha256=digest(args.student_resume),teacher_sha256=digest(teacher_path),protocol_sha256=digest(Path(__file__).resolve().parents[2]/'protocol.json'),initial_state_sha256=initial_hash,initial_terrain=initial_terrain,initial_environment=initial_environment,gain_control=gain_control,camera_transport=transport.config.manifest(dt),transport_counts=dict(captured=transport.captured,dropped=transport.dropped,delivered=transport.delivered),packet_schedule_sha256=hashlib.sha256(json.dumps(packet_events).encode()).hexdigest(),max_teacher_action_equivalence_error=max_equivalence,approaches=records,diagnostics=diag.result(),definitions=dict(stairs='center-to-outer stair border +0.2 m, upright <0.5 rad; trunk >15N; pre-reset; window_end censored',map='capture-pose, delivery-gated; native TerrainHistory every 0.1s; no GT label input',synthetic_ir='favourable depth-based cue, not physical active IR validation',false_fill='stable finite fill vetoes gap; irreducible limitation'))
        result['schema']='update116_independent_ir_v2'
        result['definitions']['synthetic_ir']='Not used: aligned RGB luminance proxy before independent sensor corruption. No depth-derived IR.'
        result['definitions']['ir_missing']='-1 internal invalid marker; excluded from map IR evidence; zero for fixed student input'
        with Path(args.replay_out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        print('[UPDATE116] COMPLETE',args.replay_out,flush=True)
    finally:rec.close()
