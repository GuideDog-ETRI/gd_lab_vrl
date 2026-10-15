"""Frozen UPDATE116 rollout. Sensor map never receives clean terrain labels."""
import hashlib,json,os,time,types
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from e1_runner import load_teacher,configure as original_configure,Recorder
from e1_core import LatentSwitch
from runtime import fix_velocity,environment_metadata
from heightmap121 import BatchMap
from sensor121 import IsaacSensor
from audit121 import Audit
from blocks118 import SEEDS


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
    assert arm in ('h0','h0d','h0d2','h0p','h0c','h2','h2m','h3','h3u') and (n==32 or os.environ.get('U118_CPU_TEST')=='1') and abs(dt-.01)<1e-9
    assert noise=='realistic' and ir_mode=='physical'
    wall_start=time.perf_counter()
    fix_velocity(base.command_manager.get_term('base_velocity'),args.replay_vx)
    obs,_=env.reset()
    gain_mode=os.environ.get('U118_GAIN_MODE','freeze110')
    assert gain_mode in ('freeze110','reset104')
    gain_control=freeze_gain_resampling(base) if gain_mode=='freeze110' else {'mode':'reset_resampling_update104'}
    if 'original' in gain_control:gain_control['original']='block_seeded_randomize_actuator_gains'
    initial_environment=environment_metadata(base)
    episodes=torch.zeros(n,dtype=torch.long,device=device);last=torch.full_like(episodes,-1)
    switch=LatentSwitch(n,device);geometry=GapTileGeometry(base)
    family=lambda:terrain_family_gate(base,(FAMILIES[family_name],))==0
    rec=Recorder(base,switch,arm,geometry,family) if family_name=='gap' else StairRecorder(base,family)
    maps=BatchMap(n,arm);history=TerrainHistory(base);diag=Diagnostics()
    sensor=IsaacSensor(base,geometry,42);audit=Audit(n);audit_packets={}
    cpu_seconds={'map_integrate':0.,'map_sample':0.};cpu_env_calls={'map_integrate':0,'map_sample':0}
    initial=torch.cat((base.scene['robot'].data.root_state_w,base.scene['robot'].data.joint_pos,base.scene['robot'].data.joint_vel),-1).cpu().numpy()
    initial_hash=hashlib.sha256(initial.tobytes()).hexdigest();initial_terrain={'levels':base.scene.terrain.terrain_levels.cpu().tolist(),'types':base.scene.terrain.terrain_types.cpu().tolist()}
    terrain_grid_sha256=hashlib.sha256(base.scene.terrain.terrain_origins.cpu().numpy().tobytes()).hexdigest()
    initial_blocks=[]
    for bi,seed in enumerate(SEEDS):
        sl=slice(bi*32,(bi+1)*32)
        gains={name:{key:value[sl] if isinstance(value,list) and len(value)==n else value for key,value in vals.items()} for name,vals in initial_environment['actuator_gains'].items()}
        initial_blocks.append(dict(seed=seed,env_range=[bi*32,(bi+1)*32],initial_state_sha256=hashlib.sha256(initial[sl].tobytes()).hexdigest(),gain_sha256=hashlib.sha256(json.dumps(gains,sort_keys=True).encode()).hexdigest(),terrain_levels=initial_terrain['levels'][sl],terrain_types=initial_terrain['types'][sl]))
    from gd_lab.core.camera_transport import CameraTransport
    transports=[CameraTransport(transport.config,dt,seed) for seed in SEEDS]
    next_captures=[t.next_capture(base.common_step_counter) for t in transports]
    next_capture=min(next_captures);base._vrl_camera_capture_step=next_capture
    block_diags=[Diagnostics() for _ in SEEDS]
    from flat118 import FlatPitchGeometry118
    if getattr(rec,'flat',None) is not None:rec.flat=FlatPitchGeometry118(base,geometry)
    start=int(base.common_step_counter);steps=round(args.replay_seconds/dt);packet_events=[];max_equivalence=0.
    # Preserve B3 E1's post-step capture/receive/action ordering exactly.
    try:
        for k in range(steps):
            step=int(base.common_step_counter);stamp=step*dt
            poses=pose_xyyaw(base).cpu().numpy();scanner=base.scene['height_scanner'].data.pos_w[:,2].cpu().numpy()
            tick=time.perf_counter();cpu_tick=time.process_time()
            packed=maps.sample(poses,scanner,stamp)
            cpu_seconds['map_sample']+=time.process_time()-cpu_tick;cpu_env_calls['map_sample']+=n
            diag.cpu.setdefault('map_sample',[]).append((time.perf_counter()-tick)*1000)
            tick=time.perf_counter()
            terrain=torch.as_tensor(packed,device=device)
            view=obs.copy();view['gast_history']=history.update(base,terrain);latent=teacher.terrain_latent(view)
            diag.cpu.setdefault('sample_history_encoder',[]).append((time.perf_counter()-tick)*1000)
            if k%10==0:
                enabled=family().cpu().numpy();gt=clean_terrain(base).cpu().numpy()
                diag.cells(packed[enabled],gt[enabled])
                audit.sample(maps,packed,poses,gt,enabled,stamp,family_name)
                for bi,d in enumerate(block_diags):
                    mask=enabled.copy();mask[:bi*32]=False;mask[(bi+1)*32:]=False
                    d.cells(packed[mask],gt[mask])
            actions=teacher.act_with_terrain_latent(obs,latent)
            if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action; abort evaluation')
            rec.sample();obs,_,done,_=env.step(actions)
            reset=done.reshape(-1).bool()
            episodes+=reset.long();switch.reset(reset);history.reset(reset.nonzero().flatten())
            reset_ids=reset.nonzero().flatten().tolist()
            if maps is not None:maps.reset(reset_ids)
            audit.reset(reset_ids)
            step=int(base.common_step_counter)
            if step==next_capture:
                assert (base._vrl_camera_snapshot_steps==step).all()
                tick=time.perf_counter();snap=base._vrl_camera_snapshot
                frames,depth,positions,ros,ks=[v.detach().cpu().numpy().copy() for v in snap]
                # Snapshot rotations are ROS; mapper is OpenGL. q_gl=q_ros*[0,1,0,0].
                quats=np.stack((-ros[...,1],ros[...,0],ros[...,3],-ros[...,2]),-1)
                body_pos=base.scene['robot'].data.root_pos_w.cpu().numpy().copy();body_quat=base.scene['robot'].data.root_quat_w.cpu().numpy().copy()
                generated=sensor.capture(positions,quats,ks,episodes.cpu().numpy(),step,depth.shape[-2:])
                validity=np.empty_like(depth,dtype=bool);labels=[]
                for e,(d,v,ir,counts,lab) in enumerate(generated):
                    depth[e]=d;validity[e]=v;frames[e,:,1]=ir;labels.append(lab)
                    for target in (diag,block_diags[e//32]):
                        for key,value in counts.items():target.corruption[key]=target.corruption.get(key,0)+value
                audit_packets[step]=labels
                audit_packets={key:value for key,value in audit_packets.items() if step-key<=30}
                frames[:,:,0]=(np.clip(depth,.15,5)-.15)/4.85
                pose=torch.cat((pose_xyyaw(base),base.scene['robot'].data.root_quat_w),-1).clone()
                capture_stamp=torch.full((n,),step*dt,device=device)
                payload=(torch.as_tensor(frames,device=device),episodes.clone(),pose,capture_stamp,depth,validity,positions,quats,ks,body_pos,body_quat)
                for bi,t in enumerate(transports):
                    if step!=next_captures[bi]:continue
                    t.capture(step,payload)
                    packet_events.append([bi,'capture',step,t.dropped])
                    next_captures[bi]=t.next_capture(step)
                next_capture=min(next_captures);base._vrl_camera_capture_step=next_capture
                diag.cpu.setdefault('sensor_preprocess',[]).append((time.perf_counter()-tick)*1000)
            for bi,t in enumerate(transports):
                for packet in t.receive(step):
                    frames,captured,pose,capture_stamp,depth,validity,positions,quats,ks,body_pos,body_quat=packet.payload
                    valid=delivery_mask(packet.capture_step,captured,episodes,last)
                    valid[:bi*32]=False;valid[(bi+1)*32:]=False
                    ids=valid.nonzero().flatten();packet_events.append([bi,'receive',packet.capture_step,packet.delivery_step,step])
                    if not len(ids):continue
                    if arm in ('h0','h0d','h0d2','h0p','h0c','h2','h2m','h3','h3u'):
                        tick=time.perf_counter();cpu_tick=time.process_time()
                        maps.integrate(ids.tolist(),depth,validity,frames[:,:,1].cpu().numpy(),positions,quats,ks,body_pos,body_quat,packet.capture_step*dt,now=step*dt,pose_sigma=.01)
                        audit.deliver(ids.tolist(),audit_packets[packet.capture_step],packet.capture_step*dt)
                        cpu_seconds['map_integrate']+=time.process_time()-cpu_tick;cpu_env_calls['map_integrate']+=len(ids)
                        diag.cpu.setdefault('map_integrate',[]).append((time.perf_counter()-tick)*1000)
                    last[ids]=packet.capture_step
            if k==0 or (k+1)%100==0:print(f'[UPDATE121] {arm}/{family_name}/{noise}/{ir_mode} step {k+1}/{steps}',flush=True)
        if gain_mode=='freeze110':assert environment_metadata(base)==initial_environment
        records=rec.metrics.finish() if family_name=='gap' else rec.finish()
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result=dict(schema='update116_v1',arm=arm,condition=args.e1_condition,seed=agent_cfg.seed,noise=noise,ir=ir_mode,family=family_name,num_envs=n,seconds=steps*dt,student_iteration=iteration,student_sha256=None,teacher_sha256=digest(teacher_path),protocol_sha256=digest(Path(__file__).resolve().parents[2]/'experiments/e1/protocol121.json'),initial_state_sha256=initial_hash,initial_terrain=initial_terrain,initial_environment=initial_environment,gain_control=gain_control,camera_transport=transport.config.manifest(dt),transport_counts=dict(captured=transport.captured,dropped=transport.dropped,delivered=transport.delivered),packet_schedule_sha256=hashlib.sha256(json.dumps(packet_events).encode()).hexdigest(),max_teacher_action_equivalence_error=max_equivalence,approaches=records,diagnostics=diag.result(),definitions=dict(stairs='center-to-outer stair border +0.2 m, upright <0.5 rad; trunk >15N; pre-reset; window_end censored',map='capture-pose, delivery-gated; native TerrainHistory every 0.1s; no GT label input',synthetic_ir='favourable depth-based cue, not physical active IR validation',false_fill='stable finite fill vetoes gap; irreducible limitation'))
        result['schema']='update121_seed42_32env'
        result['initial_blocks']=initial_blocks
        result['terrain_grid_sha256']=terrain_grid_sha256
        result['block_diagnostics']=[d.result() for d in block_diags]
        result['packet_events']=packet_events
        result['block_transport']=[dict(seed=seed,captured=t.captured,dropped=t.dropped,delivered=t.delivered,schedule_sha256=hashlib.sha256(json.dumps([x for x in packet_events if x[0]==bi]).encode()).hexdigest()) for bi,(seed,t) in enumerate(zip(SEEDS,transports))]
        result['transport_counts']={key:sum(getattr(t,key) for t in transports) for key in ('captured','dropped','delivered')}
        result['camera_transport']['scope']='single_seed42_32env_four_camera_set'
        result['worker_cpu_seconds']={}
        result['steps_per_wall_second']=steps/(time.perf_counter()-wall_start)
        result['environment']=environment_metadata(base)
        result['start_step']=start
        result['policy_steps']=steps
        if getattr(rec,'flat',None) is not None:result['update34_flat_pitch']=rec.flat.result()
        result['latent_comparison_enabled']=False
        result['rollout_wall_seconds']=time.perf_counter()-wall_start
        result['map_cpu_seconds']=cpu_seconds
        result['map_cpu_env_calls']=cpu_env_calls
        result['map_cpu_ms_per_env_call']={key:1000*cpu_seconds[key]/max(cpu_env_calls[key],1) for key in cpu_seconds}
        result['definitions']['synthetic_ir']='Not used: aligned RGB luminance proxy before independent sensor corruption. No depth-derived IR.'
        result['definitions']['ir_missing']='-1 internal invalid marker; excluded from map IR evidence; zero for fixed student input'
        result['audit_metrics']=audit.result()
        result['sensor_patch_areas']=[dict(key=[float(v) for v in key],fractions=[p.area_fraction for p in fields]) for key,fields in sensor.fields.items()]
        result['definitions']['synthetic_ir']='Physical IR from pre-corruption terrain mesh hit, world reflectance, projector occlusion, exposure and independent noise.'
        result['definitions']['false_fill']='World-fixed clipped polygons; 25% quadrature target, actual injection and surviving counts reported.'
        result['provenance_file']=Path(args.replay_out).with_suffix('.npz').name
        audit.save(Path(args.replay_out).with_suffix('.npz'))
        with Path(args.replay_out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        print('[UPDATE121] COMPLETE',args.replay_out,flush=True)
    finally:
        rec.close()
        # Serial BatchMap owns no workers or external resources.
