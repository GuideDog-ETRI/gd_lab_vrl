"""Frozen-teacher E1, UPDATE121 dynamics/history/metric definitions preserved."""
import os,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from e1_runner import load_teacher,configure,Recorder
from e1_core import LatentSwitch
from runtime import fix_velocity,environment_metadata
from e1_heightmap121 import Diagnostics,StairRecorder,FAMILIES,configure as _family_configure
from lidar126 import Lidar,Maps,SENSOR_COUNT
configure=_family_configure  # UPDATE129: U116_FAMILY selects gap or stairs, as in the camera path

@torch.inference_mode()
def run(env,teacher,student,attention,transport,args,agent_cfg,env_cfg,teacher_path,iteration):
    from gd_lab.core.camera_transport import delivery_mask
    from gd_lab.gast.observations import TerrainHistory,pose_xyyaw,clean_terrain
    from gd_lab.mdp.terrain_families import terrain_family_gate
    from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
    from gain_control import freeze_gain_resampling
    base=env.unwrapped;n=base.num_envs;device=base.device;dt=base.step_dt;arm=os.environ['U126_ARM']
    assert arm in ('L0','L0M','L0D','L0D2','L0P','L0S','G','GV','L1','L2') and n==32 and abs(dt-.01)<1e-9 and student is None
    begin=time.perf_counter();fix_velocity(base.command_manager.get_term('base_velocity'),args.replay_vx)
    obs,_=env.reset();gain_control=freeze_gain_resampling(base)
    if 'original' in gain_control:gain_control['original']='block_seeded_randomize_actuator_gains'
    initial_environment=environment_metadata(base)
    initial=torch.cat((base.scene['robot'].data.root_state_w,base.scene['robot'].data.joint_pos,base.scene['robot'].data.joint_vel),-1).cpu().numpy()
    initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
    terrain={'levels':base.scene.terrain.terrain_levels.cpu().tolist(),'types':base.scene.terrain.terrain_types.cpu().tolist()}
    gridsha=hashlib.sha256(base.scene.terrain.terrain_origins.cpu().numpy().tobytes()).hexdigest()
    episodes=torch.zeros(n,dtype=torch.long,device=device);last=torch.full_like(episodes,-1)
    switch=LatentSwitch(n,device);geometry=GapTileGeometry(base)
    family_name=os.environ.get('U116_FAMILY','gap')
    if family_name=='gap':
        family=lambda:terrain_family_gate(base,('platform_gap',))==0
        rec=Recorder(base,switch,arm,geometry,family)
        from flat118 import FlatPitchGeometry118
        if rec.flat is not None:rec.flat=FlatPitchGeometry118(base,geometry)
    else:
        family=lambda:terrain_family_gate(base,(FAMILIES[family_name],))==0
        rec=StairRecorder(base,family);rec.flat=None
    foot_samples=[];deep_samples=[];false_deep_locations=[];maps=Maps(n,arm);diag=Diagnostics()
    # UPDATE136: a Ray+IR teacher (3-channel CNN) gets [height, valid, IR] frames; IR from rayir_e1.CellIr.
    rayir=teacher.terrain_encoder.channels==3 if hasattr(teacher.terrain_encoder,'channels') else False
    if rayir:
        from rayir_e1 import RayIrHistory,CellIr
        history=RayIrHistory(base);cell_ir=CellIr(base,agent_cfg.seed);frame_width=8*(3*187+1)
    else:
        history=TerrainHistory(base);cell_ir=None;frame_width=8*375
    # UPDATE130 camera path: U130_SENSOR=cam swaps the LiDAR for the four vendor_new depth cameras (same packet).
    sensor_kind=os.environ.get('U130_SENSOR','lidar');assert sensor_kind in ('lidar','cam')
    if sensor_kind=='cam':
        from cam130 import CamSensor;sensor=CamSensor(base,geometry)
    else:sensor=Lidar(base)
    trace129=[]
    counters={k:0 for k in ['gap_cells','gap_direct','gap_carving','gap_unknown','flat_cells','false_gap_5cm','flat_sensor_beams','flat_sensor_hits','delivered_env_packets','foot_lane_cells','foot_lane_valid','gap_deep','flat_deep','gap_deep_carving','flat_deep_carving']}
    next_capture=transport.next_capture(base.common_step_counter);base._vrl_camera_capture_step=next_capture
    start=int(base.common_step_counter);steps=round(args.replay_seconds/dt);events=[];timing={'mapping':[],'sampling_history_actor':[]}
    # UPDATE157: no model/RNG/override is active when U157_POSE_ERR is unset.
    from pose_error157 import from_environ,PoseError,transform_packet
    pose_cfg=from_environ();pose_model=None;odom=None
    if pose_cfg is not None:
        if arm in ('G','GV'):
            raise ValueError('UPDATE157 requires a sensor mapper arm; G/GV are GT diagnostic paths')
        pose_model=PoseError(pose_cfg,n,dt,agent_cfg.seed,device)
        robot=base.scene['robot']
        odom=pose_model.update(pose_xyyaw(base),robot.data.root_lin_vel_w[:,:2].norm(dim=-1),robot.data.root_ang_vel_w[:,2])
    try:
        for k in range(steps):
            step=int(base.common_step_counter);stamp=step*dt;t=time.perf_counter()
            poses=pose_xyyaw(base).cpu().numpy();scanner=base.scene['height_scanner'].data.pos_w[:,2].cpu().numpy()
            map_poses=poses if pose_model is None else odom.cpu().numpy()
            packed=maps.sample(map_poses,scanner,stamp)
            if arm in ('G','GV'):
                # UPDATE129 diagnostics: ground-truth teacher map (G), or the same masked to LiDAR-observed cells (GV).
                gt_now=clean_terrain(base).cpu().numpy();vis=np.ones((n,187),bool) if arm=='G' else maps.visible(poses,stamp)
                vis&=gt_now[:,187:374]>0
                packed=np.concatenate((np.where(vis,gt_now[:,:187],0.),vis.astype(np.float64)),1).astype(np.float32);maps.last_sources=vis.astype(np.uint8)
            terrain_tensor=torch.as_tensor(packed,device=device)
            if cell_ir is not None:terrain_tensor=torch.cat((terrain_tensor,cell_ir(base,terrain_tensor[:,187:]>0)),-1)
            view=obs.copy()
            view['gast_history']=history.update(base,terrain_tensor) if pose_model is None else history.update(base,terrain_tensor,pose_override=odom)
            assert view['gast_history'].shape==(n,frame_width) and torch.isfinite(view['gast_history']).all()
            latent=teacher.terrain_latent(view);assert torch.isfinite(latent).all()
            actions=teacher.act_with_terrain_latent(obs,latent);assert torch.isfinite(actions).all()
            timing['sampling_history_actor'].append(time.perf_counter()-t)
            if k%10==0:
                gt=clean_terrain(base).cpu().numpy();enabled=family().cpu().numpy();gv=gt[:,187:374]>0;gap=(gt[:,374:561]>0)&gv&enabled[:,None]
                trace129.append(dict(step=step,pred=packed[:,:187].astype(np.float16),valid=(packed[:,187:]>0),gt=gt[:,:187].astype(np.float16),gap=gap,source=maps.last_sources.astype(np.int8) if maps.last_sources is not None else np.zeros((n,187),np.int8),x=poses[:,0].astype(np.float32)))
                diag.cells(packed[enabled],gt[enabled]);source=maps.last_sources;pv=packed[:,187:]>0  # UPDATE130C: validity from packed, not source
                counters['gap_cells']+=int(gap.sum())
                for label,code in [('direct',1),('carving',2),('unknown',0),('deepmem',3),('struct',4),('pending',5)]:counters['gap_'+label]=counters.get('gap_'+label,0)+int((gap&(source==code)).sum())
                # Evaluation-only live front-foot x in yaw-aligned policy coordinates.
                robot=base.scene['robot'];feet=robot.data.body_pos_w[:,[robot.body_names.index('FL_foot'),robot.body_names.index('FR_foot')],:2].cpu().numpy()
                rel=feet-poses[:,None,:2]
                fx=rel[:,:,0]*np.cos(poses[:,2,None])+rel[:,:,1]*np.sin(poses[:,2,None])
                xx=maps.direct.x[None,:];yy=maps.direct.y[None,:]
                frontx=np.where(yy>=0,fx[:,0,None],fx[:,1,None])
                lane=(xx-frontx>=-1e-9)&(xx-frontx<=.4+1e-9)&(abs(yy)>=.12)&(abs(yy)<=.26)&enabled[:,None]
                counters['foot_lane_cells']+=int(lane.sum());counters['foot_lane_valid']+=int((lane&pv).sum())
                foot_samples.append(np.stack([lane,pv],axis=1))
                flat=gv&~(gt[:,374:561]>0)&enabled[:,None]
                # Evaluation only: encoded deep cells and world coordinates relative to all slot edges.
                deep=pv&(packed[:,:187]>=4.999)
                counters['gap_deep']+=int((gap&deep).sum());counters['flat_deep']+=int((flat&deep).sum())
                counters['gap_deep_carving']+=int((gap&deep&(source==2)).sum());counters['flat_deep_carving']+=int((flat&deep&(source==2)).sum())
                deep_samples.append(np.stack([gap,flat,deep,source==2],axis=1))
                if family_name=='gap':slots,_=geometry.for_envs(base);edges=slots.cpu().numpy()+base.scene.env_origins[:,0].cpu().numpy()[:,None,None]
                ee,cc=np.nonzero(flat&deep) if family_name=='gap' else (np.array([],int),np.array([],int))
                for e,cell in zip(ee,cc):
                    wx=poses[e,0]+maps.direct.x[cell]*np.cos(poses[e,2])-maps.direct.y[cell]*np.sin(poses[e,2])
                    wy=poses[e,1]+maps.direct.x[cell]*np.sin(poses[e,2])+maps.direct.y[cell]*np.cos(poses[e,2])
                    edge=edges[e].reshape(-1);distance=float(np.min(np.abs(edge-wx)))
                    false_deep_locations.append(dict(step=step,env_id=int(e),cell=int(cell),world_x=float(wx),world_y=float(wy),policy_x=float(maps.direct.x[cell]),policy_y=float(maps.direct.y[cell]),nearest_slot_edge_distance_m=distance,slot_edges_world_x=edges[e].tolist(),source=int(source[e,cell]),height=float(packed[e,cell]),gt_height=float(gt[e,cell])))
                counters['flat_cells']+=int(flat.sum());counters['false_gap_5cm']+=int((flat&pv&((packed[:,:187]-gt[:,:187])/5>.05)).sum())
            rec.sample();obs,_,done,_=env.step(actions)
            reset=done.reshape(-1).bool();episodes+=reset.long();switch.reset(reset);ids=reset.nonzero().flatten();history.reset(ids);maps.reset(ids.tolist())
            if cell_ir is not None:cell_ir.reset(ids)
            if pose_model is not None:
                pose_model.reset(ids)
                robot=base.scene['robot']
                odom=pose_model.update(pose_xyyaw(base),robot.data.root_lin_vel_w[:,:2].norm(dim=-1),robot.data.root_ang_vel_w[:,2])
            step=int(base.common_step_counter)
            if step==next_capture:
                packet=sensor.capture(episodes.cpu().numpy(),step)
                if pose_model is not None:
                    packet=transform_packet(packet,odom.cpu().numpy())
                # Evaluator-only flat classification; never enters sensor or mapper.
                z=base.scene['height_scanner'].data.ray_hits_w[...,2].cpu().numpy()
                flat_env=np.isfinite(z).all(1)&(np.ptp(z,axis=1)<.03)
                counters['flat_sensor_beams']+=int(flat_env.sum()*SENSOR_COUNT*120*90)
                counters['flat_sensor_hits']+=int(packet['valid'][flat_env].sum())
                transport.capture(step,(packet,episodes.clone()));events.append(['capture',step,transport.dropped])
                next_capture=transport.next_capture(step);base._vrl_camera_capture_step=next_capture
            for delivery in transport.receive(step):
                packet,captured=delivery.payload;mask=delivery_mask(delivery.capture_step,captured,episodes,last);ids=mask.nonzero().flatten()
                events.append(['receive',delivery.capture_step,delivery.delivery_step,step])
                if len(ids):
                    t=time.perf_counter();maps.integrate(ids.tolist(),packet,delivery.capture_step*dt,step*dt);timing['mapping'].append(time.perf_counter()-t)
                    last[ids]=delivery.capture_step;counters['delivered_env_packets']+=len(ids)
            if k==0 or (k+1)%100==0:print(f'[UPDATE128] {arm} step {k+1}/{steps}',flush=True)
        assert environment_metadata(base)==initial_environment
        records=rec.metrics.finish() if family_name=='gap' else rec.finish()
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result=dict(schema='update128_lidar_l2_deep',arm=arm+('x2' if SENSOR_COUNT==2 and sensor_kind=='lidar' else '')+('_cam' if sensor_kind=='cam' else ''),sensor=sensor_kind,condition=args.e1_condition,seed=agent_cfg.seed,num_envs=n,seconds=steps*dt,teacher_sha256=digest(teacher_path),student_sha256=None,initial_state_sha256=initial_hash,initial_terrain=terrain,terrain_grid_sha256=gridsha,initial_environment=initial_environment,gain_control=gain_control,approaches=records,diagnostics=diag.result(),lidar=sensor.report(),mapper=maps.report(),cell_counts=counters,transport_config=transport.config.manifest(dt),transport_counts={key:getattr(transport,key) for key in ('captured','dropped','delivered')},packet_events=events,packet_schedule_sha256=hashlib.sha256(json.dumps(events).encode()).hexdigest(),start_step=start,policy_steps=steps,rollout_wall_seconds=time.perf_counter()-begin,nan_checks_passed=True,timing={name:dict(n=len(v),total_seconds=sum(v),mean_ms=1000*np.mean(v) if v else None,p95_ms=1000*np.quantile(v,.95) if v else None) for name,v in timing.items()},definitions=dict(map='Unchanged h0 5cm reducer, 2s TTL; native 11x17x8 TerrainHistory; L2 same carving certificate, output +5 valid 1; no labels into mapper',self_occlusion='Actual moving URDF collision triangles; foot spheres tessellated subdivision2',flat_false_gap='Known non-gap E1 cells with predicted height more than 5cm below GT; denominator all known non-gap cells, unknown excluded from numerator'))
        if rec.flat is not None:result['update34_flat_pitch']=rec.flat.result()
        result['update136_rayir']=cell_ir.summary() if cell_ir is not None else None
        if pose_model is not None:
            result['update157_pose_err']=pose_model.summary()
        Path(args.replay_out).with_name('false_deep_locations.json').write_text(json.dumps(false_deep_locations,indent=2,allow_nan=False))
        np.savez_compressed(Path(args.replay_out).with_name('deep_samples.npz'),samples=np.asarray(deep_samples))
        if getattr(maps,'dump',None):
            import pickle;pickle.dump(maps.dump,open(Path(args.replay_out).with_name('raw130.pkl'),'wb'))
        np.savez_compressed(Path(args.replay_out).with_name('trace129.npz'),**{k:np.stack([t[k] for t in trace129]) for k in trace129[0]})
        np.savez_compressed(Path(args.replay_out).with_name('foot_lane_samples.npz'),samples=np.asarray(foot_samples))
        with Path(args.replay_out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        print('[UPDATE128] COMPLETE',args.replay_out,flush=True)
    finally:rec.close()
