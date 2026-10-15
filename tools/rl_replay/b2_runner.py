"""UPDATE31 training adapter. No import starts Isaac or selects a CUDA device."""
import hashlib,json,os,time,types
from pathlib import Path
import torch
from gd_lab.gast.b2_objective import configure_student,groups,WindowLoss,action_rows,PacketBank

def budget_reason(captures,elapsed):
    if captures>=6000:return 'capture_limit'
    if elapsed>=5400:return 'time_limit'
    return None

def mixed_velocity(term,n,device):
    values=torch.tensor([.6,1.,1.2],device=device)[torch.arange(n,device=device)%3]
    def enforce():
        term.vel_command_b[:,0]=values;term.vel_command_b[:,1:]=0
        for name in ('is_standing_env','is_heading_env'):
            if hasattr(term,name):getattr(term,name)[:]=False
    for name in ('_resample_command','_update_command'):
        original=getattr(term,name)
        def wrapped(self,*args,_original=original,**kwargs):
            result=_original(*args,**kwargs);enforce();return result
        setattr(term,name,types.MethodType(wrapped,term))
    enforce();return values

class TerrainGroups:
    def __init__(self,base):
        from gd_lab.mdp.platform_gap_finetune import GapTileGeometry
        from gd_lab.mdp.gap_stair_v2 import _StairGeometry
        from gd_lab.mdp.platform_gap_terms import boarding_family_mask
        self.gap=GapTileGeometry(base);self.stair=_StairGeometry(base);self.base=base
        self.family=boarding_family_mask
    @torch.no_grad()
    def labels(self,obs):
        b=self.base;p=b.scene['robot'].data.root_pos_w-b.scene.env_origins
        slots,_=self.gap.for_envs(b);active=self.family(b)
        distance=slots[:,1,0]-p[:,0]
        gap_distance=(slots.flatten(1)-p[:,0,None]).abs().min(-1).values
        cols=b.scene.terrain.terrain_types.long();radius=p[:,:2].abs().max(-1).values
        edges=self.stair.edges[cols];delta=(radius[:,None]-edges).abs()
        delta=torch.where(torch.isfinite(delta),delta,float('inf')).min(-1).values
        safe=(~active | (gap_distance>1.3)) & (~self.stair.is_stair[cols] | (delta>1.3))
        clean=obs['gast_clean'];h=clean[:,:187]/5;valid=(clean[:,187:374]>.5)&torch.isfinite(h)
        span=torch.where(valid,h,-float('inf')).max(-1).values-torch.where(valid,h,float('inf')).min(-1).values
        flat=safe & (valid.float().mean(-1)>.9) & (span<=.02)
        return groups(distance,flat,active)

def save_checkpoint(student,optimizer,source,args,teacher_path,iteration,captures,updates,elapsed,path):
    # Preserve camera/export contracts; mark the frozen actor explicitly.
    payload={k:v for k,v in source.items() if k not in ('optimizer','model','student_policy_state','student_top5_selection','action_distillation')}
    payload.update(model=student.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,
        teacher_checkpoint=str(Path(teacher_path).resolve()),teacher_sha256=hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest(),
        action_distillation={'student_policy':'frozen','action_target':'execution_same_detached_gate'},
        teacher_behavior_target='execution_same_proprio_same_detached_student_gate',
        camera_noise_enabled=False,gap_observed_supervision=False,seen_supervision_seconds=0.,
        student_geometry_supervision='teacher_ray_visible_valid_cells_only',
        update31={'trainable':args.b2_trainable,'group_weight':args.b2_group_weight,'seed':args.seed,
          'scheduled_captures':captures,'optimizer_updates':updates,'elapsed_seconds':elapsed,
          'loss':'R(action_execution)+0.1*R(raw_capture)+spatial_capture','lr':1e-4,'clip':1.,'bptt':16,
          'synthetic_cpu_smoke':bool(getattr(args,'b2_synthetic_cpu_smoke',False))},
        camera_transport={'interval_ms':(70,100),'delay_ms':(0,150),'drop_probability':.05},
        update19_synthetic_export_smoke_NOT_TRAINED=bool(getattr(args,'b2_synthetic_cpu_smoke',False)))
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def run(env,teacher,student,attention,transport,args,agent_cfg,env_cfg,teacher_path,iteration):
    b=env.unwrapped;n=b.num_envs;dt=b.step_dt;device=b.device
    out=Path(args.b2_out).resolve();out.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.student_resume,map_location='cpu',weights_only=False)
    if iteration!=34624:raise ValueError('UPDATE31 requires control 34624')
    params=configure_student(student,args.b2_trainable)
    teacher.teacher.requires_grad_(False) if hasattr(teacher,'teacher') else teacher.requires_grad_(False)
    teacher.actor.eval();optimizer=torch.optim.Adam(params,lr=1e-4) # fresh, never restored
    speed=mixed_velocity(b.command_manager.get_term('base_velocity'),n,device)
    with torch.no_grad():obs,_=env.reset()
    bank=PacketBank(student,n,device);terrain=TerrainGroups(b)
    start=time.monotonic();captures=updates=0;window=WindowLoss(args.b2_group_weight)
    next_capture=transport.next_capture(b.common_step_counter);b._vrl_camera_capture_step=next_capture
    meta={'seed':agent_cfg.seed,'trainable':args.b2_trainable,'group_weight':args.b2_group_weight,'num_envs':n,
          'speeds':speed.cpu().tolist(),'transport':transport.config.manifest(dt),'max_captures':6000,'seconds_budget':5400,
          'source_sha256':hashlib.sha256(Path(args.student_resume).read_bytes()).hexdigest(),
          'trainable_parameters':[name for name,p in student.named_parameters() if p.requires_grad],
          'groups':'prep active gap d[.3,1]; flat scan span<=.02m, >90% valid, gap/stair edges>1.3m; other remaining rows',
          'spatial':'original masked cell reduction, no distance weighting',
          'note':'UPDATE31 B2 names training prescription, not UPDATE19 residual actor; actor remains 3-input frozen'}
    (out/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
    log=(out/'windows.jsonl').open('x',buffering=1)
    packets_log=(out/'packets.jsonl').open('x',buffering=1)
    gate_sum=gate_rows=gate_positive=0
    in_window=0;stop_capture=None;stop_reason=None
    try:
        while True:
            step=int(b.common_step_counter)
            with torch.no_grad():
                # Copy all environment targets: env.step may mutate observation buffers.
                packed=teacher._actor_input(obs,inference=True).detach().clone()
                base=packed[:,:-32];teacher_current=packed[:,-32:]
                labels=terrain.labels(obs);gate=bank.effective(step*dt)
                gate_sum+=float(gate.sum());gate_rows+=gate.numel();gate_positive+=int((gate>0).sum())
            values,action=action_rows(teacher.actor,base,bank.raw,teacher_current,gate)
            window.action.add(values,labels,gate[:,0]>0)
            with torch.no_grad():obs,_,done,_=env.step(action.detach())
            bank.reset(done.reshape(-1).bool())
            step=int(b.common_step_counter)
            if stop_capture is None and step==next_capture:
                with torch.no_grad():
                    extra=attention.capture(obs,teacher.act_inference(obs))
                    extra=tuple(v.detach().clone() if torch.is_tensor(v) else v for v in extra)
                    payload=(b._vrl_camera_snapshot[0].detach().clone(),bank.episode.clone(),extra,terrain.labels(obs).clone())
                    transport.capture(step,payload)
                captures+=1;in_window+=1
                stop_reason=budget_reason(captures,time.monotonic()-start)
                if stop_reason:
                    stop_capture=step
                    b._vrl_camera_capture_step=-1
                else:
                    next_capture=transport.next_capture(step);b._vrl_camera_capture_step=next_capture
            for packet in transport.receive(step):
                frames,epochs,extra,capture_groups=packet.payload
                valid=bank.valid(packet.capture_step,epochs)
                for value in (frames,extra[1],extra[2],extra[3],extra[6]):valid &= torch.isfinite(value).flatten(1).all(-1)
                ids=valid.nonzero().flatten()
                packets_log.write(json.dumps({'capture_step':packet.capture_step,'delivery_step':step,
                    'delay_seconds':(step-packet.capture_step)*dt,'accepted_rows':len(ids),
                    'rejected_rows':n-len(ids),'episodes':epochs.cpu().tolist()})+'\n')
                if not len(ids):continue
                _,memory,spatial,_,quality=student.encode(frames[ids],bank.hidden[ids],extra[3][ids],(step-packet.capture_step)*dt,torch.ones(len(ids),device=device))
                raw=student.head(memory[:,:187*32].reshape(-1,187,32))
                if not torch.isfinite(raw).all() or not torch.isfinite(memory).all():raise RuntimeError('nonfinite student outputs')
                bank.publish(ids,memory,raw,quality,extra[4][ids],packet.capture_step)
                window.raw.add((raw-extra[6][ids].detach()).square().mean(-1),capture_groups[ids])
                window.spatial(spatial,extra[1][ids],extra[2][ids])
            final=stop_capture is not None and step>=stop_capture+transport.delay[1]+1
            # Final extra policy tick uses the last delivered packet before saving.
            boundary=(in_window>=16 and stop_capture is None) or final
            if boundary:
                optimizer.zero_grad(set_to_none=True)
                loss,stats=window.value(bank.raw.new_zeros(()))
                if loss.requires_grad:
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
                    optimizer.step();updates+=1;stats['grad_norm']=float(norm)
                bank.detach();stats.update(captures=captures,updates=updates,elapsed=time.monotonic()-start)
                stats.update(effective_gate_mean=gate_sum/max(1,gate_rows),effective_gate_rows=gate_rows,
                             positive_gate_rows=gate_positive,dropped_captures=transport.dropped,
                             delivered_packets=transport.delivered)
                log.write(json.dumps(stats)+'\n')
                gate_sum=gate_rows=gate_positive=0
                window=WindowLoss(args.b2_group_weight);in_window=0
                if final:break
        elapsed=time.monotonic()-start
        save_checkpoint(student,optimizer,source,args,teacher_path,iteration+captures,captures,updates,elapsed,out/f'perception_{iteration+captures}.pt')
        (out/'complete.json').write_text(json.dumps({'reason':stop_reason,'captures':captures,'updates':updates,'elapsed':elapsed},indent=2)+'\n')
    finally:
        log.close();packets_log.close()
