"""Student-driven online DAgger. Imported only by explicitly launched 90.111 job."""
import copy,hashlib,json,time
from pathlib import Path
import torch
from gd_lab.gast.student_v3 import BeliefStudent
from gd_lab.gast.v3_state import BeliefBank
from gd_lab.gast.b2_objective import PacketBank,action_rows
from gd_lab.gast.b3_objective import WindowLoss
from b3_runner import TerrainGroups,mixed_velocity,budget_reason
from v3_sampling import ApproachSampler
from v3_labels import Labels

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
    return x

def run(env,teacher,original,attention,transport,args,agent_cfg,env_cfg,teacher_path,iteration):
    from gd_lab.gast.observations import pose_xyyaw
    b=env.unwrapped;n=b.num_envs;dev=b.device
    assert abs(b.step_dt-.01)<1e-9 and n==32
    assert digest(teacher_path)=='db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4'
    assert digest(args.student_resume)=='7d8ee014d6fa17608a4630514e0ca888aa80ca5825414175899c5f257ceeb4ee'
    original.eval().requires_grad_(False);teacher.teacher.eval().requires_grad_(False)
    model=BeliefStudent(original,args.v3_size).to(dev)
    if args.v3_resume:
        saved=torch.load(args.v3_resume,map_location=dev,weights_only=False)
        assert saved['student_arch']==model.architecture and saved['size']==args.v3_size
        assert saved['teacher_sha256']==digest(teacher_path)
        model.load_state_dict(saved['model'],strict=True)
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.Adam(params,lr=1e-4)
    mixed_velocity(b.command_manager.get_term('base_velocity'),n,dev)
    terrain=TerrainGroups(b);sampler=ApproachSampler(b,terrain.gap)
    with torch.no_grad():obs,_=env.reset()
    labels=Labels(b,terrain.gap);bank=BeliefBank(model,n,dev);baseline=PacketBank(original,n,dev)
    out=Path(args.b2_out);out.mkdir(parents=True,exist_ok=False)
    torch.save({'model':model.state_dict(),'student_arch':model.architecture,'size':args.v3_size,'teacher_sha256':digest(teacher_path),'manifest':{'initial_only':True}},out/'initial_student_v3.pt')
    meta={'architecture':model.architecture,'size':args.v3_size,'teacher_sha256':digest(teacher_path),'source_sha256':digest(args.student_resume),'resume':args.v3_resume,'seed':args.seed,'driver':'student_v3','split':'train','transport':transport.config.manifest(.01),'sampling':sampler.summary(),'proprio':'raw term-major obs[policy], 5 history frames, oldest->newest within term; exact observation manager dimensions in checkpoint','policy_term_dims':cpu(b.observation_manager.group_obs_term_dim['policy']),'trainable':[k for k,p in model.named_parameters() if p.requires_grad],'loss':'R(action_execution_same_detached_gate)+0.1R(raw_capture)+spatial_capture+mean(nonprep_34624_action_MSE)','round_id':args.v3_round,'bptt':'16 scheduled captures, all intervening control ticks'}
    meta['geometry_sha256']=b.cfg.scene.terrain.terrain_generator.v3_geometry_sha256
    meta['gap_metadata_sha256']=b.cfg.scene.terrain.terrain_generator.v3_gap_metadata_sha256
    meta['policy_terms']=b.observation_manager.active_terms['policy']
    source=torch.load(args.student_resume,map_location='cpu',weights_only=False)
    meta['camera_contract']=source['camera_contract']
    meta['actor_sha256']=digest(teacher_path)
    meta['driver_checkpoint_sha256']=digest(out/'initial_student_v3.pt')
    meta['initial_student_state_sha256']=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    code_root=Path(__file__).resolve().parents[2]
    meta['code_sha256']={str(p.relative_to(code_root)):digest(p) for p in [Path(__file__),code_root/'src/gd_lab/gast/student_v3.py',code_root/'src/gd_lab/gast/student.py',code_root/'src/gd_lab/gast/v3_state.py',code_root/'tools/rl_replay/v3_sampling.py',code_root/'tools/rl_replay/v3_labels.py']}
    (out/'config.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    next_capture=transport.next_capture(int(b.common_step_counter));b._vrl_camera_capture_step=next_capture
    start=time.monotonic();updates=captures=ticks=captures_in_window=0;window=WindowLoss('prep4');records=[];packets=[]
    log=(out/'windows.jsonl').open('w')
    def pose():return torch.cat((pose_xyyaw(b),b.scene['robot'].data.root_quat_w),-1).detach().clone()
    def flush():
        nonlocal records,packets
        torch.save({'ticks':records,'packets':packets,'manifest':meta},out/f'dagger_{updates:05d}.pt');records=[];packets=[]
    try:
        while True:
            step=int(b.common_step_counter)
            # Capture at current tick; transport may deliver now (zero delay).
            if step==next_capture:
                with torch.no_grad():
                    extra=tuple(v.detach().clone() for v in attention.capture(obs,teacher.act_inference(obs)))
                    payload=(b._vrl_camera_snapshot[0].detach().clone(),bank.episode.clone(),extra,terrain.labels(obs).clone())
                    transport.capture(step,payload)
                    packets.append(cpu({'capture_step':step,'payload':payload}))
                captures+=1;captures_in_window+=1;next_capture=transport.next_capture(step);b._vrl_camera_capture_step=next_capture
            for packet in transport.receive(step):
                frames,epochs,extra,capture_groups=packet.payload
                if not all(torch.isfinite(extra[k]).all() for k in (1,2,6)):raise RuntimeError('nonfinite capture supervision')
                ids,raw_capture,spatial=bank.publish(frames,extra[3],packet.capture_step,packet.delivery_step,epochs,step)
                packets.append(cpu({'capture_step':packet.capture_step,'delivery_step':packet.delivery_step,'accepted_ids':ids}))
                if not len(ids):continue
                with torch.no_grad():
                    _,memory,_,_,quality=original.encode(frames[ids],baseline.hidden[ids],extra[3][ids],(packet.delivery_step-packet.capture_step)*.01,torch.ones(len(ids),device=dev))
                    baseline.publish(ids,memory,original.head(memory[:,:5984].reshape(-1,187,32)),quality,extra[4][ids],packet.capture_step)
                window.raw.add((raw_capture-extra[6][ids].detach()).square().mean(-1),capture_groups[ids]);window.spatial(spatial,extra[1][ids],extra[2][ids])
            with torch.no_grad():
                packed=teacher._actor_input(obs,inference=True).detach().clone();base=packed[:,:-32];target=packed[:,-32:]
                group=terrain.labels(obs);gate=bank.effective(step).detach();proprio=obs['policy'].detach().clone()
                if proprio.shape!=(n,230):raise ValueError('policy history must be raw term-major [B,230]')
                current_pose=pose()
            if not torch.isfinite(proprio).all():raise RuntimeError('nonfinite proprio')
            pre_visual=bank.hidden.detach().clone() if ticks==0 else None
            pre_belief=bank.belief.detach().clone() if ticks==0 else None
            raw=bank.tick(proprio,current_pose,step)
            values,action=action_rows(teacher.actor,base,raw,target,gate)
            window.action.add(values,group,gate[:,0]>0)
            with torch.no_grad():reference=teacher.actor(torch.cat((base,baseline.raw*baseline.effective(step*.01)),-1))
            window.preserve((action-reference).square().mean(-1),group,torch.ones(n,device=dev,dtype=torch.bool))
            record=cpu({'student_update':updates,'step':step,'epoch':bank.episode,'proprio':proprio,'pose_source':'sim_root_odometry_proxy','pose':current_pose,'previous_action':bank.previous_action,'teacher_execution_raw':target,'baseline_action':reference,'action':action,'gate':gate,'quality':bank.gate,'age':bank.inputs(step)[0],'group':group,'labels':labels.read(),'h_visual':pre_visual,'h_belief':pre_belief})
            if not torch.isfinite(action).all():raise RuntimeError('nonfinite action')
            executed=action.detach().clone()
            if env.clip_actions is not None:executed=executed.clamp(-env.clip_actions,env.clip_actions)
            with torch.no_grad():obs,_,done,_=env.step(executed)
            # Cache before auto-reset clears action_manager on terminal rows.
            bank.previous_action=executed
            record['executed_action']=cpu(executed);record['done']=cpu(done);record['terminal']=cpu(labels.terminal);labels.terminal.clear();records.append(record)
            done=done.reshape(-1).bool();bank.reset(done,int(b.common_step_counter));baseline.reset(done)
            ticks+=1
            if captures_in_window>=16:
                optimizer.zero_grad(set_to_none=True);loss,stats=window.value(raw.new_zeros(()))
                if loss.requires_grad:
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
                    optimizer.step()
                flush();updates+=1;bank.detach();ticks=captures_in_window=0;window=WindowLoss('prep4')
                stats.update(driver_state_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest(),updates=updates,captures=captures,elapsed=time.monotonic()-start,rejected=bank.counters,dropped=transport.dropped)
                log.write(json.dumps(stats)+'\n');log.flush()
                reason=budget_reason(updates,time.monotonic()-start)
                if reason:break
        torch.save({'model':model.state_dict(),'student_arch':model.architecture,'size':args.v3_size,'teacher_sha256':digest(teacher_path),'manifest':meta,'updates':updates,'optimizer':optimizer.state_dict()},out/'student_v3.pt')
        (out/'complete.json').write_text(json.dumps({'reason':reason,'updates':updates,'captures':captures,'pending_packets_at_budget':len(transport.pending)}))
    finally:
        # Incomplete episodes explicitly aborted; no success inferred from cutoff.
        (out/'abort.json').write_text(json.dumps({'reason':'runner_exit','step':int(b.common_step_counter),'open_envs':list(range(n))}))
        if records:flush()
        labels.close();sampler.close();log.close()
