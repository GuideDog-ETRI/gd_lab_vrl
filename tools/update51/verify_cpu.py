import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx,onnxruntime as ort
from gd_lab.gast.student import GastStudent
from gd_lab.gast.student_v3 import BeliefStudent,EncodeGraph,TickGraph,relative_pose
from gd_lab.gast.v3_state import BeliefBank
from gd_lab.gast.b2_objective import action_rows
from gd_lab.gast.b3_objective import WindowLoss

torch.set_num_threads(1);torch.manual_seed(51)
root=Path(__file__).resolve().parents[2];out=root/'outputs/update51_cpu';out.mkdir(exist_ok=True)
source=root/'checkpoints/students/control_seen0_perception_34624.pt'
checkpoint=torch.load(source,map_location='cpu',weights_only=False)
teacher=root/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
assert hashlib.sha256(teacher.read_bytes()).hexdigest().startswith('db500e96')
visual=GastStudent();visual.load_state_dict(checkpoint['model'],strict=True)
report={'trained':False,'device':'cpu','provider':'CPUExecutionProvider','teacher_sha256':hashlib.sha256(teacher.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'models':{}}
for size in ('small','large'):
    model=BeliefStudent(visual,size)
    for k,v in model.visual.state_dict().items():assert torch.equal(v,visual.state_dict()[k]),k
    n=2;hv,hb=model.init_hidden(n,'cpu');pose=torch.zeros(n,7);pose[:,3]=1
    frames=torch.rand(n,4,2,45,80);age=torch.full((n,1),.05);yes=torch.ones(n,dtype=torch.bool);no=~yes
    encoded=model.encode(frames,pose,age,yes,no,hv)
    # Exact inherited visual path parity in initial relative coordinate frame.
    old=visual.encode(frames,hv,pose,.05,torch.ones(n))
    torch.testing.assert_close(encoded[0],old[1]);torch.testing.assert_close(encoded[2],old[2]);torch.testing.assert_close(encoded[3],old[4])
    proprio=torch.randn(n,230);prev=torch.randn(n,12);dt=torch.full((n,1),.01)
    ages=torch.tensor([[.05,0.],[.05,0.]]);motion=torch.zeros(n,3)
    raw,belief=model.tick(proprio,prev,dt,ages,yes,motion,no,encoded[0],hb)
    actor=nn.Sequential(nn.Linear(235,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)).requires_grad_(False)
    teacher_state=torch.load(teacher,map_location='cpu',weights_only=False)['model_state_dict']
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in teacher_state.items() if k.startswith('actor.')})
    base=torch.randn(n,203);target=torch.randn(n,32,requires_grad=True);gate=torch.full((n,1),.37,requires_grad=True)
    loss,act=action_rows(actor,base,raw,target,gate)
    torch.testing.assert_close(act,actor(torch.cat((base,raw*gate.detach()),-1)))
    capture_target=torch.randn(n,32);labels=torch.tensor([1,2]);window=WindowLoss('prep4')
    window.action.add(loss,labels);window.raw.add((encoded[1]-capture_target).square().mean(-1),labels)
    window.preserve((act-.1).square().mean(-1),labels,yes)
    clean=torch.cat((torch.zeros(n,187),torch.ones(n,187),torch.zeros(n,187),torch.ones(n,187)),-1)
    terrain=torch.cat((torch.zeros(n,187),torch.ones(n,187)),-1)
    window.spatial(encoded[2],clean,terrain);total,_=window.value(raw.new_zeros(()));total.backward()
    assert gate.grad is None and target.grad is None and all(p.grad is None for p in actor.parameters())
    for prefix in ('belief','head','capture_head','visual.cell_gru','visual.spatial_head'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for k,p in model.named_parameters() if k.startswith(prefix)),prefix
    assert all(p.grad is None for k,p in model.named_parameters() if k.startswith(('visual.quality','visual.cnn','visual.geometry','visual.query','visual.key','visual.value','visual.norm','visual.ff')))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        # Stale context cannot enter tick, body still updates belief.
        stale=ages.clone();stale[:,0]=.3
        a=model.tick(proprio,prev,dt,stale,yes,motion,no,encoded[0],hb)
        b=model.tick(proprio,prev,dt,stale,yes,motion,no,torch.randn_like(hv),hb)
        torch.testing.assert_close(a[0],b[0]);assert a[1].abs().sum()>0
        held=model.encode(frames,pose,age,no,no,encoded[0])[0];torch.testing.assert_close(held,encoded[0],atol=0,rtol=0)
        reset=model.encode(frames,pose,age,no,yes,encoded[0])[0];assert reset.count_nonzero()==0
    enames=['frames','relative_pose','packet_age','accepted','reset','h_visual'];eouts=['h_visual_out','raw_capture','spatial_capture','quality']
    tnames=['proprio','previous_action','dt','camera_age','camera_fresh','relative_motion','reset','h_visual','h_belief'];touts=['latent_raw','h_belief_out']
    examples=[(frames,pose,age,yes,no,hv),(proprio,prev,dt,ages,yes,motion,no,hv,hb)]
    sessions=[]
    for tag,graph,example,names,outputs in zip(('encode','tick'),(EncodeGraph(model),TickGraph(model)),examples,(enames,tnames),(eouts,touts)):
        path=out/f'{size}_{tag}.onnx'
        torch.onnx.export(graph,example,str(path),opset_version=18,dynamo=False,input_names=names,output_names=outputs,dynamic_axes={k:{0:'batch'} for k in names+outputs})
        onnx.checker.check_model(str(path));opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        sessions.append(ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']))
    error=0.;timings={1:{'encode':[],'tick':[]},2:{'encode':[],'tick':[]}}
    with torch.no_grad():
      for batch in (1,2):
        th,tb=model.init_hidden(batch,'cpu');oh,ob=th.numpy().copy(),tb.numpy().copy()
        for step in range(80):
            reset=torch.zeros(batch,dtype=torch.bool)
            if step in (0,37):reset[0]=True
            accept=torch.full((batch,),step%9==0,dtype=torch.bool)
            fr=frames[:batch];po=pose[:batch].clone();po[:,0]=.02*np.sin(step);po[:,2]=.05*np.cos(step);po[:,3]=torch.cos(po[:,2]/2);po[:,6]=torch.sin(po[:,2]/2);pa=torch.full((batch,1),.01*(step%16))
            # Explicit reset encode even when there is no new frame.
            ea=(fr,po,pa,accept,reset,th)
            ev=model.encode(*ea)
            start=time.perf_counter_ns();ov=sessions[0].run(None,{k:v.numpy() for k,v in zip(enames,ea[:-1])}|{'h_visual':oh});timings[batch]['encode'].append((time.perf_counter_ns()-start)/1e6)
            for x,y in zip(ev,ov):np.testing.assert_allclose(x.numpy(),y,atol=2e-5,rtol=2e-4);error=max(error,float(np.max(np.abs(x.numpy()-y))))
            th=ev[0];oh=ov[0];ag=torch.full((batch,2),.01*(step%40));fresh=ag[:,0]<.3
            ta=(proprio[:batch],prev[:batch],dt[:batch],ag,fresh,motion[:batch],reset,th,tb)
            tv=model.tick(*ta)
            inp={k:v.numpy() for k,v in zip(tnames,ta)};inp['h_visual']=oh;inp['h_belief']=ob
            start=time.perf_counter_ns();ov=sessions[1].run(None,inp);timings[batch]['tick'].append((time.perf_counter_ns()-start)/1e6)
            for x,y in zip(tv,ov):np.testing.assert_allclose(x.numpy(),y,atol=2e-5,rtol=2e-4);error=max(error,float(np.max(np.abs(x.numpy()-y))))
            tb=tv[1];ob=ov[1]
    bank=BeliefBank(model,2,'cpu')
    ids,_,_=bank.publish(frames,pose,0,5,bank.episode.clone(),5);assert len(ids)==2
    state=bank.hidden.clone()
    # Rejected nonfinite and negative-delay packets never change memory/stamps.
    invalid=frames.clone();invalid[0,0,0,0,0]=float('nan')
    assert not bank.valid(6,7,bank.episode,7,invalid,pose)[0]
    assert not bank.valid(8,7,bank.episode,8,frames,pose).any()
    assert not bank.valid(-1,7,bank.episode,8,frames,pose).any()
    assert not bank.valid(-1,8,bank.episode,8,frames,pose).any()
    # Relative odometry gives the same cell memory as legacy absolute pose warping.
    with torch.no_grad():
        old_pose=pose.clone();old_pose[:,0]=100;old_pose[:,1]=-30;old_pose[:,2]=.7
        old_pose[:,3]=torch.cos(old_pose[:,2]/2);old_pose[:,6]=torch.sin(old_pose[:,2]/2)
        new_pose=old_pose.clone();new_pose[:,0]+=.03;new_pose[:,2]+=.04
        new_pose[:,3]=torch.cos(new_pose[:,2]/2);new_pose[:,6]=torch.sin(new_pose[:,2]/2)
        absolute=visual.encode(frames,hv,old_pose,.05,torch.ones(n))[1]
        expected=visual.encode(frames,absolute,new_pose,.05,torch.ones(n))[1]
        local=model.encode(frames,relative_pose(old_pose,old_pose),age,yes,no,hv)[0]
        actual=model.encode(frames,relative_pose(new_pose,old_pose),age,yes,no,local)[0]
        torch.testing.assert_close(actual[:,:-4],expected[:,:-4],atol=2e-5,rtol=2e-4)

    assert len(bank.publish(frames,pose,0,6,bank.episode.clone(),6)[0])==0
    assert len(bank.publish(frames,pose,7,8,bank.episode.clone(),7)[0])==0
    assert not bank.inputs(30)[1].any();assert bank.inputs(29)[1].all()
    epoch=bank.episode.clone();bank.reset(torch.tensor([True,False]),31)
    assert bank.hidden[0].count_nonzero()==0;torch.testing.assert_close(bank.hidden[1],state[1])
    assert len(bank.publish(frames,pose,32,33,epoch,33)[0])==1
    bank.tick(proprio,pose,33)
    for bad in (33,32):
        try:bank.tick(proprio,pose,bad)
        except ValueError:pass
        else:raise AssertionError('clock accepted')
    report['models'][size]={'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'belief_dim':model.belief_dim,'max_abs_parity_error':error,'sequence_ticks':80,'batches':[1,2],'latency_ms_ort_cpu':{str(batch):{k:{f'p{p}':float(np.percentile(v[10:],p)) for p in (50,95,99)} for k,v in row.items()} for batch,row in timings.items()},'tests':'PASS'}
report['checks']=['exact_34624_visual_weights','v1_visual_output_parity','actual_GAST2000_actor_frozen_gradient','quality_trunk_frozen','belief_and_heads_gradient','gate_once_same_detached_gate','capture_and_execution_separate_targets','reset_partial','stale_300ms','hold_no_frame','duplicate_reorder_future_old_epoch_rejection','clock_duplicate_reversal_halt','onnx_recursive_80_ticks_batch1_batch2_relative_motion']
report['status']='PASS';(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
