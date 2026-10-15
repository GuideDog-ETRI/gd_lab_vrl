"""CPU contract, cache, event-host and training gradient regression tests."""
import json,os,copy
from pathlib import Path
import torch
from gd_lab.gast.student_v3_update55 import make_student
from gd_lab.gast.student_v3_update57 import ProductionEncode
from gd_lab.gast.v3_realtime import RealtimeBank,Timing,Packet,LatestEncoder,TrainingHost
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1);torch.manual_seed(57)
OUT=Path('outputs/update57_cpu');OUT.mkdir(exist_ok=True)
report={}
for size in ('small','large'):
 for asynchronous,pc in ((False,False),(True,False),(True,True)):
    s=make_student(size=size,asynchronous=asynchronous,predict_correct=pc)
    a=RealtimeBank(s,2);b=RealtimeBank(s,2,projection_cache=True)
    pose=torch.zeros(2,7);pose[:,3]=1;z=torch.zeros(2)
    with torch.no_grad():
      for step in range(90):
        now=step*10
        if step in (0,33,60):
            done=torch.tensor([True,step==60]);a.reset(done,now);b.reset(done,now)
        if step in (0,7,33,89):
            packet=Packet(step,now,now,now,torch.rand(2,4,2,45,80),pose,z,a.bank.episode.clone())
            x=a.complete(packet,now);y=b.complete(packet,now)
            for u,v in zip(x,y):
                if u is not None:assert torch.equal(u,v)
        proprio=torch.randn(2,230)
        x=a.tick(proprio,pose,z,now);y=b.tick(proprio,pose,z,now)
        assert torch.equal(x,y),(size,asynchronous,pc,step)
        assert torch.equal(a.bank.belief,b.bank.belief)
        if step==88:assert not a.bank.inputs(now/10)[1].any()
    report[f'{size}_{asynchronous}_{pc}']='90 ticks batch2 cache bit equal, stale/reset/epoch'
# Gradient accumulation across held projection and weight-change invalidation.
s=make_student();t=copy.deepcopy(s);a=RealtimeBank(s,1);b=RealtimeBank(t,1,projection_cache=True)
pose=torch.zeros(1,7);pose[:,3]=1;z=torch.zeros(1)
p=Packet(0,0,0,0,torch.rand(1,4,2,45,80),pose,z,torch.zeros(1,dtype=torch.long))
a.complete(p,0);b.complete(p,0)
la=lb=0
for now in (0,9,21):
    proprio=torch.randn(1,230);la=la+a.tick(proprio,pose,z,now).square().sum();lb=lb+b.tick(proprio,pose,z,now).square().sum()
la.backward();lb.backward()
for (name,x),(_,y) in zip(s.named_parameters(),t.named_parameters()):
    if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)
with torch.no_grad():t.visual_projection[0].bias.add_(.1)
b.detach(weights_changed=True)
assert torch.equal(b.cache.value,t.visual_projection(b.bank.hidden[:,:5984]))
report['cache_gradient_and_weight_refresh']='PASS (gradient tolerance, not bit identity for reordered sums)'
# Backpressure: finish packet 1 even if newer work arrived; waiting 2 replaced by 3.
q=LatestEncoder(50);seen=[]
for seq,now in ((1,0),(2,10),(3,20)):
    q.submit(Packet(seq,now,now,now,p.frames,pose,z,p.epochs),now);q.poll(now,lambda p,t:seen.append((p.sequence,t)))
q.poll(50,lambda p,t:seen.append((p.sequence,t)));q.poll(100,lambda p,t:seen.append((p.sequence,t)))
assert seen==[(1,50),(3,100)] and q.dropped==1 and q.max_occupancy<=2
assert not q.submit(Packet(2,100,100,100,p.frames,pose,z,p.epochs),100)
report['bounded_queue']='PASS in-flight1 + latest1, drop/reorder, no starvation'
# A late old-epoch completion cannot restore state/cache.
b.reset(torch.ones(1,dtype=torch.bool),101)
assert len(b.complete(p,120)[0])==0 and not b.cache.valid.any()
assert len(b.complete(Packet(100,200,120,200,p.frames,pose,z,b.bank.episode.clone()),120)[0])==0
report['inflight_reset_future_timestamp']='PASS'
# Fixed physics host: actual applied previous action, jitter, proprio age, loss backward.
for delay in (0,1,2):
    records=[];physical=[];student=make_student()
    host=TrainingHost(student,1,Timing(jitter_ms=2,long_probability=.15,long_interval_ms=30,action_delay_ticks=delay,proprio_delay_ms=7,encode_ms=25),True,clip_actions=1.)
    def observe():return torch.full((1,230),host.now_ms/1000),pose,z
    def advance(dt,action):physical.append((host.now_ms,dt,action.clone()));return None
    def actor(raw,gate,proprio):return raw[:,:12]+.01
    host.run_until(400,advance=advance,observe=observe,capture=lambda:(p.frames,None),actor=actor,on_tick=records.append,capture_period_ms=10)
    assert len(physical)==400 and all(dt==.001 for _,dt,_ in physical)
    for i,r in enumerate(records):
        expected=torch.zeros(1,12) if i==0 else records[i-1]['applied_action']
        assert torch.equal(r['previous_action'],expected)
        expected=torch.zeros(1,12) if i<delay else records[i-delay]['action'].detach().clamp(-1,1)
        assert torch.equal(r['applied_action'],expected)
        assert r['proprio_sample_ms']==max(0,r['now_ms']-7)
    assert any(r['dt_ms']==30 for r in records)
    assert all(abs(r['release_jitter_ms'])<=2 for r in records)
    assert any(r['dt_ms'] in (8,9,11,12) for r in records)
    loss=sum(r['raw'].square().mean() for r in records);loss.backward()
    assert student.belief.weight_ih.grad.abs().sum()>0
    assert host.queue.max_occupancy<=2 and host.queue.dropped>0
report['training_host']='PASS 1ms physics, release jitter, rare30ms, delays0/1/2, applied feedback, backward'
# Long vision delay + burst loss are independent of proprio, original timestamps retained.
student=make_student();records=[];packets=[]
host=TrainingHost(student,1,Timing(long_vision_probability=1.,long_vision_ms=1200,proprio_delay_ms=5,timestamp_offset_ms=-3,odometry_std_m=.01),True)
with torch.no_grad():host.run_until(1401,advance=lambda dt,a:None,observe=lambda:(torch.zeros(1,230),pose,z),capture=lambda:(p.frames,None),actor=lambda raw,g,obs:raw[:,:12],on_tick=records.append,on_encode=lambda p,r:packets.append(p))
assert packets and packets[0].capture_ms==0 and packets[0].arrival_ms==1200 and packets[0].observed_capture_ms==-3
assert all(r['gate'].eq(0).all() for r in records)
assert not torch.equal(packets[0].pose,pose)
host=TrainingHost(student,1,Timing(burst_probability=1.,burst_packets=3))
with torch.no_grad():host.run_until(241,advance=lambda dt,a:None,observe=lambda:(torch.zeros(1,230),pose,z),capture=lambda:(p.frames,None),actor=lambda raw,g,obs:raw[:,:12])
assert host.dropped==4
report['long_delay_burst_sensor_separation']='PASS >1s stale, original/observed timestamp split, independent odometry/proprio'
report['status']='PASS'
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
