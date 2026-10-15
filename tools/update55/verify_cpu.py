"""CPU-only functional/gradient regression; no optimizer steps."""
import copy,inspect,json,os,unittest,tempfile
from pathlib import Path
import torch
from torch import nn
from gd_lab.gast.student_v3 import BeliefStudent
from gd_lab.gast.student_v3_update55 import make_student,make_bank,publish,tick,capture_transform,AsyncTickGraph
from gd_lab.gast.belief_aux import BeliefAux,LookaheadAux,aligned_targets
from gd_lab.gast.v3_sequence import SequenceLoader,SequenceReplay,replay_loss,slice_row
from gd_lab.gast.b2_objective import action_rows

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/update55_cpu';OUT.mkdir(exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1)
def pose(n=2):
    p=torch.zeros(n,7);p[:,3]=1;return p

def future_records(length,n=2):
    return [dict(step=i,epoch=torch.zeros(n,dtype=torch.long),done=torch.zeros(n,dtype=torch.bool),
        executed_action=torch.randn(n,12,requires_grad=True),teacher_execution_raw=torch.randn(n,32,requires_grad=True),
        group=torch.tensor([0,1])[:n]) for i in range(length)]

class Checks(unittest.TestCase):
    def test_default_exact_model_loss_gradient_rng(self):
        for size in ('small','large'):
            torch.manual_seed(55);old=BeliefStudent(size=size)
            torch.manual_seed(55);new=make_student(size=size)
            self.assertEqual(type(new),BeliefStudent)
            for name,p in old.state_dict().items():self.assertTrue(torch.equal(p,new.state_dict()[name]),name)
            rng=torch.random.get_rng_state();aux=LookaheadAux(new.belief_dim)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(aux.parameters(),[])
            hv,hb=old.init_hidden(2,'cpu');hv=torch.randn_like(hv)
            args=(torch.randn(2,230),torch.randn(2,12),torch.full((2,1),.01),torch.zeros(2,2),torch.ones(2,dtype=torch.bool),torch.zeros(2,3),torch.zeros(2,dtype=torch.bool),hv,hb)
            a=old.tick(*args);b=new.tick(*args)
            for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
            la=a[0].square().mean();lb=b[0].square().mean();lb2,_=aux.value(lb);self.assertIs(lb,lb2)
            la.backward();lb2.backward()
            for (name,p),q in zip(old.named_parameters(),new.parameters()):
                self.assertTrue((p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad),name)

    def test_async_event_reset_clock_and_transform(self):
        m=make_student(asynchronous=True);bank=make_bank(m,2,'cpu');p=pose();frames=torch.rand(2,4,2,45,80);z=torch.tensor([.5,.6])
        epoch=bank.episode.clone()
        self.assertEqual(len(publish(bank,frames,p,0,2,epoch,2,z)[0]),2)
        self.assertTrue(bank.new_packet.all());tick(bank,torch.zeros(2,230),p,2,z);self.assertFalse(bank.new_packet.any())
        self.assertEqual(len(publish(bank,frames,p,0,3,epoch,3,z)[0]),0)
        self.assertFalse(bank.new_packet.any())
        self.assertEqual(len(publish(bank,frames,p,3,5,epoch,4,z)[0]),0)
        bank.reset(torch.tensor([True,False]),3);self.assertFalse(bank.ready[0])
        self.assertEqual(publish(bank,frames,p,3,4,epoch,4,z)[0].tolist(),[1])
        self.assertEqual(publish(bank,frames,p,4,4,bank.episode.clone(),4,z)[0].tolist(),[0,1])
        tick(bank,torch.randn(2,230),p,4,z)
        for step in (4,3):
            with self.assertRaises(ValueError):tick(bank,torch.zeros(2,230),p,step,z)
        hv,hb=m.init_hidden(2,'cpu');yes=torch.ones(2,dtype=torch.bool);no=~yes
        args=[torch.randn(2,230),torch.randn(2,12),torch.full((2,1),.01),torch.zeros(2,2),yes,torch.zeros(2,3),no,hv,hb,no,torch.eye(4)[:3].repeat(2,1,1)]
        first=m.tick(*args);args[7]=torch.randn_like(hv);second=m.tick(*args)
        self.assertTrue(torch.equal(first[0],second[0])) # held visual cannot be re-injected
        args[9]=yes;third=m.tick(*args);self.assertFalse(torch.equal(second[0],third[0]))
        args[6]=yes;args[8]=torch.randn_like(hb);a=m.tick(*args);args[8]=hb;b=m.tick(*args)
        self.assertTrue(torch.equal(a[0],b[0]))
        now=pose();capture=pose();capture[:,0]=1;now[:,2]=torch.pi/2;now[:,3]=2**-.5;now[:,6]=2**-.5
        transform=capture_transform(now,torch.ones(2),capture,torch.zeros(2))
        torch.testing.assert_close(transform[:,:,-1],torch.tensor([[0.,-1.,-1.]]).repeat(2,1),atol=1e-6,rtol=0)
        self.assertNotIn('target',str(inspect.signature(AsyncTickGraph.forward)))

    def test_aux_gradient_front_unknown_and_future(self):
        for kind in ('front','future'):
            model=make_student(asynchronous=True)
            frames=torch.rand(2,4,2,45,80);hv,hb=model.init_hidden(2,'cpu');yes=torch.ones(2,dtype=torch.bool);no=~yes
            encoded=model.encode(frames,pose(),torch.zeros(2,1),yes,no,hv)
            raw,h=model.tick(torch.randn(2,230),torch.zeros(2,12),torch.full((2,1),.01),torch.zeros(2,2),yes,torch.zeros(2,3),no,encoded[0],hb,yes,torch.eye(4)[:3].repeat(2,1,1))
            aux=LookaheadAux(model.belief_dim,front_coefficient=.2 if kind=='front' else 0.,future_coefficient=.2 if kind=='future' else 0.)
            rng=torch.random.get_rng_state()
            records=future_records(45)
            if kind=='front':
                terrain=torch.cat((torch.zeros(2,187),torch.ones(2,187)),-1).requires_grad_()
                terrain.data[0,187+12]=0
                target,mask=aligned_targets(terrain,pose(),torch.zeros(2),pose(),torch.zeros(2),yes)
                aux.add(h,target,mask,torch.tensor([0,1]));self.assertFalse(mask[0,12,4])
            else:
                for r in records:aux.add_transition(h,r)
            loss,stats=aux.value(raw.new_zeros(()));loss.backward()
            for prefix in ('belief','visual_projection','visual.cell_gru'):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.named_parameters() if name.startswith(prefix)),prefix)
            for name,p in model.named_parameters():
                if not p.requires_grad or name.startswith(('head.','capture_head.','visual.spatial_head.')):self.assertIsNone(p.grad,name)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in aux.parameters()))
            for r in records:
                self.assertIsNone(r['executed_action'].grad);self.assertIsNone(r['teacher_execution_raw'].grad)
            if kind=='front':self.assertIsNone(terrain.grad)
            else:self.assertEqual(stats['future_valid_pairs'],{'10':70,'20':50,'40':10})

    def test_future_reset_boundary_missing_step_no_leak(self):
        aux=LookaheadAux(8,future_coefficient=1.);records=future_records(45);hs=[torch.randn(2,8,requires_grad=True) for _ in records]
        for i,r in enumerate(records):
            if i>=20:r['epoch'][0]=1
            if i==19:r['done'][0]=True
            aux.add_transition(hs[i],r)
        _,counts=aux.future_loss(torch.tensor(0.));self.assertEqual(counts,{'10':60,'20':30,'40':5})
        records[25]['step']+=1
        _,bad=aux.future_loss(torch.tensor(0.));self.assertLess(bad['10'],counts['10'])
        short=LookaheadAux(8,future_coefficient=1.)
        for h,r in zip(hs[:10],records[:10]):short.add_transition(h,r)
        loss,counts=short.future_loss(torch.tensor(0.));self.assertEqual(float(loss),0.);self.assertEqual(sum(counts.values()),0)
        # Training action/target mutations cannot affect current inference.
        model=make_student();bank=make_bank(model,2,'cpu');p=pose();proprio=torch.randn(2,230)
        a=tick(bank,proprio,p,0,torch.zeros(2));bank2=make_bank(model,2,'cpu')
        records[-1]['teacher_execution_raw']=torch.full((2,32),float('nan'))
        records[-1]['executed_action']=torch.randn(2,12)*1000
        b=tick(bank2,proprio,p,0,torch.zeros(2));self.assertTrue(torch.equal(a,b))

    def test_predict_correct_separate_opt_in(self):
        with self.assertRaises(ValueError):make_student(predict_correct=True)
        model=make_student(asynchronous=True,predict_correct=True)
        hv,hb=model.init_hidden(2,'cpu');yes=torch.ones(2,dtype=torch.bool);no=~yes
        args=[torch.randn(2,230),torch.randn(2,12),torch.full((2,1),.01),torch.zeros(2,2),yes,torch.randn(2,3),no,hv,hb,no,torch.eye(4)[:3].repeat(2,1,1)]
        raw,h=model.tick(*args);raw.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.predictor.parameters()))
        self.assertTrue(all(p.grad is None or p.grad.count_nonzero()==0 for p in model.belief.parameters()))
        model.zero_grad(set_to_none=True);args[9]=yes
        raw,h=model.tick(*args);raw.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.belief.parameters()))

    def test_replay_burn_in_and_protected_buckets(self):
        model=make_student();actor=nn.Linear(235,12).requires_grad_(False)
        records=[]
        for i in range(72):
            r=dict(step=i,epoch=torch.zeros(1,dtype=torch.long),done=torch.zeros(1,dtype=torch.bool),
                proprio=torch.randn(1,230),pose=pose(1),root_z=torch.zeros(1),previous_action=torch.randn(1,12),
                executed_action=torch.randn(1,12),scanner_z=torch.zeros(1),teacher_execution_raw=torch.randn(1,32),actor_base=torch.randn(1,203),baseline_action=torch.randn(1,12),
                group=torch.zeros(1,dtype=torch.long),tick_packets=[],h_visual=torch.full((1,6116),float('nan')),
                h_belief=torch.full((1,256),float('nan')),labels={'phase':['landing' if i==31 else 'flat'],'family':['v3_flat'],'force_world':{}})
            if i%12==0:
                terrain=torch.cat((torch.zeros(1,187),torch.ones(1,187)),-1)
                clean=torch.cat((terrain,torch.zeros(1,187),torch.ones(1,187)),-1)
                extra=(r['actor_base'],clean,terrain,r['pose'],torch.tensor([i*.01]),torch.zeros(1,12),r['teacher_execution_raw'])
                r['tick_packets']=[{'capture_step':i,'delivery_step':i,'payload':(torch.rand(1,4,2,45,80),r['epoch'],extra,r['group'],{'root_z':r['root_z'],'scanner_z':r['scanner_z']})}]
            records.append(r)
        buffer=SequenceReplay(16,16,2)
        for r in records:buffer.append(r)
        self.assertEqual(buffer.coverage()['flat'],2);self.assertEqual(buffer.coverage()['landing'],1)
        seq=buffer.buckets['landing'][0]
        self.assertNotIn('h_visual',seq[0]);self.assertNotIn('h_belief',seq[0])
        seen=[];hook=model.belief.register_forward_hook(lambda m,i,o:seen.append((torch.is_grad_enabled(),i[1].detach().clone())))
        loss,stats=replay_loss(model,actor,seq,16);hook.remove()
        self.assertEqual([x[0] for x in seen],[False]*16+[True]*16)
        self.assertEqual(int(seen[0][1].count_nonzero()),0);self.assertGreater(int(seen[16][1].count_nonzero()),0)
        loss.backward();self.assertTrue(any(p.grad is not None for p in model.belief.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.parameters()));self.assertEqual(stats['preserve_n'],16)
        altered=copy.deepcopy(seq);altered[15]['proprio']+=5
        other,_=replay_loss(model,actor,altered,16);self.assertNotEqual(float(loss),float(other))
        loader=SequenceLoader(16,16,40);self.assertEqual(loader.span,72)
        with self.assertRaises(ValueError):loader.split(records[:34])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sequence.pt';torch.save({'ticks':records},path)
            loaded=list(loader.load(path));self.assertEqual(len(loaded),1)
            aux=LookaheadAux(model.belief_dim,future_coefficient=.1)
            model.zero_grad(set_to_none=True)
            future,detail=replay_loss(model,actor,loaded[0],16,aux,16,aux_only=True)
            self.assertEqual(detail['future_valid_pairs'],{'10':16,'20':16,'40':16})
            future.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.belief.parameters()))
            self.assertTrue(all(p.grad is None for p in actor.parameters()))
        bad=copy.deepcopy(seq);bad[3]['epoch']+=1
        with self.assertRaises(ValueError):replay_loss(model,actor,bad,16)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'device':'cpu','optimizer_steps':0}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
