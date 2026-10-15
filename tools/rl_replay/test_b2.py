"""CPU-only autograd/transport tests using the real student and frozen actor weights."""
import os,sys,json,unittest,types
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from torch import nn
from gd_lab.gast.student import GastStudent
from gd_lab.gast.geometry import reconstruction_loss
from gd_lab.gast.b2_objective import configure_student,groups,GroupMean,WindowLoss,action_rows,PacketBank
from b2_runner import save_checkpoint,mixed_velocity

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source=torch.load(ROOT/'checkpoints/students/control_seen0_perception_34624.pt',map_location='cpu',weights_only=False)
        cls.teacher=ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
        state=torch.load(cls.teacher,map_location='cpu',weights_only=False)['model_state_dict']
        weights={k[len('actor.'):]:v for k,v in state.items() if k.startswith('actor.')}
        width=weights['0.weight'].shape[1];cls.base_width=width-32
        cls.actor=nn.Sequential(nn.Linear(width,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12))
        cls.actor.load_state_dict(weights);cls.actor.requires_grad_(False);cls.actor.eval()
    def student(self,mode):
        s=GastStudent('vendor_new');s.load_state_dict(self.source['model']);params=configure_student(s,mode)
        return s,params
    def test_all_four_options_gradient_and_checkpoint(self):
        for mode in ['memory_head','all_but_quality']:
            for weighting in ['equal3','prep4']:
                torch.manual_seed(31);s,params=self.student(mode);opt=torch.optim.Adam(params,lr=1e-4)
                before={n:p.detach().clone() for n,p in s.named_parameters() if not p.requires_grad}
                bank=PacketBank(s,3,'cpu');w=WindowLoss(weighting)
                frames=torch.rand(3,4,2,45,80);pose=torch.tensor([[0.,0,0,1,0,0,0]]).repeat(3,1)
                teacher_capture=torch.randn(3,32)*.2;teacher_current=teacher_capture+.4
                clean=torch.cat((torch.zeros(3,187),torch.ones(3,187),torch.zeros(3,187),torch.ones(3,187)),-1)
                terrain=clean[:,:374];labels=torch.tensor([0,1,2])
                for step in range(2):
                    _,memory,spatial,_,gate=s.encode(frames,bank.hidden,pose,.1,torch.ones(3))
                    raw=s.head(memory[:,:5984].reshape(-1,187,32))
                    if gate.requires_grad:gate.retain_grad()
                    bank.publish(torch.arange(3),memory,raw,gate,torch.ones(3)*step*.1,step)
                    w.raw.add((raw-teacher_capture).square().mean(-1),labels)
                    w.spatial(spatial,clean,terrain)
                    base=torch.randn(3,self.base_width);effective=bank.effective(step*.1+.1)
                    error,actual=action_rows(self.actor,base,bank.raw,teacher_current,effective)
                    w.action.add(error,labels)
                    with torch.no_grad():expected=self.actor(torch.cat((base,raw*gate.detach()),-1))
                    self.assertTrue(torch.equal(actual,expected))
                loss,counts=w.value(torch.tensor(0.));loss.backward()
                self.assertIsNone(gate.grad);self.assertTrue(all(p.grad is None for p in self.actor.parameters()))
                for prefix in ['cell_gru','temporal_attention','age_embed','head']:
                    self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in s.named_parameters() if n.startswith(prefix) and p.grad is not None),0)
                for name,p in s.named_parameters():
                    if not p.requires_grad:self.assertIsNone(p.grad,name)
                if mode=='all_but_quality':
                    for prefix in ['cnn','query','spatial_head']:
                        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in s.named_parameters() if n.startswith(prefix) and p.grad is not None),0)
                self.assertEqual(counts['raw_n'],{'flat':2,'prep':2,'other':2})
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step();bank.detach()
                for n,v in before.items():self.assertTrue(torch.equal(v,dict(s.named_parameters())[n]))
                self.assertFalse(bank.raw.requires_grad);self.assertFalse(bank.hidden.requires_grad)
                out=ROOT/'outputs/update31_cpu';out.mkdir(exist_ok=True)
                args=types.SimpleNamespace(b2_trainable=mode,b2_group_weight=weighting,seed=42,b2_synthetic_cpu_smoke=True)
                save_checkpoint(s,opt,self.source,args,self.teacher,34626,2,1,0.,out/f'{mode}_{weighting}.pt')
    def test_group_empty_and_weights(self):
        for mode,expected in [('equal3',8/3),('prep4',2.2)]:
            g=GroupMean(mode);v=torch.tensor([1.,2.,5.],requires_grad=True);g.add(v,torch.arange(3))
            self.assertAlmostEqual(float(g.value(v.sum()*0)),expected,places=5)
        g=GroupMean('prep4');v=torch.tensor([3.],requires_grad=True);g.add(v,torch.tensor([0]));self.assertEqual(float(g.value(v.sum()*0)),3.)
        empty=GroupMean('equal3');self.assertEqual(float(empty.value(torch.tensor(0.))),0.)
        labels=groups(torch.tensor([.3,1.,1.001,float('inf')]),torch.tensor([True,False,True,False]),torch.tensor([True,True,True,False]))
        self.assertEqual(labels.tolist(),[1,1,0,2])
    def test_capture_vs_execution_target_and_detached_gate(self):
        actor=nn.Linear(3,1,bias=False);actor.weight.data.fill_(1);actor.requires_grad_(False)
        raw=torch.tensor([[1.,2.]],requires_grad=True);base=torch.tensor([[4.]],requires_grad=True)
        gate=torch.tensor([[.2]],requires_grad=True);teacher_now=torch.tensor([[3.,4.]],requires_grad=True)
        values,actual=action_rows(actor,base,raw,teacher_now,gate)
        self.assertAlmostEqual(float(actual),4.6,places=5);self.assertAlmostEqual(float(values),.64,places=5)
        values.sum().backward();self.assertIsNone(gate.grad);self.assertIsNone(base.grad);self.assertIsNone(teacher_now.grad)
        capture=torch.tensor([[1.,2.]]);self.assertEqual(float((raw.detach()-capture).square().mean()),0.)
    def test_reset_reorder_and_held_graph_lifetime(self):
        s,_=self.student('memory_head');b=PacketBank(s,2,'cpu');old=torch.ones(2,32,requires_grad=True)
        b.publish(torch.arange(2),b.hidden,old,torch.ones(2,1)*.5,torch.zeros(2),1)
        self.assertEqual(b.valid(1,torch.zeros(2,dtype=torch.long)).tolist(),[False,False])
        b.reset(torch.tensor([True,False]));self.assertEqual(b.valid(2,torch.zeros(2,dtype=torch.long)).tolist(),[False,True])
        (b.raw.sum()+b.raw.sum()).backward();self.assertTrue(torch.equal(old.grad[0],torch.zeros(32)));self.assertTrue(torch.equal(old.grad[1],torch.ones(32)*2))
        self.assertEqual(b.effective(.3).tolist(),[[0.],[0.]])
        b.detach();self.assertFalse(b.raw.requires_grad)
    def test_spatial_matches_existing_capture_reduction(self):
        pred=torch.randn(2,187,6,requires_grad=True)
        clean=torch.cat((torch.zeros(2,187),torch.ones(2,187),torch.zeros(2,187),torch.ones(2,187)),-1)
        w=WindowLoss('equal3');w.spatial(pred,clean,clean[:,:374]);loss,_=w.value(pred.sum()*0)
        torch.testing.assert_close(loss,reconstruction_loss(pred,clean,clean[:,:374]))
    def test_mixed_velocities_enforced_on_reset(self):
        term=types.SimpleNamespace(vel_command_b=torch.zeros(32,3),_update_command=lambda:None,_resample_command=lambda ids:None)
        v=mixed_velocity(term,32,'cpu');self.assertEqual([int((v==x).sum()) for x in [.6,1.,1.2]],[11,11,10])
        term.vel_command_b.zero_();term._resample_command(None);torch.testing.assert_close(term.vel_command_b[:,0],v)

if __name__=='__main__':unittest.main(verbosity=2)
