"""CPU-only E1 invariants and mock closed-loop adapter; never imports Isaac."""
import os,sys,unittest,json,types,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import torch,numpy as np
from e1_core import LatentSwitch,action_at_boundary,CONDITIONS
from e1_metrics import summarize_episode
import e1_heldout_runner as e1_runner
import tempfile
class Obs(dict):
    def __init__(self,value):
        super().__init__(gast_clean=torch.zeros(2,748),policy=torch.zeros(2,230));self.value=value
    def __getitem__(self,k):return self.value[k] if isinstance(k,tuple) else super().__getitem__(k)
    def __len__(self):return 2
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('camera_transport',ROOT/'src/gd_lab/core/camera_transport.py')
transport_mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=transport_mod;spec.loader.exec_module(transport_mod)

class Teacher:
    def __init__(self):self.actor=torch.nn.Linear(33,12,bias=False);self.steps=[]
    def terrain_latent(self,obs):self.steps.append(int(obs[0,0]));return torch.ones(len(obs),32)*2
    def act_with_terrain_latent(self,obs,l):return self.actor(torch.cat([obs[:,:1],l],-1))
    def _actor_input(self,obs,inference=True):return torch.cat((obs[:,:1],self.terrain_latent(obs)),-1)
    def act_inference(self,obs):return self.act_with_terrain_latent(obs,self.terrain_latent(obs))

class Tests(unittest.TestCase):
    def test_switch_and_gate(self):
        z=torch.ones(2,32);r=torch.ones(2,dtype=torch.bool);g=torch.full((2,1),.2);s=LatentSwitch(2,'cpu')
        common=(z,z*.2,g,r,torch.tensor([.299,.3]),z*2)
        a,q=s.select('a_student_gate',*common,torch.ones(2)*2,0)
        self.assertTrue(torch.equal(a[0],z[0]*.2));self.assertTrue(torch.equal(a[1],z[1]*0))
        b,_=s.select('b_student_gate1',*common,torch.ones(2)*2,0);self.assertTrue(torch.equal(b,z))
        f,_=s.select('f_teacher_student_gate',*common,torch.ones(2)*2,0);self.assertTrue(torch.equal(f,z*2*q))
        d,_=s.select('d_teacher_early',*common,torch.tensor([1.001,1.]),7)
        self.assertEqual(s.first_step.tolist(),[-1,7]);self.assertTrue(torch.equal(d[1],z[1]*2))
        s.select('d_teacher_early',*common,torch.tensor([1.,2.]),8)
        self.assertEqual(s.first_step.tolist(),[8,7]);s.reset(torch.tensor([True,False]))
        self.assertEqual(s.first_step.tolist(),[-1,7])
        s=LatentSwitch(2,'cpu');s.select('e_teacher_late',*common,torch.tensor([.301,.3]),9)
        self.assertEqual(s.first_step.tolist(),[-1,9])
    def test_full_teacher_equivalence_no_grad(self):
        teacher=Teacher();obs=torch.ones(2,1,requires_grad=True);z=torch.ones(2,32,requires_grad=True)
        a,_,_=action_at_boundary(teacher,obs,LatentSwitch(2,'cpu'),'c_teacher_full',z,z,torch.ones(2,1),torch.ones(2,dtype=torch.bool),torch.zeros(2),torch.ones(2),0)
        self.assertTrue(torch.equal(a,teacher.act_inference(obs)))
        self.assertFalse(a.requires_grad);self.assertTrue(all(p.grad is None for p in teacher.actor.parameters()))
    def test_episode_metrics_terminal_and_censor(self):
        rows=[]
        for k,x in enumerate(np.linspace(-2,1.1,40)):
            foot=np.array([[x-.3,0,.03],[x-.3,0,.03],[x+.3,0,.03],[x+.3,0,.03]])
            rows.append([k*.1,x,.6,0,0,*foot.flatten(),1,1,1,1,-1])
        r=summarize_episode(rows,0,.1,0,['HR_foot','HL_foot','FR_foot','FL_foot'],0,0,'window_end','a')
        self.assertTrue(r['crossed']);self.assertTrue(r['clean']);self.assertFalse(r['censored'])
        r=summarize_episode(rows[:20],0,.1,0,['HR_foot','HL_foot','FR_foot','FL_foot'],0,0,'window_end','a')
        self.assertTrue(r['censored']);self.assertFalse(r['crossed'])
        rows[25][2]=.1
        r=summarize_episode(rows,0,.1,0,['HR_foot','HL_foot','FR_foot','FL_foot'],0,0,'terminated','a')
        self.assertTrue(r['fall'])
    def test_delivery_epoch_and_order(self):
        mask=transport_mod.delivery_mask(10,torch.tensor([0,0,1]),torch.tensor([1,0,1]),torch.tensor([0,11,9]))
        self.assertEqual(mask.tolist(),[False,False,True])
        cfg=transport_mod.CameraTransportConfig((70,100),(0,150),.05)
        def sequence():
            t=transport_mod.CameraTransport(cfg,.01,42);step=0;out=[]
            for _ in range(100):
                step=t.next_capture(step);t.capture(step,None)
                out.append((step,t.dropped,[(p.capture_step,p.delivery_step) for p in t.receive(step)]))
            return out
        self.assertEqual(sequence(),sequence())
    def test_source_no_optimizer_and_inference_mode(self):
        self.assertNotIn('torch.optim',(Path(e1_runner.__file__).read_text()))
        s=(ROOT/'scripts/gast/train_student_live.py').read_text()
        self.assertIn('optimizer = None if args_cli.replay_out',s)
        self.assertIn('@torch.inference_mode()',Path(e1_runner.__file__).read_text())
    def test_mock_runner_all_conditions(self):
        from types import SimpleNamespace as N
        class Geometry:
            def __init__(self,b):pass
            def for_envs(self,b):return torch.tensor([[[-1.,-.9],[1.5,1.6]]]).repeat(2,1,1),torch.zeros(2,2)
        class Student(torch.nn.Module):
            def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.ones(1))
            def init_hidden(self,n,dev):return torch.zeros(n,5984)
            def head(self,m):return m.mean(1)
            def encode(self,f,h,p,age,available):
                m=h+.01;raw=self.head(m.reshape(-1,187,32));q=torch.ones(len(f),1)*.5*max(0,1-age/.3)
                return raw*q,m,None,None,q
        class Attention:
            def __init__(self,b):self.b=b;self.latent=torch.zeros(2,32);self.ready=torch.zeros(2,dtype=torch.bool);self.stamp=torch.ones(2)*-100
            def reset(self,r):self.latent[r]=0;self.ready[r]=False;self.stamp[r]=-100
            def capture(self,obs,act):return (obs,obs,torch.ones(2,374),torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]).repeat(2,1),torch.ones(2)*self.b.common_step_counter*.01,act)
        class Env:
            def __init__(self):
                self.unwrapped=self;self.num_envs=2;self.device='cpu';self.step_dt=.01;self.common_step_counter=0
                names=['HR_foot','HL_foot','FR_foot','FL_foot'];self.term=N(vel_command_b=torch.zeros(2,3),_resample_command=lambda *a:None,_update_command=lambda *a:None)
                self.command_manager=N(get_term=lambda n:self.term)
                self.robot=N(body_names=names,actuators={},data=N(root_pos_w=torch.zeros(2,3),root_quat_w=torch.tensor([[1.,0,0,0]]).repeat(2,1),body_pos_w=torch.zeros(2,4,3),root_state_w=torch.zeros(2,13),joint_pos=torch.zeros(2,12)))
                class Scene(dict):pass
                self.scene=Scene(robot=self.robot,contact_forces=N(body_names=names,data=N(net_forces_w=torch.ones(2,4,3)*20)))
                self.scene.env_origins=torch.zeros(2,3);self.scene.terrain=N(terrain_levels=torch.zeros(2,dtype=torch.long),terrain_types=torch.zeros(2,dtype=torch.long))
                self.termination_manager=N(time_outs=torch.zeros(2,dtype=torch.bool));self.cfg=N(sim=N(dt=.002),decimation=5)
            def reset(self):return Obs(torch.zeros(2,1)),{}
            def _reset_idx(self,ids):self.robot.data.root_pos_w[ids,0]=0
            def step(self,a):
                assert not a.requires_grad
                self.common_step_counter+=1;self.robot.data.root_pos_w[:,0]+=.1;self.robot.data.root_pos_w[:,2]=.6
                self.robot.data.body_pos_w[:,:,0]=self.robot.data.root_pos_w[:,0,None];self.robot.data.body_pos_w[:,:,2]=.03
                done=torch.tensor([self.common_step_counter==20,False]);
                if done.any():self._reset_idx(done.nonzero().flatten())
                self._vrl_camera_snapshot=(torch.zeros(2,4,2,45,80),)
                return Obs(torch.ones(2,1)*self.common_step_counter),torch.zeros(2),done,{}
            def close(self):pass
        modules={
            'gd_lab.core.camera_transport':transport_mod,
            'gd_lab.mdp.terrain_families':N(family_column_masks=lambda b:{'platform_gap':torch.tensor([0])}),
            'gd_lab.gast.distillation':N(pose_xyyaw=lambda b:torch.zeros(2,3)),
            'gd_lab.mdp.platform_gap_finetune':N(GapTileGeometry=Geometry),
            'gd_lab.mdp.platform_gap_terms':N(boarding_family_mask=lambda b:torch.ones(2,dtype=torch.bool))}
        from unittest.mock import patch
        outdir=Path(tempfile.mkdtemp(prefix='update110_collector_test_'));outdir.mkdir(exist_ok=True)
        initial=[]
        for condition in CONDITIONS:
            os.environ['UPDATE110_HELDOUT_OUT']=str(outdir/condition)
            os.environ['UPDATE114_FULLTICK']='1'
            out=outdir/(condition+'.json')
            if out.exists():out.unlink()
            b=Env();teacher=Teacher();student=Student();att=Attention(b)
            args=N(e1_condition=condition,replay_vx=1.,replay_seconds=.4,replay_out=str(out),student_resume=__file__)
            transport=transport_mod.CameraTransport(transport_mod.CameraTransportConfig((70,100),(0,150),.05),.01,42)
            with patch.dict(sys.modules,modules):e1_runner.run(b,teacher,student,att,transport,args,N(seed=42),N(),__file__,34624)
            manifest=json.loads((outdir/condition/'manifest.json').read_text());self.assertTrue(manifest['complete'])
            events=[e for shard in manifest['shards'] for e in torch.load(outdir/condition/shard['name'],weights_only=True)]
            self.assertTrue({'reset','packet','tick'}<=set(e['kind'] for e in events))
            ticks=[e for e in events if e['kind']=='tick']
            self.assertEqual(len(ticks),40)
            self.assertEqual(sum(e['score'] for e in ticks),8)
            self.assertTrue(all(e['proprio'].shape==(2,230) for e in ticks))
            self.assertTrue(torch.equal(ticks[20]['previous_action'][0],torch.zeros(12)))
            self.assertTrue(any(e['kind']=='reset' and e['episodes'][0]==1 for e in events))
            d=json.loads(out.read_text());initial.append(d['initial_state_sha256'])
            self.assertEqual(set(range(40))-set(teacher.steps),set())
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in student.parameters()))
            self.assertTrue(any(r['termination']=='terminated' for r in d['approaches']))
            terminal=next(r for r in d['approaches'] if r['termination']=='terminated')
            self.assertAlmostEqual(terminal['last_recorded_base_x'],2.,places=5)
            self.assertEqual(d['max_teacher_action_equivalence_error'],0)
        self.assertEqual(len(set(initial)),1)

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
