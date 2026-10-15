"""CPU integration: real student/autograd/transport, synthetic environment and clock."""
import os,sys,json,types,unittest
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from torch import nn
from gd_lab.gast.student import GastStudent
from gd_lab.core.camera_transport import CameraTransport,CameraTransportConfig
import b2_runner
N=types.SimpleNamespace

class Env:
    def __init__(self):
        self.unwrapped=self;self.num_envs=3;self.step_dt=.02;self.device='cpu';self.common_step_counter=0
        term=N(vel_command_b=torch.zeros(3,3),_resample_command=lambda ids:None,_update_command=lambda:None)
        self.command_manager=N(get_term=lambda name:term)
        self.images=torch.rand(3,4,2,45,80)
    def obs(self):return {'time':self.common_step_counter*.02}
    def reset(self):return self.obs(),{}
    def step(self,action):
        assert not action.requires_grad and torch.isfinite(action).all()
        self.common_step_counter+=1
        self._vrl_camera_snapshot=(self.images,)
        done=torch.tensor([self.common_step_counter in (41,92),False,False])
        return self.obs(),None,done,{}

class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.actor=nn.Linear(34,12);self.encoder=nn.Linear(1,32)
    def _actor_input(self,obs,inference=True):
        raw=self.encoder(torch.full((3,1),obs['time']))
        return torch.cat((torch.ones(3,2),raw),-1)
    def act_inference(self,obs):return self.actor(self._actor_input(obs))

class LoopTest(unittest.TestCase):
    def test_budget_boundaries(self):
        self.assertIsNone(b2_runner.budget_reason(5999,5399.9))
        self.assertEqual(b2_runner.budget_reason(6000,1),'capture_limit')
        self.assertEqual(b2_runner.budget_reason(1,5400),'time_limit')
    def test_real_student_multiple_windows_and_normal_save(self):
        torch.set_num_threads(1);torch.manual_seed(31)
        source=ROOT/'checkpoints/students/control_seen0_perception_34624.pt'
        teacher_path=ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
        for mode in ('memory_head','all_but_quality'):
            env=Env();teacher=Teacher();student=GastStudent('vendor_new')
            student.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['model'])
            clean=torch.cat((torch.zeros(3,187),torch.ones(3,187),torch.zeros(3,187),torch.ones(3,187)),-1)
            pose=torch.tensor([[0.,0,0,1,0,0,0]]).repeat(3,1)
            def capture(obs,action):
                return (None,clean,clean[:,:374],pose,torch.full((3,),obs['time']),None,teacher._actor_input(obs)[:,-32:])
            out=ROOT/'outputs/update31_cpu'/('loop_'+mode)
            # Each run gets a fresh child; previous evidence is preserved.
            index=0
            while out.exists():
                index+=1;out=ROOT/'outputs/update31_cpu'/('loop_'+mode+'_'+str(index))
            args=N(b2_trainable=mode,b2_group_weight='prep4',seed=42,student_resume=str(source),b2_out=str(out),b2_synthetic_cpu_smoke=True)
            transport=CameraTransport(CameraTransportConfig((70,100),(0,150),.05),.02,42)
            clock=N(monotonic=lambda:5401. if env.common_step_counter>=180 else env.common_step_counter*.02)
            terrain=lambda base:N(labels=lambda obs:torch.tensor([0,1,2]))
            with patch.object(b2_runner,'TerrainGroups',terrain),patch.object(b2_runner,'time',clock):
                b2_runner.run(env,teacher,student,N(capture=capture),transport,args,N(seed=42),None,teacher_path,34624)
            summary=json.loads((out/'complete.json').read_text())
            self.assertEqual(summary['reason'],'time_limit');self.assertGreaterEqual(summary['updates'],3)
            self.assertGreaterEqual(summary['captures'],32);self.assertFalse(transport.pending)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in student.quality.parameters()))
            windows=[json.loads(line) for line in (out/'windows.jsonl').read_text().splitlines()]
            self.assertTrue(any(w['action_n']['prep']>0 for w in windows))
            checkpoint=torch.load(next(out.glob('perception_*.pt')),map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['iteration'],34624+summary['captures'])
            self.assertTrue(checkpoint['update31']['synthetic_cpu_smoke'])
            print(mode,summary,flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
