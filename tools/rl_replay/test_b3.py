"""CPU-only B3 objective, sampler and recurrent-loop contract tests."""
import os,sys,json,types,unittest
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from gd_lab.gast.b3_objective import WindowLoss
from gd_lab.gast.student import GastStudent
from gd_lab.core.camera_transport import CameraTransport,CameraTransportConfig
from test_b2_loop import Env,Teacher
from b3_sampling import ApproachSampler,approach_done
import b3_runner
N=types.SimpleNamespace

class Tests(unittest.TestCase):
    def test_preservation_independent_mean_and_prep_exclusion(self):
        w=WindowLoss();v=torch.tensor([1.,100.,3.],requires_grad=True)
        w.preserve(v,torch.tensor([0,1,2]),torch.ones(3,dtype=torch.bool))
        loss,s=w.value(torch.tensor(0.));self.assertEqual(float(loss),2.);self.assertEqual(s['preserve_n'],2)
        loss.backward();torch.testing.assert_close(v.grad,torch.tensor([.5,0,.5]))
        self.assertEqual(s['action_group_mean'],{'flat':None,'prep':None,'other':None})
        w=WindowLoss();loss,s=w.value(torch.tensor(0.));self.assertEqual(float(loss),0.)
    def test_budget(self):
        self.assertIsNone(b3_runner.budget_reason(1999,10799))
        self.assertEqual(b3_runner.budget_reason(2000,1),'update_limit')
        self.assertEqual(b3_runner.budget_reason(1,10800),'time_limit')
    def test_sampler_partition_resets_and_stall(self):
        terrain=N(terrain_types=torch.zeros(32,dtype=torch.long),terrain_levels=torch.zeros(32,dtype=torch.long),
            env_origins=torch.zeros(32,3),terrain_origins=torch.zeros(1,30,3))
        default=torch.zeros(32,13);default[:,2]=.55;default[:,3]=1
        data=N(default_root_state=default,root_pos_w=default[:,:3].clone())
        poses=[]
        robot=N(data=data,write_root_pose_to_sim=lambda p,env_ids:(poses.append((p.clone(),env_ids.clone())),data.root_pos_w.__setitem__(env_ids,p[:,:3])),
            write_root_velocity_to_sim=lambda v,env_ids:None)
        class Scene:
            def __getitem__(self,k):return robot
        scene=Scene();scene.terrain=terrain;scene.env_origins=terrain.env_origins
        reset_ids=[];env=N(num_envs=32,device='cpu',scene=scene,common_step_counter=0,step_dt=.02,
            _reset_idx=lambda ids:reset_ids.append(ids.clone()))
        columns={'platform_gap':torch.arange(10,30),'pyramid_stairs':torch.arange(6),'random_rough':torch.arange(6,10)}
        slots=torch.tensor([[[-1.6,-1.4],[1.4,1.6]]]).repeat(32,1,1)
        with patch.dict(sys.modules,{'gd_lab.mdp.terrain_families':N(family_column_masks=lambda b:columns)}):
            sampler=ApproachSampler(env,N(for_envs=lambda b:(slots,None)))
        env._reset_idx(torch.arange(32));self.assertEqual(len(poses[0][1]),24)
        self.assertTrue((terrain.terrain_types[:24]>=10).all());self.assertTrue((terrain.terrain_types[24:28]<6).all())
        self.assertTrue((terrain.terrain_types[28:]>=6).all());self.assertEqual(len(reset_ids[0]),32)
        d=1.4-data.root_pos_w[:24,0];self.assertTrue(((d>=2.)&(d<=2.2)).all())
        env.common_step_counter=101;done=approach_done(env)
        self.assertEqual(int(done.sum()),24);self.assertFalse(done[24:].any())
        env._reset_idx(torch.tensor([0,25]));self.assertEqual(poses[-1][1].tolist(),[0])
        data.root_pos_w[0,0]=2.5;self.assertTrue(sampler.done()[0]);sampler.close()
        self.assertFalse(approach_done(env).any())
    def test_real_student_parallel_reference_and_save(self):
        torch.set_num_threads(1);torch.manual_seed(37)
        env=Env();teacher=Teacher();student=GastStudent('vendor_new')
        source=ROOT/'checkpoints/students/control_seen0_perception_34624.pt'
        initial=torch.load(source,map_location='cpu',weights_only=False);student.load_state_dict(initial['model'])
        teacher_path=ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
        clean=torch.cat((torch.zeros(3,187),torch.ones(3,187),torch.zeros(3,187),torch.ones(3,187)),-1)
        pose=torch.tensor([[0.,0,0,1,0,0,0]]).repeat(3,1)
        def capture(obs,action):return (None,clean,clean[:,:374],pose,torch.full((3,),obs['time']),None,teacher._actor_input(obs)[:,-32:])
        out=ROOT/'outputs/cpu_b3_loop';i=0
        while out.exists():i+=1;out=ROOT/f'outputs/cpu_b3_loop_{i}'
        args=N(b2_trainable='memory_head',b2_group_weight='prep4',seed=42,student_resume=str(source),b2_out=str(out),b2_synthetic_cpu_smoke=True)
        clock=N(monotonic=lambda:10801. if env.common_step_counter>=180 else env.common_step_counter*.02)
        terrain=lambda b:N(gap=None,labels=lambda obs:torch.tensor([0,1,2]))
        sampler=lambda *a:N(summary=lambda:{'synthetic':True},close=lambda:None)
        transport=CameraTransport(CameraTransportConfig((70,100),(0,150),.05),.02,42)
        refs=[];original_copy=b3_runner.copy.deepcopy
        def copy_reference(s):
            r=original_copy(s);refs.append(r);return r
        with patch.object(b3_runner,'TerrainGroups',terrain),patch.object(b3_runner,'ApproachSampler',sampler),patch.object(b3_runner,'time',clock),patch.object(b3_runner,'copy',N(deepcopy=copy_reference)):
            b3_runner.run(env,teacher,student,N(capture=capture),transport,args,N(seed=42),None,teacher_path,34624)
        summary=json.loads((out/'complete.json').read_text());self.assertEqual(summary['reason'],'time_limit')
        self.assertGreaterEqual(summary['updates'],3);self.assertFalse(transport.pending)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in refs[0].parameters()))
        for name,v in refs[0].state_dict().items():self.assertTrue(torch.equal(v,initial['model'][name]))
        for name,p in student.named_parameters():
            if not name.startswith(('cell_gru.','temporal_attention.','age_embed.','head.')):
                self.assertIsNone(p.grad);self.assertTrue(torch.equal(p,initial['model'][name]))
        records=[json.loads(x) for x in (out/'windows.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['preserve_n']>0 for r in records));self.assertGreater(records[-1]['preserve'],0.)
        checkpoint=torch.load(next(out.glob('perception*.pt')),map_location='cpu',weights_only=False)
        self.assertEqual(checkpoint['update37']['preservation_coefficient'],1.)
        print('B3 loop PASS',summary,'fixture',str(out),flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
