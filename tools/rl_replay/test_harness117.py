"""CPU-only contracts and complete rollout loop with simulator stubs."""
import ast,sys,types,tempfile,json,os,unittest,importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from heightmap116 import rays,rotation
from e1_heightmap117 import Diagnostics,StairRecorder,run
ROOT=Path(__file__).resolve().parents[2]
N=types.SimpleNamespace
_read_bytes=Path.read_bytes

def extracted(path,names,ns):
    tree=ast.parse((ROOT/path).read_text());nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)

class Contracts(unittest.TestCase):
    def test_camera_ros_to_opengl(self):
        rng=np.random.default_rng(116)
        for _ in range(12):
            q=rng.normal(size=4);q/=np.linalg.norm(q);gl=np.array([-q[1],q[0],q[3],-q[2]])
            k=np.array([[45.,0,40],[0,45,22.5],[0,0,1.]])
            v,u=np.mgrid[:45,:80];ros=np.stack(((u-40)/45,(v-22.5)/45,np.ones((45,80))),-1)@rotation(q).T
            np.testing.assert_allclose(rays(k,gl,45,80),ros,atol=1e-12)
    def test_cell_metric_denominators(self):
        d=Diagnostics();gt=np.zeros((1,748));gt[:,187:374]=1;pred=np.zeros((1,374));pred[:,187:]=1
        pred[0,0]=5;d.cells(pred,gt);r=d.result()['metrics'];self.assertAlmostEqual(r['false_gap']['mean'],1/187);self.assertEqual(r['gap_recall']['n'],0)
    def test_stair_pre_reset_and_window_metrics(self):
        class Scene(dict):pass
        data=N(root_pos_w=torch.tensor([[0.,0,.5],[0.,0,.5]]),root_quat_w=torch.tensor([[1.,0,0,0],[1.,0,0,0]]))
        force=torch.zeros(2,1,3);scene=Scene(robot=N(data=data),contact_forces=N(body_names=['trunk'],data=N(net_forces_w=force)))
        scene.env_origins=torch.zeros(2,3)
        sub=N(platform_width=1.,border_width=.25)
        base=N(num_envs=2,scene=scene,termination_manager=N(time_outs=torch.zeros(2,dtype=torch.bool)),cfg=N(scene=N(terrain=N(terrain_generator=N(size=(8.,8.),sub_terrains={'pyramid_stairs':sub})))),_reset_idx=lambda *args:None)
        with patch.dict(os.environ,U116_FAMILY='pyramid_stairs'):
            rec=StairRecorder(base,lambda:torch.ones(2,dtype=torch.bool));rec.sample()
            data.root_pos_w[0,0]=4.;data.root_pos_w[1,0]=1.;force[1,0,2]=20;rec.sample()
            base._reset_idx(torch.tensor([1]));rows=rec.finish();rec.close()
        rows={r['env_id']:r for r in rows};self.assertTrue(rows[0]['passed']);self.assertFalse(rows[0]['body_contact']);self.assertTrue(rows[1]['body_contact']);self.assertEqual(rows[1]['termination'],'terminated');self.assertFalse(rows[1]['passed'])
    def test_full_loop_delivery_reset_and_frozen_arms(self):
        transport_ns={};exec((ROOT/'src/gd_lab/core/camera_transport.py').read_text(),transport_ns)
        obs_ns={'torch':torch,'HISTORY_OFFSETS':(48,32,16,8,4,2,1,0)}
        geom_ns={'torch':torch};exec((ROOT/'src/gd_lab/gast/geometry.py').read_text(),geom_ns);obs_ns['warp_memory']=geom_ns['warp_memory']
        extracted('src/gd_lab/gast/observations.py',['TerrainHistory','pose_xyyaw'],obs_ns)
        history_cls=obs_ns['TerrainHistory'];self.assertEqual(history_cls.NEVER,-10**9)
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('c','h'):
                n=4;data=N(root_pos_w=torch.tensor([[0.,0,.8]]).repeat(n,1),root_quat_w=torch.tensor([[1.,0,0,0]]).repeat(n,1),root_state_w=torch.zeros(n,13),joint_pos=torch.zeros(n,12))
                robot=N(data=data,actuators={});scene={'robot':robot,'height_scanner':N(data=N(pos_w=data.root_pos_w.clone()))}
                class Scene(dict):pass
                scene=Scene(scene);scene.env_origins=torch.zeros(n,3);scene.terrain=N(terrain_levels=torch.zeros(n,dtype=torch.long),terrain_types=torch.zeros(n,dtype=torch.long))
                term=N(vel_command_b=torch.zeros(n,3),_resample_command=lambda *a:None,_update_command=lambda *a:None,cfg=N(pulse_prob=0))
                b=N(num_envs=n,device='cpu',step_dt=.01,common_step_counter=0,scene=scene,command_manager=N(get_term=lambda name:term),cfg=N(sim=N(dt=.005),decimation=2))
                class Obs(dict):
                    @property
                    def batch_size(self):return [n]
                    def copy(self):return Obs(self)
                obs=Obs(policy=torch.zeros(n,48),gast_history=torch.zeros(n,3000))
                teacher=N(terrain_latent=lambda o:torch.ones(n,32),act_with_terrain_latent=lambda o,z:z[:,:12],act_inference=lambda o:torch.ones(n,12))
                student=N(eval=lambda:None,parameters=lambda:[],init_hidden=lambda n,d:torch.zeros(n,8),encode=lambda f,h,p,a,r:(torch.ones(len(f),32)*.2,h,None,None,torch.ones(len(f),1)*.2))
                attention=N(latent=torch.zeros(n,32),ready=torch.zeros(n,dtype=torch.bool),stamp=torch.full((n,),-100.))
                def reset_att(mask):attention.latent[mask]=0;attention.ready[mask]=False;attention.stamp[mask]=-100
                attention.reset=reset_att
                resets=[];actions=[]
                def step(action):
                    actions.append(action.clone());b.common_step_counter+=1
                    frames=torch.ones(n,4,2,45,80)*.7;depth=torch.ones(n,4,45,80)*.8
                    pos=torch.tensor([.5,0,.8]).repeat(n,4,1);ros=torch.tensor([0.,1.,0,0]).repeat(n,4,1);ks=torch.tensor([[45.,0,40],[0,45,22.5],[0,0,1.]]).repeat(n,4,1,1)
                    b._vrl_camera_snapshot=(frames,depth,pos,ros,ks);b._vrl_camera_snapshot_steps=torch.full((n,),b.common_step_counter)
                    done=torch.zeros(n,dtype=torch.bool)
                    if b.common_step_counter==17:done[0]=True;resets.append(17)
                    return obs,None,done,None
                env=N(unwrapped=b,num_envs=n,reset=lambda:(obs,None),step=step)
                class Rec:
                    def __init__(self,*a):self.metrics=N(finish=lambda:[])
                    def sample(self,*a):pass
                    def close(self):pass
                Geometry=lambda base:N(for_envs=lambda base:(torch.tensor([[[0.,.2],[1.,1.2]]]).repeat(n,1,1),torch.zeros(n,2)))
                clean=lambda base:torch.cat((torch.ones(n,187)*1.5,torch.ones(n,187),torch.zeros(n,187),torch.ones(n,187)),-1)
                modules={'gd_lab.core.camera_transport':N(**transport_ns),'gd_lab.gast.observations':N(TerrainHistory=history_cls,pose_xyyaw=obs_ns['pose_xyyaw'],clean_terrain=clean),'gd_lab.mdp.terrain_families':N(terrain_family_gate=lambda b,f:torch.zeros(n)),'gd_lab.mdp.platform_gap_finetune':N(GapTileGeometry=Geometry)}
                output=Path(tmp)/(arm+'.json');checkpoint=Path(tmp)/'fake.pt';checkpoint.write_bytes(b'fixture')
                config=transport_ns['CameraTransportConfig']((70,100),(0,150),.05);transport=transport_ns['CameraTransport'](config,.01,7)
                args=N(replay_vx=1.,replay_seconds=.4,e1_condition='a_student_gate',student_resume=str(checkpoint),replay_out=str(output))
                with patch.dict(sys.modules,modules),patch.dict(os.environ,U116_ARM=arm,U116_NOISE='clean',U116_IR='proxy',U116_FAMILY='gap'),patch('e1_heightmap117.Recorder',Rec),patch.object(Path,'read_bytes',lambda p: b'CPU fixture protocol' if p.name=='protocol117.json' else _read_bytes(p)):
                    run(env,teacher,None,None,transport,args,N(seed=7),None,str(checkpoint),58084)
                result=json.loads(output.read_text());self.assertEqual(result['arm'],arm);self.assertEqual(len(actions),40);self.assertEqual(resets,[17]);self.assertGreater(result['transport_counts']['delivered'],0)
                if arm in ('a','C','B4'):self.assertTrue(any(torch.equal(a,torch.zeros_like(a)) for a in actions));self.assertTrue(any((a>.1).all() for a in actions))
                if arm=='c':self.assertEqual(result['max_teacher_action_equivalence_error'],0)
                if arm=='h':self.assertGreater(result['diagnostics']['wall_batch_ms']['map_integrate']['n'],0)
if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
