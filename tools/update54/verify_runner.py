"""Run real UPDATE51/54 runners on a deterministic CPU fake host/transport.
No simulator. Optimizer step is deliberately replaced with a no-op.
"""
import copy,json,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import torch._dynamo  # initialize before temporary module injection
from verify_cpu import fixture, ROOT, OUT
import v3_runner as old
import v3_runner_update54 as new
from gd_lab.gast.student import GastStudent
from gd_lab.core.camera_geometry import mounted_camera_world_poses

class Host:
    clip_actions=1.
    def __init__(self):
        self.b,self.obs,self.extra,self.epochs,self.contract=fixture(32)
        self.b.num_envs=32;self.b.device='cpu';self.b.step_dt=.01;self.b.common_step_counter=0
        self.b.command_manager=types.SimpleNamespace(get_term=lambda name:None)
        self.b.observation_manager=types.SimpleNamespace(group_obs_term_dim={'policy':[230]},active_terms={'policy':['fake']})
        self.b.cfg.scene=types.SimpleNamespace(terrain=types.SimpleNamespace(terrain_generator=types.SimpleNamespace(v3_geometry_sha256='fake',v3_gap_metadata_sha256='fake')))
        self.unwrapped=self.b;self.executed=[];self.refresh()
    def refresh(self):
        b=self.b;s=b.common_step_counter
        d=b.scene['robot'].data;d.root_pos_w[:,0]=s*.01;d.root_pos_w[:,2]=.5+s*.002
        scan=b.scene['height_scanner'].data;scan.pos_w.copy_(d.root_pos_w)
        y,x=torch.meshgrid(torch.linspace(-.5,.5,11),torch.linspace(-.8,.8,17),indexing='ij')
        scan.ray_hits_w[...,0]=x.flatten()[None]+d.root_pos_w[:,0,None]
        b._vrl_camera_snapshot=list(b._vrl_camera_snapshot)
        b._vrl_camera_snapshot[2],b._vrl_camera_snapshot[3]=mounted_camera_world_poses(d.root_pos_w,d.root_quat_w,self.contract)
        b._vrl_camera_snapshot=tuple(b._vrl_camera_snapshot)
        b._vrl_camera_snapshot_steps.fill_(s);b._vrl_teacher_terrain_capture_steps.fill_(s)
        t=self.obs['terrain'];t[:,:187]=(scan.pos_w[:,2,None]-.5)*5
        b._vrl_teacher_terrain_snapshot=t.clone()
        self.obs['policy']=torch.full((32,230),s*.01)
        self.obs['step']=s
    def reset(self):return self.obs,{}
    def step(self,action):
        self.executed.append(action.clone());self.b.common_step_counter+=1;self.refresh()
        done=torch.zeros(32,dtype=torch.bool)
        if self.b.common_step_counter==5:done[0]=True
        return self.obs,None,done,{}

class Teacher:
    def __init__(self):
        self.actor=nn.Sequential(nn.Linear(235,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)).requires_grad_(False)
        state=torch.load(ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt',map_location='cpu',weights_only=False)['model_state_dict']
        self.actor.load_state_dict({k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')});self.teacher=self.actor
    def _actor_input(self,obs,inference=True):
        return torch.full((32,235),obs['step']*.01)
    def act_inference(self,obs):return self.actor(self._actor_input(obs))

class Attention:
    def __init__(self,host,teacher):self.host=host;self.teacher=teacher;self.camera_contract=host.contract
    def capture(self,obs,action):
        b=self.host.b;s=b.common_step_counter;t=obs['terrain'];d=b.scene['robot'].data
        pose=torch.cat((d.root_pos_w[:,:2],torch.zeros(32,1),d.root_quat_w),-1)
        packed=self.teacher._actor_input(obs)
        return packed[:,:203],torch.cat((t,torch.zeros(32,187),torch.ones(32,187)),-1),t.clone(),pose,torch.full((32,),s*.01),action,packed[:,203:]

class Transport:
    def __init__(self):self.pending=[];self.dropped=0;self.config=types.SimpleNamespace(manifest=lambda dt:{'fake':True})
    def next_capture(self,step):return step+1
    def capture(self,step,payload):self.pending.append(types.SimpleNamespace(capture_step=step,delivery_step=step+2,payload=payload))
    def receive(self,step):
        result=[p for p in self.pending if p.delivery_step<=step];self.pending=[p for p in self.pending if p.delivery_step>step];return result

class Groups:
    def __init__(self,b):self.b=b;self.gap=None
    def labels(self,obs):return (torch.arange(32)+obs['step'])%3
class Sampler:
    def __init__(self,*args):pass
    def summary(self):return {'fake':True}
    def close(self):pass
class Labels(Sampler):
    def __init__(self,*args):self.terminal={}
    def read(self):return {}

observations=types.ModuleType('gd_lab.gast.observations')
observations.pose_xyyaw=lambda b:torch.cat((b.scene['robot'].data.root_pos_w[:,:2],torch.zeros(32,1)),-1)

def run(module,coefficient,path):
    torch.manual_seed(54);host=Host();teacher=Teacher();visual=GastStudent()
    source=ROOT/'checkpoints/students/control_seen0_perception_34624.pt'
    visual.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['model'])
    args=types.SimpleNamespace(v3_size='small',v3_resume=None,v3_aux_lambda=coefficient,student_resume=str(source),b2_out=str(path),seed=54,v3_round=0)
    with patch.dict(sys.modules,{'gd_lab.gast.observations':observations}), patch.multiple(module,TerrainGroups=Groups,ApproachSampler=Sampler,Labels=Labels,mixed_velocity=lambda *a:None,budget_reason=lambda *a:'cpu_test'), patch.object(torch.optim.Adam,'step',lambda *a,**kw:None):
        module.run(host,teacher,visual,Attention(host,teacher),Transport(),args,None,None,ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt',0)
    stats=json.loads((path/'windows.jsonl').read_text());data=torch.load(path/'dagger_00000.pt',map_location='cpu',weights_only=False)
    saved=torch.load(path/'student_v3.pt',map_location='cpu',weights_only=False)
    return torch.stack(host.executed),stats,data,saved

with tempfile.TemporaryDirectory(prefix='update54_runner_') as temp:
    root=Path(temp);a=run(old,0,root/'old');b=run(new,0,root/'zero');c=run(new,.1,root/'enabled')
    assert torch.equal(a[0],b[0]) and torch.equal(a[0],c[0])
    for key in ('action','raw','spatial','preserve','action_n','raw_n','spatial_rows','preserve_n','driver_state_sha256'):
        assert a[1][key]==b[1][key]==c[1][key],key
    for ta,tb in zip(a[2]['ticks'],b[2]['ticks']):
        for key in ('teacher_execution_raw','action','gate','quality','proprio','baseline_action','epoch'):
            assert torch.equal(ta[key],tb[key]),key
    assert b[3]['belief_aux'] is None and c[3]['belief_aux'] is not None
    assert c[1]['aux_loss']>0 and sum(c[1]['aux_group_n'].values())>0
    # Existing export loader accepts enabled checkpoint's model strictly.
    from gd_lab.gast.student_v3 import BeliefStudent
    deployed=BeliefStudent(size='small');deployed.load_state_dict(c[3]['model'],strict=True)
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('export51',ROOT/'tools/update51/export_v3.py');export=module_from_spec(spec);spec.loader.exec_module(export)
    export.export(root/'zero/student_v3.pt',root/'zero_onnx');export.export(root/'enabled/student_v3.pt',root/'enabled_onnx')
    for tag in ('encode','tick'):
        assert (root/'zero_onnx'/f'{tag}.onnx').read_bytes()==(root/'enabled_onnx'/f'{tag}.onnx').read_bytes()
    report={'status':'PASS','host':'deterministic CPU fake; actual runners and actual GAST2000 actor','ticks':len(a[0]),
        'delay_ticks':2,'partial_reset_step':5,'optimizer_steps':0,'lambda0_execution_and_base_losses_bit_equal':True,
        'lambda_positive_checkpoint_export_graph_bytes_equal':True,'aux_stats':{k:v for k,v in c[1].items() if k.startswith('aux')}}
    (OUT/'runner_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
