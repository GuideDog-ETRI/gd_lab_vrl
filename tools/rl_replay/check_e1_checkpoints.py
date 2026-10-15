"""CPU-only checkpoint/architecture preflight, no Isaac application."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
torch.set_num_threads(1)
from gd_lab.gast.student import GastStudent
p=ROOT/'checkpoints/students/control_seen0_perception_34624.pt'
t=ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
s=torch.load(p,map_location='cpu',weights_only=False)
assert s['iteration']==34624 and s['student_arch']=='gast_spatiotemporal_v1'
sha=hashlib.sha256(t.read_bytes()).hexdigest();assert s['teacher_sha256']==sha
student=GastStudent('vendor_new').eval();student.load_state_dict(s['model'],strict=True)
errors=[]
with torch.inference_mode():
    torch.manual_seed(26);frames=torch.rand(1,4,2,45,80);pose=torch.tensor([[0.,0,0,1,0,0,0]]);hidden=student.init_hidden(1,'cpu')
    for age in [0.,.15,.3]:
        latent,memory,_,_,gate=student.encode(frames,hidden,pose,age,torch.ones(1))
        raw=student.head(memory[:,:5984].reshape(-1,187,32));errors.append(float((latent-raw*gate).abs().max()))
        assert torch.equal(latent,raw*gate)
result={'student_iteration':s['iteration'],'student_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'teacher_sha256':sha,'strict_student_load':'PASS','actual_raw_gate_identity_max_abs_error':max(errors),'device':'cpu','simulator_initialized':False}
out=ROOT/'outputs/update26_checkpoint_preflight.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
# Build the real checkpoint architecture without importing Isaac config modules.
# Only extract the unchanged GastActorCritic class; its dependencies are torch-only.
import ast,types,yaml
from tensordict import TensorDict
from torch import nn
from gd_lab.teachers.cvtt.actor_critic import DreamwaqVrlActorCritic
from gd_lab.gast.temporal import TemporalTerrainEncoder
from e1_runner import load_teacher
from e1_core import action_at_boundary,LatentSwitch
source=(ROOT/'src/gd_lab/gast/teacher.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GastActorCritic')
ns={'torch':torch,'nn':nn,'DreamwaqVrlActorCritic':DreamwaqVrlActorCritic,'TemporalTerrainEncoder':TemporalTerrainEncoder}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual GastActorCritic CPU>', 'exec'),ns)
saved=torch.load(t,map_location='cpu',weights_only=False);state=saved['model_state_dict']
actor_dim=state['actor_obs_normalizer._mean'].numel();critic_dim=state['critic_obs_normalizer._mean'].numel()
obs=TensorDict({'policy':torch.randn(2,actor_dim)*.1,'critic':torch.randn(2,critic_dim)*.1,'terrain':torch.cat((torch.zeros(2,187),torch.ones(2,187)),-1),'gast_history':torch.zeros(2,3000)},batch_size=[2])
cfg=yaml.load((t.parent.parent/'params/agent.yaml').read_text(),Loader=yaml.FullLoader)
from unittest.mock import patch
with patch('torch.optim.Adam',side_effect=AssertionError('optimizer forbidden')):
    teacher=load_teacher(types.SimpleNamespace(get_observations=lambda:obs,num_actions=12),cfg,t,'cpu',policy_class=ns['GastActorCritic'])
    z=torch.zeros(2,32)
    action,_,_=action_at_boundary(teacher,obs,LatentSwitch(2,'cpu'),'c_teacher_full',z,z,torch.zeros(2,1),torch.zeros(2,dtype=torch.bool),torch.zeros(2),torch.ones(2),0)
    with torch.inference_mode():expected=teacher.act_with_terrain_latent(obs,teacher.terrain_latent(obs))
    assert torch.equal(action,expected) and not action.requires_grad
result['strict_teacher_load_without_any_Adam']='PASS';result['actual_teacher_full_injection_action_error']=float((action-expected).abs().max())
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
