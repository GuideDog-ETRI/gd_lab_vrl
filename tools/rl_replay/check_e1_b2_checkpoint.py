"""CPU-only direct resume compatibility check. Never imports Isaac or converts weights."""
import os,sys,json,hashlib,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from gd_lab.gast.student import GastStudent
from gd_lab.core.camera_contract import check_checkpoint_camera_contract

def check(path):
    torch.set_num_threads(1)
    baseline_requested=os.environ.get('E1_B2_BASELINE')=='1'
    if baseline_requested:
        expected='7d8ee014d6fa17608a4630514e0ca888aa80ca5825414175899c5f257ceeb4ee'
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('E1_B2_BASELINE requires the exact approved control34624 SHA256')
    c=torch.load(path,map_location='cpu',weights_only=False)
    t=ROOT/'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
    assert c['teacher_sha256']==hashlib.sha256(t.read_bytes()).hexdigest()
    assert c['student_arch']=='gast_spatiotemporal_v1' and c['gast_distillation_contract']==2
    assert 0<=int(c['iteration'])<10**9
    if not baseline_requested:
        assert c['action_distillation']['student_policy']=='frozen'
    baseline=torch.load(ROOT/'checkpoints/students/control_seen0_perception_34624.pt',map_location='cpu',weights_only=False)
    assert c['camera_contract']==baseline['camera_contract']
    check_checkpoint_camera_contract(c['camera_contract'],purpose='resume')
    student=GastStudent('vendor_new').eval();student.load_state_dict(c['model'],strict=True)
    with torch.inference_mode():
        torch.manual_seed(34);frames=torch.rand(1,4,2,45,80);pose=torch.tensor([[0.,0,0,1,0,0,0]])
        for age in (0.,.15,.3):
            z,m,_,_,g=student.encode(frames,student.init_hidden(1,'cpu'),pose,age,torch.ones(1))
            assert torch.isfinite(m).all() and torch.equal(z,student.head(m[:,:5984].reshape(-1,187,32))*g)
    return {'checkpoint':str(Path(path).resolve()),'iteration':c['iteration'],'strict_load':'PASS',
      'resume_metadata':'PASS','raw_gate_identity':'PASS','conversion_required':False,
      'synthetic_cpu_fixture':bool(c.get('update31',{}).get('synthetic_cpu_smoke'))}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');args=parser.parse_args()
    result=check(args.checkpoint)
    if result['synthetic_cpu_fixture']:raise ValueError('CPU fixture is not an evaluation checkpoint')
    print(json.dumps(result,indent=2))
