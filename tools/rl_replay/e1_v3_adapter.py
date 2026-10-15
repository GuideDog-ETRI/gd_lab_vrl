"""UPDATE62: exact UPDATE51 vanilla modules, with the original E1 environment.
Only these model/host modules come from UPDATE51; terrain stays in E1.
"""
import hashlib
import importlib.util
import json
from numbers import Integral
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT.parent / 'update51_student_v3'
CONFIG = V3 / 'outputs/remote90_run_config/config.json'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Pin the model implementation to the actual remote90 training configuration.
reference = json.loads(CONFIG.read_text())
load_module('gd_lab.gast.geometry', V3 / 'src/gd_lab/gast/geometry.py')
for relative in ('src/gd_lab/gast/student.py', 'src/gd_lab/gast/student_v3.py', 'src/gd_lab/gast/v3_state.py'):
    if digest(V3 / relative) != reference['code_sha256'][relative]:
        raise ValueError(f'UPDATE51 training source changed: {relative}')
    load_module('gd_lab.gast.' + Path(relative).stem, V3 / relative)
from gd_lab.gast.student_v3 import BeliefStudent
from gd_lab.gast.v3_state import BeliefBank

def contract_value(value, key):
    """Normalize Python checkpoint vs JSON representation, without tolerances."""
    if isinstance(value, dict):
        return {k: contract_value(v, key) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [contract_value(v, key) for v in value]
    # Training config json.dump(default=str) stringifies numpy shape integers.
    if key == 'policy_term_dims':
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, str) and value.isdecimal():
            return int(value)
    return value

def load_checkpoint(path, device='cpu', allow_fixture=False):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'update114' in saved.get('manifest', {}):
        sys.path.insert(0,str(ROOT))
        from v3_replay_adapter import load_v3
        return load_v3(path,device)
    if saved.get('synthetic_cpu_fixture') and not allow_fixture:
        raise ValueError('CPU fixture cannot be evaluated on Isaac')
    if saved.get('student_arch') != 'gast_belief_v3' or saved.get('size') != 'small':
        raise ValueError('requires gast_belief_v3 small vanilla')
    if saved.get('teacher_sha256') != reference['teacher_sha256']:
        raise ValueError('teacher hash differs from remote90 run')
    manifest = saved.get('manifest', {})
    expected = dict(reference)
    if saved.get('aux_lambda', 0) != 0:
        # UPDATE70 explicitly authorized training-only terrain head. Runtime ABI is unchanged.
        if saved.get('aux_lambda') != 0.1 or manifest.get('auxiliary', {}).get('lambda') != 0.1:
            raise ValueError('unsupported auxiliary coefficient')
        if saved.get('updates') != 1528 or manifest.get('seed') != 42 or manifest.get('resume') is not None:
            raise ValueError('UPDATE70 requires fresh s42 1528-update auxiliary run')
        codes = dict(reference['code_sha256'])
        del codes['tools/rl_replay/v3_runner.py']
        for name in ('tools/rl_replay/v3_runner_update70.py', 'src/gd_lab/gast/belief_aux.py', 'src/gd_lab/gast/v3_packet_contract.py'):
            codes[name] = digest(V3/name)
        expected['code_sha256'] = codes
    for key in ('architecture', 'size', 'source_sha256', 'transport', 'policy_terms', 'policy_term_dims', 'camera_contract', 'code_sha256'):
        if contract_value(manifest.get(key), key) != contract_value(expected[key], key):
            raise ValueError(f'checkpoint training contract differs: {key}')
    if not isinstance(saved.get('updates'), int) or saved['updates'] < 1:
        raise ValueError('requires completed training updates')
    teacher = ROOT / 'checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt'
    if digest(teacher) != reference['teacher_sha256']:
        raise ValueError('local teacher hash differs')
    student = BeliefStudent(size='small')
    student.load_state_dict(saved['model'], strict=True)
    student.to(device).eval().requires_grad_(False)
    return student, saved

class E1V3Driver:
    def __init__(self, student, n, device):
        self.bank = BeliefBank(student, n, device)

    def deliver(self, packet, step):
        frames, epochs, pose = packet.payload
        return self.bank.publish(frames, pose, packet.capture_step, packet.delivery_step, epochs, step)

    def action(self, actor, base, proprio, pose, step):
        if proprio.shape != (len(base), 230):
            raise ValueError('requires raw term-major policy [B,230]')
        raw = self.bank.tick(proprio, pose, step)
        gate = self.bank.effective(step)
        latent = raw * gate
        action = actor(torch.cat((base.detach(), latent), -1))
        if not torch.isfinite(action).all() or not torch.isfinite(latent).all():
            raise ValueError('nonfinite v3 action/latent')
        return action, latent, gate

    def after_step(self, executed, done, step):
        self.bank.previous_action = executed.detach().clone()
        self.bank.reset(done, step)

if __name__ == '__main__':
    torch.set_num_threads(1)
    _, saved = load_checkpoint(sys.argv[1])
    print(json.dumps({'strict_load': 'PASS', 'student_arch': saved['student_arch'],
                      'updates': saved['updates'], 'sha256': digest(sys.argv[1]),
                      'training_contract': 'remote90 vanilla small', 'device': 'cpu'}, indent=2))
