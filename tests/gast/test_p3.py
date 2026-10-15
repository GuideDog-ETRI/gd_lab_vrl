import copy
import torch
import pytest
from gd_lab.gast.student import GastStudent
from gd_lab.gast.p3 import (configure_train_parts, restore_optimizer,
                           latent_groups, balanced_latent_loss,
                           actor_jacobian_latent_loss)


@pytest.mark.parametrize('parts', ['all', 'head', 'head+gate'])
def test_real_student_frozen_parameters_and_memory(parts):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    student = GastStudent()
    params = configure_train_parts(student, parts)
    before = {k: v.clone() for k, v in student.named_parameters()}
    optimizer = torch.optim.Adam(params, lr=.001)
    frames = torch.rand(2, 4, 2, 45, 80)
    pose = torch.zeros(2, 7); pose[:, 3] = 1
    z, h, spatial, quality, gate = student.encode(frames, student.init_hidden(2, 'cpu'), pose)
    if parts != 'all':
        assert not h.requires_grad
        assert not spatial.requires_grad
    (z.square().mean() + quality.square().mean()).backward()
    optimizer.step()
    changed = []
    for name, p in student.named_parameters():
        selected = parts == 'all' or name.startswith('head.') or (parts == 'head+gate' and name.startswith('quality.'))
        assert p.requires_grad == selected
        if not selected:
            assert p.grad is None and torch.equal(p, before[name])
        if not torch.equal(p, before[name]):
            changed.append(name)
    assert any(n.startswith('head.') for n in changed)
    assert any(n.startswith('quality.') for n in changed) == (parts != 'head')
    assert {id(p) for p in params} == {id(p) for p in student.parameters() if p.requires_grad}


def test_default_update_bitwise_equal():
    torch.manual_seed(1)
    a = GastStudent(); b = copy.deepcopy(a)
    before_rng = torch.get_rng_state().clone()
    params = configure_train_parts(b, 'all')
    assert torch.equal(before_rng, torch.get_rng_state())
    frames=torch.rand(2,4,2,45,80); pose=torch.zeros(2,7); pose[:,3]=1
    ha=a.init_hidden(2,'cpu'); hb=b.init_hidden(2,'cpu')
    for age in (0., .1):
        za,ha,*_=a.encode(frames,ha,pose,age)
        zb,hb,*_=b.encode(frames,hb,pose,age)
        assert torch.equal(za,zb) and torch.equal(ha,hb)
    oa = torch.optim.Adam(a.parameters(), lr=.001)
    ob = torch.optim.Adam(params, lr=.001)
    x = torch.rand(2,187,32)
    for model, opt in ((a,oa),(b,ob)):
        model.head(x).square().mean().backward(); opt.step()
    assert all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items())


def test_optimizer_transition_and_subset_resume():
    s = GastStudent(); old = torch.optim.Adam(s.parameters())
    opt = torch.optim.Adam(configure_train_parts(s,'head'), lr=.002)
    assert not restore_optimizer(opt, {'optimizer':old.state_dict()}, 'head')
    assert opt.param_groups[0]['lr'] == .002
    s.head(torch.rand(1,187,32)).sum().backward(); opt.step()
    saved = {'train_parts':'head','optimizer':opt.state_dict()}
    other = torch.optim.Adam(s.head.parameters(), lr=.003)
    assert restore_optimizer(other,saved,'head')
    assert other.param_groups[0]['lr'] == .002 and len(other.state)>0
    with pytest.raises(ValueError): restore_optimizer(old,saved,'all')


def test_probe_groups_overlap_and_history():
    clean = torch.zeros(5,4,187); clean[:,3]=1
    terrain = torch.zeros(5,2,187); seen=torch.zeros(5,187,dtype=torch.bool)
    a,b=5*17+10,5*17+11
    clean[:4,2,a]=1; clean[0,2,b]=1
    terrain[0,1,a]=1; seen[1,a]=True
    clean[3,2,a]=0; clean[3,2,5*17+2]=1
    groups=latent_groups(clean,terrain,seen)
    assert groups.tolist()==[[True,False,True,False],[False,True,False,False],
                            [False,False,True,False],[False,False,False,False],
                            [False,False,False,True]]
    losses=torch.tensor([2.,4.,6.,100.,8.],requires_grad=True)
    loss=balanced_latent_loss(losses,groups)
    assert loss.item()==(2+4+(2+6)/2+8)/4
    loss.backward(); assert losses.grad[3]==0


def test_balance_empty_groups_fallback_and_stub():
    x=torch.tensor([1.,3.],requires_grad=True)
    assert balanced_latent_loss(x,torch.zeros(2,4,dtype=torch.bool))==2
    groups=torch.tensor([[1,0,0,0],[1,0,0,0]],dtype=torch.bool)
    assert balanced_latent_loss(x,groups)==2
    with pytest.raises(NotImplementedError): actor_jacobian_latent_loss()


def test_cli_and_launcher_contract_without_importing_isaac():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    source=(root/'scripts/gast/train_student_live.py').read_text()
    ast.parse(source)
    assert "choices=('all', 'head', 'head+gate'), default='all'" in source
    assert "if args_cli.actor_jacobian_latent_weight != 0:" in source
    launch=(root/'experiments/gast/train_gastv21_student_gapfocus.sh').read_text()
    for token in ('GAST_TRAIN_PARTS:-all','GAST_GROUP_BALANCED_LATENT:-0','GAST_ACTOR_JACOBIAN_LATENT_WEIGHT:-0'):
        assert token in launch
