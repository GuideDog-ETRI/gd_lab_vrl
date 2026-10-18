"""A0 -> A preserves learned weights/Adam state while enabling old parameters."""
import torch
from cpu_fixture import observations, policy_pair, runtime, ROOT
from gd_lab.gast_nav.ppo import GastNavPPO
from gd_lab.gast_nav.pilot_runner import transition_trainable


def test_a0_to_a_retains_adam_and_enables_old_parameters():
    _, policy = policy_pair(observations(2))
    alg = GastNavPPO(policy, nav_config=runtime()['ppo'],
        trainable_manifest=ROOT/'configs/gast_nav/trainable_A0.yaml',
        device='cpu', schedule='fixed', learning_rate=1.e-3)
    existing = [p for group in alg.optimizer.param_groups for p in group['params']]
    old_frozen = [p for p in policy.parameters() if not p.requires_grad]
    assert old_frozen
    for p in existing:
        p.grad = torch.ones_like(p)
    alg.optimizer.step(); alg.optimizer.zero_grad(set_to_none=True)
    before = {p: {k:v.clone() if torch.is_tensor(v) else v for k,v in state.items()}
              for p,state in alg.optimizer.state.items()}
    weights = {k:v.clone() for k,v in policy.state_dict().items()}
    transition_trainable(alg, ROOT/'configs/gast_nav/trainable_A.yaml')
    assert all(p.requires_grad for p in old_frozen)
    for name,value in policy.state_dict().items():
        assert torch.equal(value,weights[name])
    for p,state in before.items():
        for k,v in state.items():
            torch.testing.assert_close(alg.optimizer.state[p][k],v,rtol=0,atol=0)
    assert all(p not in alg.optimizer.state for p in old_frozen)
    rates = {group['lr'] for group in alg.optimizer.param_groups}
    assert rates == {1.e-3, 1.e-4}
    p = old_frozen[0]; prior = p.detach().clone(); p.grad = torch.ones_like(p)
    alg.optimizer.step()
    assert not torch.equal(p,prior)
