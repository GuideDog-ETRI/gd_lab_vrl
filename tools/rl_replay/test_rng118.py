import sys,types,torch,numpy as np,random
sys.path.insert(0,'/home/user/gd_project/archive/aux_worktrees/update116_independent_ir/tools/rl_replay')
from blocks118 import install
N=types.SimpleNamespace
results=[]
for order in (list(range(512)),list(reversed(range(512)))):
    out=torch.zeros(512,3)
    def event(env,env_ids):out[env_ids]=torch.tensor([torch.rand(()).item(),np.random.random(),random.random()])
    cfg=N(events=N(draw=N(mode='reset',func=event)));install(cfg)
    torch.manual_seed(99);before=torch.random.get_rng_state().clone()
    cfg.events.draw.func(N(num_envs=512,device='cpu'),torch.tensor(order))
    assert torch.equal(before,torch.random.get_rng_state())
    results.append(out)
assert torch.equal(*results)
assert len({tuple(results[0][e].tolist()) for e in (0,128,256,384)})==4
print('512 env block reset RNG: order independence, global RNG restoration, four independent seeds OK')
