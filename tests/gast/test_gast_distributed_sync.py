"""Run with python -m torch.distributed.run --standalone --nproc_per_node=3."""
import importlib.util
from pathlib import Path
import torch
import torch.distributed as dist
from rsl_rl.networks.normalization import EmpiricalNormalization

path = Path(__file__).parents[2]/'src/gd_lab/rl/distributed.py'
spec = importlib.util.spec_from_file_location('gast_distributed', path)
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
dist.init_process_group('gloo')
rank = dist.get_rank()
p = torch.nn.Parameter(torch.tensor([1.]))
q = torch.nn.Parameter(torch.tensor([2.]))
p.grad = torch.tensor([float(rank+1)])
if rank == 0: q.grad = torch.tensor([3.])
d.average_gradients([p, q])
assert p.grad.item() == 2. and q.grad.item() == 1.
assert d.any_rank(rank == 1, 'cpu')
norm = EmpiricalNormalization(2)
x = torch.arange((rank+1)*2).float().reshape(-1,2)+rank*10
d.update_normalizer(norm, x)
all_x = [None]*3
dist.all_gather_object(all_x, x)
expected = torch.cat(all_x)
torch.testing.assert_close(norm.mean, expected.mean(0))
torch.testing.assert_close(norm._var.squeeze(0), expected.var(0, unbiased=False))
assert norm.count.item() == 6
d.assert_synchronized(norm, [])
if rank == 0: print('GAST_DISTRIBUTED_UNIT_PASS', flush=True)
dist.destroy_process_group()
