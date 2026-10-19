import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('ablation181',Path(__file__).resolve().parents[2]/'scripts/gast_nav/ablation181.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_diagnostic_independent_selectors():
 d=dict(unknown=torch.tensor([True,False,False]),missing_traversable=torch.tensor([False,True,False]),occupied=torch.tensor([False,False,True]))
 z=torch.zeros(3,dtype=torch.bool)
 for mode,key in [('unknown','unknown'),('support','missing_traversable'),('occupied','occupied')]:
  known,out=m.select181(mode,z,d)
  assert torch.equal(known,~d[key])
  assert torch.equal(out['occupied'],d['occupied'] if mode=='occupied' else z)
 for mode in ('off','swept','health'):
  known,out=m.select181(mode,z,d);assert known.all() and not out['occupied'].any()
 assert not m.select181('all',z,d)[0].any()
 assert d['occupied'][2]
