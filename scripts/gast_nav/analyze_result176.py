"""Summarize actual final-run gates without replacing failed acceptance checks."""
import json,sys
from pathlib import Path
import torch
run=Path(sys.argv[1]);out={}
for step in (0,299,599):
 p=run/f'temporal_probe_{step}.pt'
 if not p.exists():continue
 data=torch.load(p,map_location='cpu',weights_only=False);d=data['swept_diagnostics']
 row={k:int(v.sum()) for k,v in d.items() if isinstance(v,torch.Tensor) and v.ndim==1}
 row['required_cells']=int(d['required'].sum())
 row['required_envs']=int(d['required'].flatten(1).any(1).sum())
 for key in ('occupied','unknown','missing_support'):
  if key in d and 'qualified' in d:row['qualified_and_'+key]=int((d['qualified']&d[key]).sum())
 f=data['occupancy_frame']
 row.update(capture_free_cells=int((f.state==1).sum()),capture_occupied_cells=int((f.state==2).sum()),capture_support_cells=int(f.support.observed.sum()))
 row['producer_stats']=data['producer_stats'];out[str(step)]=row
(run/'gate_attribution176.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
