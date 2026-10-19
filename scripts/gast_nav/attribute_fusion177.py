"""Attribute final gate masks in the saved current query chart (not raw capture chart)."""
import json,sys
from pathlib import Path
import torch
run=Path(sys.argv[1]);output={}
for tick in (0,299,599):
 p=run/f'temporal_probe_{tick}.pt'
 if not p.exists():continue
 data=torch.load(p,map_location='cpu',weights_only=False);d=data['swept_diagnostics'];f=data.get('fusion_frame177')
 if f is None:continue
 req=d['required'];unknown=req&(f.state==0);occupied=req&(f.state==2);missing=req&~f.support.observed
 row=dict(required_cells=int(req.sum()),unknown_cells=int(unknown.sum()),occupied_cells=int(occupied.sum()),missing_support_cells=int(missing.sum()),unknown_and_missing=int((unknown&missing).sum()),free_with_support_cells=int((req&(f.state==1)&f.support.observed).sum()),qualified_envs=int(d['qualified'].sum()),producer_stats=data['producer_stats'])
 # Grid centers match OccConfig/WIDE_GRID, x fastest.
 x=torch.arange(req.shape[-1])*.1-2.4
 row['x_bins']={}
 for lo,hi in ((-2.45,-.85),(-.85,0.),(0.,.85),(.85,2.45)):
  mask=(x>=lo)&(x<hi)
  row['x_bins'][f'{lo}:{hi}']={k:int((v&mask[None,None]).sum()) for k,v in [('required',req),('unknown',unknown),('occupied',occupied),('missing_support',missing)]}
 output[str(tick)]=row
(run/'fusion_attribution177.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
