"""Read-only UPDATE183 diagnostics and gate table; no inferred missing results."""
import json,sys
from pathlib import Path
import torch
out=Path(sys.argv[1]);runs=[Path(p) for p in sys.argv[2:]];rows=[]
for run in runs:
 r=json.loads((run/'result.json').read_text());cfg=json.loads((run/'resolved_config.json').read_text());ttl=cfg['front_depth183']['support_ttl_s']
 eligible=r.get('quick200',r.get('d5_denominator',{}));groups=eligible.get('groups',[]);ticks=sum(x['ticks'] for x in groups)
 row=dict(run=run.name,ttl_s=ttl,status=r['status'],steps=r.get('steps'),eligible_ticks=ticks,
  eligible_hold=eligible.get('hold_fraction'),observed_ticks=sum(x['sums']['observed'] for x in groups),
  progress=r.get('quick200',r).get('reward_means',{}).get('progress'),post3_progress=r.get('gate',{}).get('progress'),ready=r.get('gate',{}).get('ready'),wall_s=r.get('wall_s'),peak_reserved_bytes=r.get('peak_reserved_bytes'))
 for tick in (199,299,599):
  p=run/f'temporal_probe_{tick}.pt'
  if not p.exists():continue
  d=torch.load(p,map_location='cpu',weights_only=False);dg=d['swept_diagnostics'];o=d['supervisor_observation179'];req=dg['required'];fresh=(o.age_s>=0)&(o.age_s<=ttl)
  good=(o.state==1)&fresh;occ=(o.state==2)&fresh;support=torch.isfinite(o.support)&(o.support_age_s>=0)&(o.support_age_s<=ttl)
  row[f'tick_{tick}']=dict(observed=int(d['envelope_available'].sum()),required=int(req.sum()),unknown=int((req&~good&~occ).sum()),occupied=int((req&occ).sum()),missing_support=int((req&~support).sum()),entry=d.get('entry183'),producer_stats=d['producer_stats'])
 rows.append(row)
out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
