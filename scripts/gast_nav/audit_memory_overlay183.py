"""Snapshot invariants: memory never erodes or rejuvenates direct observation."""
from pathlib import Path
import json,sys,torch
rows=[]
for arg in sys.argv[1:]:
 run=Path(arg);cfg=json.loads((run/'resolved_config.json').read_text());ttl=cfg['front_depth183']['support_ttl_s']
 for tick in (199,299):
  file=run/f'temporal_probe_{tick}.pt'
  if not file.exists():continue
  d=torch.load(file,map_location='cpu',weights_only=False);cur=d['current183'];out=d['supervisor_observation179']
  cf=(cur.state==1)&torch.isfinite(cur.age_s)&(cur.age_s>=0)&(cur.age_s<=ttl)
  cs=torch.isfinite(cur.support)&torch.isfinite(cur.support_age_s)&(cur.support_age_s>=0)&(cur.support_age_s<=ttl)
  lost=cf&(out.state==0);changed_support=cs&(~torch.isfinite(out.support)|((out.support-cur.support).abs()>1e-6))
  renew=cf&(out.state==1)&((out.age_s-cur.age_s).abs()>1e-6)
  assert not bool(lost.any()|changed_support.any()|renew.any())
  rows.append(dict(run=run.name,tick=tick,direct_free=int(cf.sum()),lost_direct_free=int(lost.sum()),direct_support=int(cs.sum()),changed_direct_support=int(changed_support.sum()),rejuvenated_direct_free=int(renew.sum()),memory_filled_free=int(((out.state==1)&~cf).sum()),retained_occupied_over_current_free=int((cf&(out.state==2)).sum())))
path=Path('logs/gast/nav_v1/update183/overlay_audit.json');path.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
