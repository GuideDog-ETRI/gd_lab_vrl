"""Exact D-5 gate observation, including measured contact support sidecar."""
from pathlib import Path
import sys,json,torch
run=Path(sys.argv[1]);out={}
for tick in (0,299,599):
 p=run/f'temporal_probe_{tick}.pt'
 if not p.exists():continue
 d=torch.load(p,map_location='cpu',weights_only=False);o=d['supervisor_observation179'];gate=d['swept_diagnostics'];r=gate['required']
 fresh=torch.isfinite(o.age_s)&(o.age_s>=0)&(o.age_s<=.5)
 support=torch.isfinite(o.support)&torch.isfinite(o.support_age_s)&(o.support_age_s>=0)&(o.support_age_s<=.5)
 gap=torch.zeros_like(r) if o.walkable_gap is None else o.walkable_gap&torch.isfinite(o.walkable_gap_age_s)&(o.walkable_gap_age_s>=0)&(o.walkable_gap_age_s<=.5)
 free=(o.state==1)&fresh;occupied=(o.state==2)&fresh;unknown=~free&~occupied
 xy=torch.stack(torch.meshgrid(torch.arange(r.shape[-2])*.1-2.1,torch.arange(r.shape[-1])*.1-2.4,indexing='ij'),-1)
 masks=dict(required=r,unknown=r&unknown,occupied=r&occupied,missing_support=r&~support,missing_traversable=r&~(support|gap),unknown_and_missing=r&unknown&~support,free_with_support=r&free&support)
 row={name:int(mask.sum()) for name,mask in masks.items()}
 row.update(body_excluded=int((gate['required_before_exclusion']&~r).sum()),qualified=int(gate['qualified'].sum()),cells_satisfied=int(gate['cells_satisfied'].sum()),outside=int(gate['outside'].sum()),route_shape=list(d['route_features179'].shape),route_finite=bool(torch.isfinite(d['route_features179']).all()),producer=d['producer_stats'])
 row['x_bins']={}
 for a,b in ((-2.45,-.85),(-.85,0),(0,.35),(.35,.85),(.85,2.45)):
  subset=(xy[...,1]>=a)&(xy[...,1]<b)
  row['x_bins'][f'{a}:{b}']={name:int((mask&subset).sum()) for name,mask in masks.items()}
 out[str(tick)]=row
(run/'attribution179.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:{name:value for name,value in row.items() if name not in ('x_bins','producer')} for k,row in out.items()},indent=2))
