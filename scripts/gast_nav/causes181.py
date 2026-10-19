from pathlib import Path
import json,torch
root=Path(__file__).resolve().parents[2];run=root/'logs/gast/nav_v1/update180_final600_v1';out={}
for tick in (299,599):
 d=torch.load(run/f'temporal_probe_{tick}.pt',map_location='cpu',weights_only=False);e=torch.load(run/f'evidence_{tick}.pt',map_location='cpu',weights_only=False);g=d['swept_diagnostics'];o=d['supervisor_observation179'];r=g['required']
 masks={key:g[key] for key in ('unknown','missing_traversable','occupied','outside','qualified','valid_epoch','cells_satisfied')}
 masks['unknown_and_missing']=g['unknown']&g['missing_traversable']
 masks['unknown_only_no_occupied']=g['unknown']&~g['occupied']
 masks['raw_geometry_pass']=g['qualified']&g['valid_epoch']
 row={}
 for kind in sorted({v['kind'] for v in e['rows']}):
  ids=torch.tensor([i for i,v in enumerate(e['rows']) if v['kind']==kind]);row[kind]={'envs':len(ids),**{key:int(mask[ids].sum()) for key,mask in masks.items()}}
 out[str(tick)]=row
(root/'logs/gast/nav_v1/update181/causes180.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
