"""Descriptive same-env body-chart region counts, not causal state parity."""
from pathlib import Path
import json,torch
from gd_lab.gast_nav.map_types import WIDE_GRID
root=Path(__file__).resolve().parents[2];xy=WIDE_GRID.xy(device='cpu',dtype=torch.float32);x=xy[...,0];y=xy[...,1]
regions={'front_gap_085_16':(x>=.85)&(x<=1.6)&(y.abs()<=.55),'front_leg_side':(x>=0)&(x<=.35)&(y.abs()>=.30)&(y.abs()<=.40),'terrain_roi':(x.abs()<=.85)&(y.abs()<=.55)}
rows=[]
for name in ['update182_after200_v1','update183_ttl05_200_final','update183_ttl1_200_final','update183_ttl2_200_final']:
 p=root/'logs/gast/nav_v1'/name/'temporal_probe_199.pt';d=torch.load(p,map_location='cpu',weights_only=False);o=d['supervisor_observation179'];ttl=2. if 'ttl2_' in name else 1. if 'ttl1_' in name else .5
 f=(o.state[0]==1)&(o.age_s[0]>=0)&(o.age_s[0]<=ttl);h=torch.isfinite(o.support[0])&(o.support_age_s[0]>=0)&(o.support_age_s[0]<=ttl)
 rows.append(dict(run=name,env_id=0,tick=199,regions={k:dict(cells=int(v.sum()),free=int((v&f).sum()),support=int((v&h).sum()),both=int((v&f&h).sum())) for k,v in regions.items()}))
p=root/'logs/gast/nav_v1/update183/region_comparison.json';p.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
