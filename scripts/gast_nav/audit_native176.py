"""Native-pattern 2cm-column phase audit; geometry is explicit test fixture only."""
import json,time
from pathlib import Path
import torch
from gd_lab.gast_nav.native176 import Native176
from gd_lab.gast_nav.oracle_patterns176 import mounted_rays176,apply_self_sectors176
from gd_lab.gast_nav.oracle_occgrid176 import oracle_first_hits176
from gd_lab.gast_nav.producer168 import SensorPrimitives

def main():
 torch.set_num_threads(2);device='cuda:0';native=Native176(device)
 positions=[(x,y) for x in (1.,2.,-1.,-2.) for y in (-.045,-.015,.015,.045)]
 n=len(positions);c=torch.tensor([[x,y,-.25] for x,y in positions],device=device)[:,None]
 h=torch.tensor([.01,.01,.3],device=device).expand(n,1,3).clone()
 scene=SensorPrimitives(c,torch.eye(3,device=device).expand(n,1,3,3),h,torch.zeros(n,1,device=device,dtype=torch.long),torch.ones(n,1,device=device,dtype=torch.bool),torch.ones(n,device=device,dtype=torch.bool))
 records=[]
 for phase in (0,1,7,19):
  dirs,src,rings,mount=native.pattern_at(phase)
  rays,local=mounted_rays176(dirs,src,rings,mount,native.rot[None].expand(n,-1,-1,-1),native.pos[None].expand(n,-1,-1),native.profile)
  keep=rays.coverage_valid.any(0)
  from dataclasses import replace
  rays=replace(rays,**{k:getattr(rays,k)[:,keep] for k in rays.__dataclass_fields__})
  ids=torch.full((n,),phase,device=device,dtype=torch.long)
  r=apply_self_sectors176(oracle_first_hits176(scene,rays,scene_capture_id=ids,ray_capture_id=ids),local[:,keep],native.profile)
  count=(r.valid&torch.isfinite(r.distance_m)&~r.self_hit).sum(-1).tolist()
  records.extend(dict(phase=phase,x=x,y=y,hits=k) for (x,y),k in zip(positions,count))
 result=dict(column_width_m=.02,trials=len(records),misses=sum(r['hits']==0 for r in records),records=records,profile=native.manifest)
 out=Path('logs/gast/nav_v1/update176_native_audit.json');out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('records','profile')}))
if __name__=='__main__':main()
