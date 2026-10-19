"""Necessary XY bounds check against every recorded stop trial; no coverage proof."""
import json
from pathlib import Path
import torch,yaml
from gd_lab.gast_nav.analytic_sweep177 import stopping_sweep177
from gd_lab.gast_nav.body_projection import CollisionFootprint
from gd_lab.core.camera_geometry import camera_rotation_matrix
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
settings=yaml.safe_load((ROOT/'configs/gast_nav/analytic_sweep177.yaml').read_text())
rows=[]
for folder in ('all','postures','rest'):
 for file in sorted((ROOT/f'logs/gast/nav_v1/update174_envelope_{folder}').glob('*_seed*.pt')):
  d=torch.load(file,map_location='cpu',weights_only=False);T,N=d['pos'].shape[:2]
  starts=d['starts']-1;ids=torch.arange(N);pose=d['start_pose'];c,s=pose[:,3].cos(),pose[:,3].sin();z=torch.zeros_like(c);o=torch.ones_like(c)
  yaw=torch.stack((c,s,z,-s,c,z,z,z,o),-1).reshape(N,3,3)
  rot=torch.einsum('nij,tnljk->tnlik',yaw,camera_rotation_matrix(d['quat']))
  pos=torch.einsum('nij,tnlj->tnli',yaw,d['pos']-pose[None,:,None,:3])
  centres=[];halves=[];fp=CollisionFootprint(d['body_names'],'cpu')
  for i,kind,off,axis,size in fp.parts:
   centres.append(pos[:,:,i]+torch.einsum('tnij,j->tni',rot[:,:,i],off))
   halves.append((rot[:,:,i]@axis).abs()@size if kind=='box' else torch.ones_like(pos[:,:,i])*size)
  centres=torch.stack(centres,2);halves=torch.stack(halves,2)
  v=torch.zeros(N,2);w=torch.zeros(N)
  if d['mode']=='NAV':v[:,0]=d['value']
  else:w[:]=d['value']
  sweep,_=stopping_sweep177(centres[starts,ids],halves[starts,ids],v,w,control_dt=d['dt'],**settings)
  alo=(sweep.centres-sweep.half)[...,:2].amin(1);ahi=(sweep.centres+sweep.half)[...,:2].amax(1)
  valid=torch.arange(T)[:,None]>=starts[None]
  mlo=(centres-halves)[...,:2].masked_fill(~valid[:,:,None,None],torch.inf).amin((0,2))
  mhi=(centres+halves)[...,:2].masked_fill(~valid[:,:,None,None],-torch.inf).amax((0,2))
  excess=torch.maximum(alo-mlo,mhi-ahi).clamp_min(0).amax(-1)
  for kind in sorted(set(d['kinds'])):
   selected=torch.tensor([k==kind for k in d['kinds']]);e=excess[selected]
   rows.append(dict(source=str(file.relative_to(ROOT)),kind=kind,mode=d['mode'],value=float(d['value']),trials=len(e),outside_bounds=int((e>1e-5).sum()),max_excess_m=float(e.max()),outside_with_margin_010=int((e>.10+1e-5).sum()),outside_with_margin_020=int((e>.20+1e-5).sum())))
result=dict(method='Necessary per-trial XY bounding-box inclusion only; inside bounds is NOT proof of union coverage. Every recorded post-stop tick included, including non-stopping trials. No P2, no empirical padding added.',settings=settings,trials=sum(r['trials'] for r in rows),outside_bounds=sum(r['outside_bounds'] for r in rows),outside_with_margin_010=sum(r['outside_with_margin_010'] for r in rows),outside_with_margin_020=sum(r['outside_with_margin_020'] for r in rows),max_excess_m=max(r['max_excess_m'] for r in rows),rows=rows)
p=ROOT/'logs/gast/nav_v1/update177/analytic_empirical_audit.json';p.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
