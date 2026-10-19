"""Necessary outer-bound check; violations prove actual projected under-coverage.
This audit NEVER grants runtime qualification.
"""
import json,hashlib
from pathlib import Path
import torch,yaml
from gd_lab.gast_nav.sweep179 import measured_sweep179
from gd_lab.gast_nav.body_projection import CollisionFootprint
from gd_lab.core.camera_geometry import camera_rotation_matrix
ROOT=Path(__file__).resolve().parents[2];torch.set_num_threads(2)
cfg=yaml.safe_load((ROOT/'configs/gast_nav/pipeline179.yaml').read_text())
settings={k:v for k,v in cfg.items() if k not in ('current_body_margin_m','qualification_file')}
cfg['qualification_file']='logs/gast/nav_v1/update179/sweep_outer_bounds.json'
rows=[]
for folder in ('all','postures','rest'):
 for file in sorted((ROOT/f'logs/gast/nav_v1/update174_envelope_{folder}').glob('*_seed*.pt')):
  d=torch.load(file,map_location='cpu',weights_only=False);T,N=d['pos'].shape[:2]
  starts=d['starts']-1;ids=torch.arange(N);pose=d['start_pose'];c,s=pose[:,3].cos(),pose[:,3].sin();z=torch.zeros_like(c);o=torch.ones_like(c)
  yaw=torch.stack((c,s,z,-s,c,z,z,z,o),-1).reshape(N,3,3)
  rot=torch.einsum('nij,tnljk->tnlik',yaw,camera_rotation_matrix(d['quat']))
  pos=torch.einsum('nij,tnlj->tnli',yaw,d['pos']-pose[None,:,None,:3])
  centres=[];half=[];fp=CollisionFootprint(d['body_names'],'cpu')
  for i,kind,off,axis,size in fp.parts:
   centres.append(pos[:,:,i]+torch.einsum('tnij,j->tni',rot[:,:,i],off))
   half.append((rot[:,:,i]@axis).abs()@size if kind=='box' else torch.ones_like(pos[:,:,i])*size)
  centres=torch.stack(centres,2);half=torch.stack(half,2);dt=d['dt']
  before=(starts-1).clamp_min(0)
  v3=torch.einsum('nij,nj->ni',yaw,(d['base_pose'][starts,ids,:3]-d['base_pose'][before,ids,:3])/dt)
  angle=d['base_pose'][starts,ids,3]-d['base_pose'][before,ids,3]
  w=torch.atan2(angle.sin(),angle.cos())/dt
  cc=centres[starts,ids];hh=half[starts,ids]
  angular=torch.zeros_like(v3);angular[:,2]=w
  relative=(cc-centres[before,ids])/dt-v3[:,None]-torch.cross(angular[:,None].expand_as(cc),cc,dim=-1)
  for j,(i,*_) in enumerate(fp.parts):
   if d['body_names'][i]=='trunk':relative[:,j]=0
  sweep,stats=measured_sweep179(cc,hh,v3[:,:2],w,relative,control_dt=dt,**settings)
  # Each measured primitive AABB must be contained in one predicted AABB
  # for that primitive. This is sufficient, not merely outer-bounds coverage.
  Q=cc.shape[1];A=settings['samples']
  lo=(sweep.centres-sweep.half).reshape(N,A,Q,3)
  hi=(sweep.centres+sweep.half).reshape(N,A,Q,3)
  contained=torch.ones(N,dtype=torch.bool);max_excess=torch.zeros(N)
  for begin in range(0,T,25):
   end=min(begin+25,T);valid=torch.arange(begin,end)[:,None]>=starts[None]
   low=(centres-half)[begin:end];high=(centres+half)[begin:end]
   outer_lo=lo[...,:2].amin((1,2));outer_hi=hi[...,:2].amax((1,2))
   best=torch.maximum(outer_lo[None,:,None]-low[...,:2],high[...,:2]-outer_hi[None,:,None]).clamp_min(0).amax((-1,-2))
   contained &= ((best<=1e-5)|~valid).all(0)
   max_excess=torch.maximum(max_excess,best.masked_fill(~valid,0).amax(0))
  stopped=(d['speed'][-5:]<.05).all(0)&(d['omega'][-5:].abs()<.1).all(0)
  for kind in sorted(set(d['kinds'])):
   mask=torch.tensor([k==kind for k in d['kinds']]);good=contained&stopped&~d['invalid'].bool()
   rows.append(dict(source=str(file.relative_to(ROOT)),source_sha=hashlib.sha256(file.read_bytes()).hexdigest(),kind=kind,mode=d['mode'],value=(-abs(float(d['value'])) if d['mode']=='TURN' else float(d['value'])),trials=int(mask.sum()),contained=int(contained[mask].sum()),stopped=int(stopped[mask].sum()),passed=int(good[mask].sum()),max_excess_m=float(max_excess[mask].max()),velocity_min=v3[mask,:2].amin(0).tolist(),velocity_max=v3[mask,:2].amax(0).tolist(),omega_min=float(w[mask].min()),omega_max=float(w[mask].max())))
  print(file.name,'contained',int(contained.sum()),'stopped',int(stopped.sum()),flush=True)
domains=[]
for key in sorted({(r['kind'],r['mode'],r['value']) for r in rows}):
 selected=[r for r in rows if (r['kind'],r['mode'],r['value'])==key]
 domains.append(dict(kind=key[0],mode=key[1],value=key[2],trials=sum(r['trials'] for r in selected),passed=sum(r['passed'] for r in selected),qualified=False,velocity_min=[min(r['velocity_min'][j] for r in selected) for j in (0,1)],velocity_max=[max(r['velocity_max'][j] for r in selected) for j in (0,1)],omega_min=min(r['omega_min'] for r in selected),omega_max=max(r['omega_max'] for r in selected)))
result=dict(settings=settings,method='All recorded ticks; necessary OUTER horizontal bounds only. Outside is definitely under-covered; inside is NOT union proof. Runtime qualification always false in this auxiliary audit.',rows=rows,domains=domains)
p=ROOT/cfg['qualification_file'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2));print('qualified domains',sum(r['qualified'] for r in domains), '/',len(domains))
