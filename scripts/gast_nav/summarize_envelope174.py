"""All-trial maximum 3D collision envelopes; temporary C diagnostic, no deployment claim."""
import argparse,json,math,hashlib
from pathlib import Path
import torch,yaml
from gd_lab.gast_nav.body_projection import CollisionFootprint
from gd_lab.core.camera_geometry import camera_rotation_matrix
from gd_lab.gast_nav.swept167 import p2_projected_inflation
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);groups={};sources=[]
root=Path(__file__).resolve().parents[2]
candidate=yaml.safe_load((root/'configs/gast_nav/curriculum_C.yaml').read_text())['swept_candidate']
for folder in a.input:
 for file in sorted(folder.glob('*_seed*.pt')):
  d=torch.load(file,map_location='cpu',weights_only=False);sources.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
  T,N=d['pos'].shape[:2];starts=d['starts']-1;ids=torch.arange(N);pose=d['start_pose'];c,s=pose[:,3].cos(),pose[:,3].sin();zero=torch.zeros_like(c);one=torch.ones_like(c)
  yaw=torch.stack((c,s,zero,-s,c,zero,zero,zero,one),-1).reshape(N,3,3)
  rot=camera_rotation_matrix(d['quat']);relrot=torch.einsum('nij,tnljk->tnlik',yaw,rot)
  local=torch.einsum('nij,tnlj->tnli',yaw,d['pos']-pose[None,:,None,:3])
  from gd_lab.gast_nav.posture174 import collision_points174
  fp=CollisionFootprint(d['body_names'],'cpu')
  posture=collision_points174(local[starts,ids],relrot[starts,ids],fp.parts)
  mask=torch.arange(T)[:,None]>=starts[None]
  planar=(d['base_pose'][1:,:,:2]-d['base_pose'][:-1,:,:2]).norm(dim=-1)/d['dt']
  fallen=((rot[:,:,d['body_names'].index('trunk'),2,2]<math.cos(math.radians(70)))&mask).any(0)
  stopped=(planar[-20:]<.05).all(0)&(d['omega'][-20:].abs()<.05).all(0)&~fallen
  fp=CollisionFootprint(d['body_names'],'cpu')
  assert max(float(off.norm())+(float(size.norm()) if kind=='box' else float(size)) for _,kind,off,_,size in fp.parts)<=.6
  lo=[];hi=[];motion=[]
  for index,kind,offset,axis,size in fp.parts:
   centre=local[:,:,index]+torch.einsum('tnij,j->tni',relrot[:,:,index],offset)
   half=(relrot[:,:,index]@axis).abs()@size if kind=='box' else torch.ones_like(centre)*size
   lo.append((centre-half).masked_fill(~mask[...,None],torch.inf).amin(0))
   hi.append((centre+half).masked_fill(~mask[...,None],-torch.inf).amax(0))
   motion.append(((centre[1:]-centre[:-1]).norm(dim=-1)+(half[1:]-half[:-1]).norm(dim=-1)).masked_fill(~mask[1:],0).amax(0))
  lo=torch.stack(lo,1);hi=torch.stack(hi,1);motion=torch.stack(motion,1)
  for kind in sorted(set(d['kinds'])):
   selected=torch.tensor([k==kind for k in d['kinds']]);key=f'{kind}:{d["mode"]}:{d["value"]:.1f}'
   g=groups.setdefault(key,dict(low=[],high=[],posture=[],n=0,stopped=0,fallen=0,invalid=0,window_s=0.,inflation=[],between=0.,speed=0.,omega=0.,body_names=d['body_names'],part_names=[d['body_names'][x[0]] for x in fp.parts]))
   g['low'].append(lo[selected].amin(0));g['high'].append(hi[selected].amax(0));g['posture'].append(posture[selected]);g['n']+=int(selected.sum());g['stopped']+=int(stopped[selected].sum());g['fallen']+=int(fallen[selected].sum());g['invalid']+=int(d['invalid'][selected].sum())
   g['between']=max(g['between'],float(motion[selected].max()));g['window_s']=max(g['window_s'],float((T-starts[selected].min())*d['dt']))
   speed=planar[int(starts[selected].min()):,selected].amax(1);omega=float(d['omega'][:,selected].abs().max());radius=float(torch.maximum(lo[selected].abs(),hi[selected].abs())[...,:2].norm(dim=-1).max())
   g['inflation'].append(p2_projected_inflation(speed,candidate,dt=d['dt'],radius_m=radius,angular_speed_max=omega))
   g['speed']=max(g['speed'],float(speed.max()));g['omega']=max(g['omega'],omega)
  print(file.name,flush=True)
summary={};models={}
for key,g in groups.items():
 low=torch.stack(g['low']).amin(0);high=torch.stack(g['high']).amax(0)
 p2=max(x['horizontal_m'] for x in g['inflation']);padding=torch.tensor([.05+p2+g['between'],.05+p2+g['between'],.05+.02+g['between']])
 models[key]=dict(centres=(low+high)/2,half=(high-low)/2+padding,postures=torch.cat(g['posture']),body_names=g['body_names'],part_names=g['part_names'],speed_max=g['speed'],omega_max=g['omega'],window_s=g['window_s'],qualified_geometry=bool(torch.isfinite(low).all()&torch.isfinite(high).all()),temporary_diagnostic_only=True,posture_format='collision_corners_sphere_centres')
 summary[key]=dict(trials=g['n'],stopped=g['stopped'],not_stopped=g['n']-g['stopped'],not_stopped_fraction=1-g['stopped']/g['n'],invalid=g['invalid'],fallen=g['fallen'],window_max_s=g['window_s'],p2_xy_m=p2,between_sample_m=g['between'],parts=len(low),raw_min=low.amin(0).tolist(),raw_max=high.amax(0).tolist())
a.output.mkdir(parents=True,exist_ok=True);torch.save(models,a.output/'models.pt')
(a.output/'summary.json').write_text(json.dumps(dict(status='diagnostic_temporary_not_deployment_guarantee',all_trials_included=True,quantile=None,margin_m=.05,source_files=sources,commands=summary),indent=2))
