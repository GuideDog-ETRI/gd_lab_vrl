from pathlib import Path
import torch,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
root=Path(__file__).resolve().parents[2];out=root/'logs/gast/nav_v1/update182';before=root/'logs/gast/nav_v1/update182_before200_v1';after=root/'logs/gast/nav_v1/update182_after200_v1'
a=torch.load(before/'head_probe182_10.pt',weights_only=False);b=torch.load(after/'head_probe182_10.pt',weights_only=False)
fig,axs=plt.subplots(2,2,figsize=(11,10),layout='constrained');extent=(-2.45,2.45,-2.15,2.15)
for col,(d,label) in enumerate(((a,'Before'),(b,'After'))):
 axs[0,col].imshow(d['raw_frame'].state[0],origin='lower',extent=extent,cmap=ListedColormap(['#bbbbbb','#4daf4a','#e41a1c']),vmin=0,vmax=2)
 axs[1,col].imshow(d['support'].observed[0],origin='lower',extent=extent,cmap=ListedColormap(['#555555','#377eb8']),vmin=0,vmax=1)
 for row in (0,1):
  axs[row,col].set(title=label+(' head occupancy' if row==0 else ' observed support'),xlabel='body x (m)',ylabel='body y (m)',aspect='equal')
  axs[row,col].add_patch(Rectangle((-.85,-.55),1.7,1.1,fill=False,edgecolor='blue',linestyle='--'))
fig.suptitle('env0, capture10 (~1 s): actual first returns, before terrain fusion\ngray UNKNOWN / green FREE / red occupied; no missing-cell fill')
fig.savefig(out/'head_grid_before_after182.png',dpi=160);plt.close(fig)
comparison=dict(pose_max_difference=float((a['pose']-b['pose']).abs().max()),ray_origin_max_difference=float((a['returns'].rays.origins-b['returns'].rays.origins).abs().max()),ray_direction_max_difference=float((a['returns'].rays.directions-b['returns'].rays.directions).abs().max()))
# Lower-camera stripe: raw native validity before any supervisor warp.
t=b['terrain'];p=torch.load(after/'lower_probe182_100.pt',weights_only=False);pose=b['pose'];c=pose[:,3].cos();s=pose[:,3].sin()
def local(world):
 d=world-pose[:,None,None,None,:3];x=c[:,None,None,None]*d[...,0]+s[:,None,None,None]*d[...,1];y=-s[:,None,None,None]*d[...,0]+c[:,None,None,None]*d[...,1]
 return torch.stack((x,y,d[...,2]),-1)
clean=p['scene_only_distance'];world=p['origins'][:,:,None,None,:]+p['dirs']*torch.nan_to_num(clean,posinf=0)[...,None];points=local(world)
H,W=clean.shape[-2:];yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij');K=p['K'];length=torch.sqrt(((xx-K[0,2])/K[0,0])**2+((yy-K[1,2])/K[1,1])**2+1)
optical=clean/length;floor=(points[...,2]+pose[:,2,None,None,None]).abs()<.02
candidate=torch.isfinite(clean)&(optical>=.15)&(optical<5.)&floor
ix=torch.floor((points[...,0]+.85)/.1).long();iy=torch.floor((points[...,1]+.55)/.1).long();inside=(ix>=0)&(ix<17)&(iy>=0)&(iy<11)
def counts(mask):
 out=torch.zeros(187,dtype=torch.long);idx=(iy*17+ix).flatten();ok=(inside&mask).flatten();out.scatter_add_(0,idx.clamp(0,186),ok.long());return out.reshape(11,17)
potential=counts(candidate);selfhit=counts(candidate&p['self_hit']);usable=counts(candidate&p['valid']&~p['self_hit']);invalid=counts(candidate&~p['valid'])
comparison['lower_stripes']={}
for name,rows in [('negative',[0,1]),('positive',[9,10])]:
 comparison['lower_stripes'][name]=dict(native_valid_cells=int(t.valid[0,rows].sum()),native_total_cells=34,potential_scene_floor_rays=int(potential[rows].sum()),confirmed_self_hit_rays=int(selfhit[rows].sum()),invalid_or_filtered_rays=int(invalid[rows].sum()),usable_floor_rays=int(usable[rows].sum()),cells_no_scene_ray=int((potential[rows]==0).sum()),cells_with_usable_floor_ray=int((usable[rows]>0).sum()))
fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained')
for ax,z,title in zip(axs,[t.valid[0],potential>0,selfhit>0,usable>0],['Native valid BEFORE warp','Scene-only floor coverage (evaluation)','Confirmed leg/self-hit pixels','Usable floor-return coverage']):
 ax.imshow(z,origin='lower',extent=(-.85,.85,-.55,.55),vmin=0,vmax=1,cmap='Blues');ax.set(title=title,xlabel='body x (m)',ylabel='body y (m)',aspect='equal');ax.axhline(.35,c='red',ls='--');ax.axhline(-.35,c='red',ls='--')
fig.savefig(out/'lower_stripes182.png',dpi=160);plt.close(fig)
(out/'comparison182.json').write_text(json.dumps(comparison,indent=2));print(json.dumps(comparison,indent=2))
