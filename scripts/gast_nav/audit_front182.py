import sys,json,math
from pathlib import Path
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from gd_lab.gast_nav.map_types import WIDE_GRID
from gd_lab.gast_nav.obs_terms import fit_observed_surface_reference
from gd_lab.gast_nav.support import make_seed_mask,grow_support
p=Path(sys.argv[1]);out=p.parent;d=torch.load(p,map_location='cpu',weights_only=False);torch.set_num_threads(2)
r=d['returns'];a=r.rays;raw=d['pre_filter_rays'];pts=a.origins+a.directions*torch.nan_to_num(r.distance_m,posinf=0)[...,None]
valid=r.valid&torch.isfinite(r.distance_m)&~r.self_hit
reference=d['contact'][...,2].masked_fill(~d['contact_valid'],torch.nan).nanmedian(-1).values
stats={'reference_z':reference.tolist(),'pose':d['pose'].tolist(),'sensors':{}}
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
for source,name in enumerate(('Ouster','Mid360','rear_assumed')):
 mask=a.source[0]==source;v=valid[0]&mask;pre=raw.source[0]==source;floor=v&(pts[0,:,2]-reference[0]).abs().lt(.03);forward=v&(pts[0,:,0]>0)
 dist=r.distance_m[0,v];directions=a.directions[0,mask]
 stats['sensors'][name]=dict(generated=int(pre.sum()),coverage_valid=int((pre&raw.coverage_valid[0]).sum()),cast=int(mask.sum()),valid_ratio=float(r.valid[0,mask].float().mean()),self_hits=int((mask&r.self_hit[0]).sum()),finite_returns=int(v.sum()),forward_returns=int(forward.sum()),floor_returns=int(floor.sum()),front_floor_1_5_to_3m=int((floor&(pts[0,:,0]>=1.5)&(pts[0,:,0]<=3)).sum()),origin_body=a.origins[0,mask][0].tolist(),direction_z_minmax=[float(directions[:,2].min()),float(directions[:,2].max())],hit_distance_quantiles=torch.quantile(dist,torch.tensor([0.,.1,.5,.9,1.])).tolist() if len(dist) else [])
 sel=mask.nonzero().flatten()[::max(1,int(mask.sum())//800)]
 end=a.origins[0,sel]+a.directions[0,sel]*torch.minimum(r.distance_m[0,sel],a.far_m[0,sel])[:,None]
 segments=torch.stack((a.origins[0,sel,:2],end[:,:2]),1).numpy();axs[source].add_collection(LineCollection(segments,colors='#999999',linewidths=.25,alpha=.5))
 axs[source].scatter(pts[0,v,0],pts[0,v,1],s=.3,c='#0055bb');axs[source].scatter(pts[0,floor,0],pts[0,floor,1],s=.4,c='#00aa44')
 axs[source].set(xlim=(-3,3),ylim=(-3,3),aspect='equal',xlabel='body x (m)',ylabel='body y (m)',title=name+'\nblue first hits; green measured floor')
 axs[source].axvline(0,color='black',linewidth=.5);axs[source].scatter(*a.origins[0,mask][0,:2],c='red',s=20)
fig.savefig(out/'head_rays182.png',dpi=150);plt.close(fig)
# Store actual world rays too, not just base-yaw: yaw rotates origins/directions.
pose=d['pose'];c=pose[:,3].cos();s=pose[:,3].sin();rot=torch.zeros(1,3,3);rot[:,0,0]=c;rot[:,0,1]=-s;rot[:,1,0]=s;rot[:,1,1]=c;rot[:,2,2]=1
world_orig=torch.einsum('nij,nrj->nri',rot,a.origins)+pose[:,None,:3];world_dir=torch.einsum('nij,nrj->nri',rot,a.directions)
torch.save(dict(origin_world=world_orig,direction_world=world_dir,source=a.source,valid=r.valid,distance_m=r.distance_m),out/'world_rays182.pt')
# Recompute fit and growth from only actual first returns, no GT filling.
settings=d['settings'];rad=settings['support_normal_radius_m'];keep=valid& (pts[...,0]>=-2.4-rad)&(pts[...,0]<=2.4+rad)&(pts[...,1]>=-2.1-rad)&(pts[...,1]<=2.1+rad)
points=pts[:,keep[0]];v=keep[:,keep[0]]
h,observed,normals=fit_observed_surface_reference(points,v,WIDE_GRID,settings)
seeds=make_seed_mask(d['contact'],d['contact_valid'],points,v,points[...,:2].norm(dim=-1)<=settings['support_body_radius_m'],grid=WIDE_GRID)
z,_,connected=grow_support(h,observed,normals,seeds,segment_type='FLAT',current_support_height=reference)
xy=WIDE_GRID.xy(device='cpu',dtype=torch.float32);front=(xy[...,0]>=1.5)&(xy[...,0]<=2.4)
for name,mask in [('all',torch.ones_like(front)),('front_1_5_to_2_4',front)]:
 stats[name]=dict(fit_observed=int((observed[0]&mask).sum()),normal_valid=int((torch.isfinite(normals[0]).all(-1)&mask).sum()),seeds=int((seeds[0]&mask).sum()),grown_support=int((torch.isfinite(z[0])&mask).sum()),producer_support=int((d['support'].observed[0]&mask).sum()),raw_free=int(((d['raw_frame'].state[0]==1)&mask).sum()))
torch.save(dict(height=h,observed=observed,normals=normals,seeds=seeds,grown=z,connected=connected),out/'fit_audit182.pt')
(out/'head_audit182.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
