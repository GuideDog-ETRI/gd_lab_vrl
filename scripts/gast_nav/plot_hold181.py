"""Read-only snapshot explanation of t=3 s corridor gate (UPDATE180 baseline)."""
import sys,json
from pathlib import Path
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
run=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
d=torch.load(run/'temporal_probe_299.pt',map_location='cpu',weights_only=False)
e=torch.load(run/'evidence_299.pt',map_location='cpu',weights_only=False)
i=next(i for i,r in enumerate(e['rows']) if r['kind']=='nav_corridor' and r['requested_go'])
o=d['supervisor_observation179'];dg=d['swept_diagnostics'];req=dg['required'][i];body=dg['required_before_exclusion'][i]&~req
fresh=torch.isfinite(o.age_s[i])&(o.age_s[i]>=0)&(o.age_s[i]<=.5)
free=(o.state[i]==1)&fresh;occ=(o.state[i]==2)&fresh;unk=~free&~occ
support=torch.isfinite(o.support[i])&torch.isfinite(o.support_age_s[i])&(o.support_age_s[i]>=0)&(o.support_age_s[i]<=.5)
gap=o.walkable_gap[i]&(o.walkable_gap_age_s[i]>=0)&(o.walkable_gap_age_s[i]<=.5)
terrain=torch.zeros_like(req);s=d['terrain_assumed_free180'];ix=(s['ids']==i).nonzero().flatten()
if len(ix):terrain=s['mask'][int(ix[0])]
state=torch.where(occ,2,torch.where(free,1,0));blocked=req&(~free|~(support|gap))
fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained')
extent=(-2.45,2.45,-2.15,2.15);x=np.arange(req.shape[1])*.1-2.4;y=np.arange(req.shape[0])*.1-2.1
items=[(state,'Occupancy: gray unknown / green FREE / red occupied',ListedColormap(['#bdbdbd','#4daf4a','#e41a1c']),2),(support.int()+2*gap.int(),'Traversable: dark missing / blue support / cyan verified gap',ListedColormap(['#555555','#377eb8','#00cccc','#00cccc']),3),(terrain.int(),'Terrain-only FREE (yellow); head UNKNOWN allowed in ROI',ListedColormap(['#dddddd','#ffcc00']),1),(blocked.int(),'Blocking required cells (red)',ListedColormap(['#dddddd','#e41a1c']),1)]
for ax,(z,title,cmap,vmax) in zip(axs.flat,items):
 ax.imshow(z.numpy(),origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
 ax.contour(x,y,req.numpy(),levels=[.5],colors='black',linewidths=1.4)
 ax.contour(x,y,body.numpy(),levels=[.5],colors='#ff00ff',linewidths=1.3)
 ax.add_patch(Rectangle((-.85,-.55),1.7,1.1,fill=False,edgecolor='#0044ff',linestyle='--',linewidth=1.5))
 ax.set(xlabel='body x (m)',ylabel='body y (m)',title=title,xlim=(-1.5,1.5),ylim=(-1.4,1.4),aspect='equal')
fig.suptitle(f'UPDATE180 baseline: corridor env {i}, tick299 (~3 s)\nblack=required; magenta=current-body excluded; blue dashed=terrain ROI')
fig.savefig(out/'corridor_t3.png',dpi=170);plt.close(fig)
summary=dict(env_id=i,tick=299,row=e['rows'][i],required=int(req.sum()),unknown=int((req&unk).sum()),occupied=int((req&occ).sum()),missing_support=int((req&~support).sum()),missing_traversable=int((req&~(support|gap)).sum()),blocked=int(blocked.sum()),terrain_only=int((req&terrain).sum()),body_excluded=int(body.sum()),blocked_xy=[[round(float(x[j]),3),round(float(y[k]),3)] for k,j in blocked.nonzero().tolist()])
(out/'corridor_t3.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('row','blocked_xy')}))
