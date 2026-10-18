"""CPU-only summaries and plots from recorded UPDATE162 results."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'logs/gast/nav_v1';out=base/'update162_analysis';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
smoke=[]
for rank in range(3):
 rows=[json.loads(l) for l in (base/f'update162_remote_smoke/rank{rank}_metrics.jsonl').read_text().splitlines()]
 smoke.append(dict(rank=rank,**{k:statistics.mean(r[k] for r in rows) for k in ('iteration_s','rollout_s','update_s','step_per_s','map_wall_ms','physics_wall_ms')},peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows)))
 axes[0].plot([r['iteration'] for r in rows],[r['reward_mean'] for r in rows],marker='o',label=f'rank {rank}')
 axes[1].plot([r['iteration'] for r in rows],[r['iteration_s'] for r in rows],marker='o',label=f'rank {rank}')
axes[0].set(title='Smoke only: reward',xlabel='Iteration',ylabel='Mean reward');axes[1].set(title='Smoke only: wall time',xlabel='Iteration',ylabel='Seconds');axes[0].legend();fig.tight_layout();fig.savefig(out/'smoke_curves.png',dpi=160);plt.close(fig)
v0=json.loads((base/'update162_v0_02/result.json').read_text());episodes=v0['episodes'];same=json.loads((base/'update162_v0_02/same_pose.json').read_text());by={e['env']:e for e in episodes}
vrows=[];srows=[]
for kind in sorted({r['kind'] for r in episodes}):
 samples=[s for s in same if by[s['env']]['kind']==kind]
 srows.append(dict(kind=kind,**{k:statistics.mean(s[k] for s in samples) for k in ('gt_valid','visible_valid','h_abs_delta','ir_abs_delta','latent_l2','action_l2')}))
 for mode in ('gt','visible'):
  for structures in (False,True):
   rows=[e for e in episodes if e['kind']==kind and e['mode']==mode and e['structures']==structures]
   clear=[e['min_root_disc_clearance_proxy_m'] for e in rows if e['min_root_disc_clearance_proxy_m'] is not None]
   vrows.append(dict(kind=kind,mode=mode,structures=structures,n=len(rows),falls=sum(e['fall'] for e in rows),stalls=sum(e['stall'] for e in rows),complete=sum(e['complete'] for e in rows),mean_progress_m=statistics.mean(e['progress_m'] for e in rows),min_clearance_proxy_m=min(clear) if clear else None,max_lateral_m=max(e['max_abs_lateral_m'] for e in rows)))
fig,axes=plt.subplots(1,2,figsize=(12,4));labels=[r['kind'].replace('nav_','')+' '+r['mode'] for r in vrows if r['structures']]
for j,metric in enumerate(('stalls','complete')):
 for b,shift in ((False,-.2),(True,.2)):
  rows=[r for r in vrows if r['structures']==b];axes[j].bar([i+shift for i in range(len(rows))],[100*r[metric]/r['n'] for r in rows],width=.4,label='structures' if b else 'bare')
 axes[j].set_xticks(range(len(labels)),labels,rotation=65,ha='right');axes[j].set_ylabel('% of 8 scenes');axes[j].set_title('V0 '+metric);axes[j].legend()
fig.tight_layout();fig.savefig(out/'v0_outcomes.png',dpi=160);plt.close(fig)
profile=[]
for p in sorted(base.glob('update162_profile_*/result.json')):
 r=json.loads(p.read_text());profile.append(dict(run=p.parent.name,status=r['status'],envs=r['envs'],steps=r.get('steps',0),wall_s=r.get('wall_s'),peak_reserved_bytes=r.get('peak_reserved_bytes'),error=r.get('error')))
checkpoint=torch.load(base/'update162_remote_pilot/stopped_timeout.pt',map_location='cpu',weights_only=False)
summary=dict(smoke=smoke,v0=vrows,same_pose=srows,v0_comparisons=v0['comparisons'],profiles=profile,stopped_checkpoint_iteration=checkpoint['iteration'],stopped_optimizer_state_entries=len(checkpoint['optimizer_state_dict']['state']))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
