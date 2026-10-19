import json,collections,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'logs/gast/nav_v1/update181';summary={}
for mode in ('off','unknown','support','occupied','swept','health','all'):
 run=root/f'logs/gast/nav_v1/update181_{mode}{600 if mode=="off" else 200}_v1'
 f=run/'tick_decisions.jsonl'
 if not f.exists():continue
 records={v['tick']:v for v in (json.loads(l) for l in f.read_text().splitlines())}
 rows=[r for v in records.values() for r in v['rows']]
 max_tick=max(r['tick'] for r in rows);start=300 if mode=='off' else 100
 selected=[r for r in rows if r['tick']>=start and r['requested_mode'] in ('NAV','TURN') and r['requested_go'] and r['phase'] not in ('PRE_BRAKE','LANDING_ENTERED','DWELL')]
 def summarize(rs):
  n=len(rs)
  return dict(ticks=n,hold_fraction=sum(r['controller']=='HOLD_3906' for r in rs)/max(n,1),nonpolicy_fraction=sum(r['controller']!='NAV_POLICY' for r in rs)/max(n,1),observed_fraction=sum(r['observed'] for r in rs)/max(n,1),effective_go_fraction=sum(r['effective_go'] for r in rs)/max(n,1),nav1_invalid=sum(not r['nav1_valid'] for r in rs),reasons=dict(collections.Counter(r['reason'] or 'OK' for r in rs)),controllers=dict(collections.Counter(r['controller'] for r in rs)),handshake=dict(collections.Counter(r['handshake'] for r in rs)),fsm=dict(collections.Counter(r['phase'] for r in rs)))
 matched=[r for r in rows if 100<=r['tick']<200 and r['requested_mode'] in ('NAV','TURN') and r['requested_go'] and r['phase'] not in ('PRE_BRAKE','LANDING_ENTERED','DWELL')]
 item=dict(matched_1_2s=summarize(matched),unique_ticks=len(records),log_records=len(f.read_text().splitlines()),mode=mode,last_tick=max_tick,window_start=start,eligible=summarize(selected),all_window=summarize([r for r in rows if r['tick']>=start]),groups={})
 for key in sorted({(r['kind'],r['requested_mode']) for r in selected}):item['groups']['/'.join(key)]=summarize([r for r in selected if (r['kind'],r['requested_mode'])==key])
 pm=run/'producer_metrics.jsonl'
 if pm.exists():
  sizes=[v['required_cells'] for v in map(json.loads,pm.read_text().splitlines()) if 'required_cells' in v]
  if sizes:item['required_cells_all_envs']=dict(min=min(sizes),max=max(sizes),mean=sum(sizes)/len(sizes))
 item['outside_tick_fraction']=sum(r['sweep_outside'] for r in selected)/max(len(selected),1)
 result=run/'result.json'
 if result.exists():
  r=json.loads(result.read_text());item.update(status=r['status'],progress_mean_all_ticks=r['reward_means'].get('progress'),peak_reserved_bytes=r['peak_reserved_bytes'],wall_s=r['wall_s'])
 summary[mode]=item
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
