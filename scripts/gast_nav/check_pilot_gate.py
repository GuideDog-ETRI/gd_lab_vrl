#!/usr/bin/env python3
"""Read-only UPDATE163 gate; never launches or changes training."""
import argparse,json,math,statistics
from pathlib import Path

def assess(log,expected=200):
    log=Path(log);rows=[];reasons=[];completed=[]
    for rank in range(3):
        path=log/f'rank{rank}_metrics.jsonl'
        records=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        rows.append(records)
        result=log/f'rank{rank}_result.json'
        completed.append(json.loads(result.read_text()) if result.exists() else {})
        if completed[-1].get('status') not in (None,'completed'):reasons.append(f'rank{rank}: failed result')
        if any(v.get('nan') is not False or v.get('frozen_checksums_ok') is not True for v in records):reasons.append(f'rank{rank}: integrity')
        if any(v.get('iteration')!=i+1 for i,v in enumerate(records)):reasons.append(f'rank{rank}: iteration order')
    common=min(map(len,rows))
    if any(len({rows[r][i]['ddp_fingerprint'] for r in range(3)})!=1 for i in range(common)):reasons.append('DDP fingerprint mismatch')
    all_rows=[v for records in rows for v in records]
    peak=max((v['peak_reserved_bytes'] for v in all_rows),default=0)/2**30
    medians=[statistics.median(v['iteration_s'] for v in records) if records else None for records in rows]
    total=sum(v['num_envs']*v['horizon'] for v in all_rows)
    hold=sum(v['controller_fractions']['1']*v['num_envs']*v['horizon'] for v in all_rows)/total if total else None
    if peak>60:reasons.append('peak reserved >60 GiB')
    for records in rows:
        if any(a['iteration_s']>60 and b['iteration_s']>60 for a,b in zip(records,records[1:])):reasons.append('consecutive >60s')
    complete=all(len(records)==expected for records in rows) and all(v.get('status')=='completed' and v.get('iteration')==expected and v.get('optimizer_steps')==expected*20 for v in completed)
    if complete:
        if any(v>40 for v in medians):reasons.append('median >40s')
        if not 0<hold<1:reasons.append('HOLD fraction degenerate')
        for r,records in enumerate(rows):
            for v in records:
                if v['num_envs']!=(1366 if r==0 else 1365) or v['horizon']!=100:reasons.append('not full-size pilot');break
                if any(not math.isfinite(x) for x in v['rewards'].values()):reasons.append('nonfinite reward');break
                counters=v['person_contact']
                if any(not math.isfinite(counters[k]) or counters[k]<0 for k in ('person_contact','invalid','episodes_started')):reasons.append('invalid contact counters');break
    return dict(status='fail' if reasons else 'pass' if complete else 'incomplete',
        completed_iterations=common,median_s=medians,peak_reserved_gib=peak,hold_fraction=hold,reasons=sorted(set(reasons)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('log',type=Path);args=parser.parse_args()
    result=assess(args.log);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='pass' else 1 if result['status']=='fail' else 2)
