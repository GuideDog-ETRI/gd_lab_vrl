"""Summarize completed E1 runs; refuse unmatched initial states/terrain."""
import json,sys,statistics
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
def main():
    if sys.argv[1]=='--b2':
        from e1_b2_metrics import write_table
        write_table([Path(p) for p in sys.argv[3:]],Path(sys.argv[2]));return
    root=Path(sys.argv[1]).resolve()
    if not root.is_relative_to(ROOT):raise ValueError('output outside runtime copy')
    groups=defaultdict(list)
    for p in root.glob('s*_*.json'):
        d=json.loads(p.read_text());groups[(d['seed'],p.name.split('_vx')[1].split('_')[0])].append(d)
    lines=['# E1 closed-loop results','','| seed | vx | condition | approaches | reached | clean/reached | slot dip | front apex cm | hind apex cm | pitch early deg | pitch prep deg | last front TD margin m | terminated | censored |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    avg=lambda vs: statistics.mean(vs) if vs else None
    fmt=lambda v:f'{v:.3f}' if v is not None else '—'
    for (seed,vx),ds in sorted(groups.items()):
        if len(ds)!=6:raise ValueError('need all six conditions per seed/speed')
        for key in ['initial_state_sha256','initial_terrain','student_sha256','teacher_sha256','camera_transport','transport_counts']:
            if any(d[key]!=ds[0][key] for d in ds):raise ValueError('unmatched '+key)
        for d in ds:
            rs=d['approaches'];sw=[s for r in rs for s in r['crossing_swings'] if s['status']=='complete']
            reach=sum(r['reached'] for r in rs);clean=sum(r['clean'] for r in rs)
            pitch=lambda key:avg([r['pitch'][key]['mean_delta_deg'] for r in rs if r['pitch'][key]['mean_delta_deg'] is not None])
            fields=[seed,vx,d['condition'],len(rs),reach,f'{clean}/{reach}',sum(r['foot_drop'] for r in rs),
                fmt(avg([s['apex_cm'] for s in sw if s['group']=='front'])),fmt(avg([s['apex_cm'] for s in sw if s['group']=='hind'])),
                fmt(pitch('pre_1p0_0p7')),fmt(pitch('pre_0p7_0p4')),
                fmt(avg([r['last_front_touchdown']['margin_m'] for r in rs if r['last_front_touchdown']])),
                sum(r['termination']=='terminated' for r in rs),sum(r['censored'] for r in rs)]
            lines.append('| '+' | '.join(map(str,fields))+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
