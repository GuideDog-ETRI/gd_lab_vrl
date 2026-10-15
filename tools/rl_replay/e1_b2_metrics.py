"""UPDATE34 run-level flat-relative pitch; legacy summaries remain explicitly approximate."""
import json,math,statistics
from pathlib import Path

class FlatPitchGeometry:
    def __init__(self,base,gap):
        from gd_lab.mdp.gap_stair_v2 import _StairGeometry
        from gd_lab.mdp.platform_gap_terms import boarding_family_mask
        self.base=base;self.gap=gap;self.stair=_StairGeometry(base);self.family=boarding_family_mask
        self.total=0.;self.count=0
    def sample(self,ids,pitch):
        import torch
        b=self.base;p=b.scene['robot'].data.root_pos_w-b.scene.env_origins
        slots,_=self.gap.for_envs(b)
        distance=(slots.flatten(1)-p[:,0,None]).abs().min(-1).values
        gap_ok=~self.family(b) | (distance>=1.3)
        cols=b.scene.terrain.terrain_types.long();edges=self.stair.edges[cols]
        # Euclidean distance to square pyramid perimeters, including corners.
        xy=p[:,:2].abs();outside=(xy[:,None,:]-edges[:,:,None]).clamp_min(0).norm(dim=-1)
        inside=edges-xy.max(-1).values[:,None]
        sd=torch.where(inside>=0,inside,outside)
        sd=torch.where(torch.isfinite(edges),sd,float('inf')).min(-1).values
        safe=gap_ok & (~self.stair.is_stair[cols] | (sd>=1.3))
        # A scan confirms the edge-distant surface is level, excluding ramps.
        clean=b.observation_manager.compute_group('gast_clean')
        h=clean[:,:187]/5;valid=(clean[:,187:374]>.5)&torch.isfinite(h)
        span=torch.where(valid,h,-float('inf')).max(-1).values-torch.where(valid,h,float('inf')).min(-1).values
        safe &= (valid.float().mean(-1)>.9)&(span<=.02)&torch.isfinite(pitch)
        values=torch.rad2deg(pitch[ids][safe[ids]])
        self.total+=float(values.double().sum());self.count+=len(values)
    def result(self):
        return {'mean_deg':self.total/self.count if self.count else None,'samples':self.count,
          'definition':'same run all env policy samples; all local gap/stair edges >=1.3m; clean scan >90% valid and span <=.02m',
          'geometry_verified':True}

def summarize(data):
    records=data['approaches'];absolute=[];legacy_flat=[];legacy_delta=[]
    for r in records:
        p=r['pitch']['pre_0p7_0p4'];flat=r.get('flat_pitch_deg')
        if flat is not None:legacy_flat.append(flat)
        v=p.get('mean_absolute_deg')
        if v is None and p.get('mean_delta_deg') is not None and flat is not None:v=p['mean_delta_deg']+flat
        if v is not None:absolute.append(v)
        if p.get('mean_delta_deg') is not None:legacy_delta.append(p['mean_delta_deg'])
    avg=lambda x:statistics.mean(x) if x else None
    exact=data.get('update34_flat_pitch',{});flat=exact.get('mean_deg');prep=avg(absolute)
    delta=prep-flat if prep is not None and flat is not None else None
    sw=[s for r in records for s in r['crossing_swings'] if s['status']=='complete' and s['group']=='front']
    return {'condition':data['condition'],'student_iteration':data['student_iteration'],
      'approaches':len(records),'pitch_approaches':len(absolute),'reached':sum(r['reached'] for r in records),
      'clean':sum(r['clean'] for r in records),'foot_drop':sum(r['foot_drop'] for r in records),
      'front_apex_cm':avg([s['apex_cm'] for s in sw]),'prep_absolute_deg':prep,
      'flat_pitch_deg':flat,'flat_samples':exact.get('samples'), 'delta_pitch_deg':delta,
      'pass_delta_ge2':delta>=2 if delta is not None else None,
      'legacy_flat_episode_mean_deg':avg(legacy_flat),'legacy_delta_episode_mean_deg':avg(legacy_delta),
      'legacy_run_delta_proxy_deg':prep-avg(legacy_flat) if prep is not None and legacy_flat else None,
      'note':'prep averaged equally per approach, including failures up to existing metric cut; flat is same-run sample mean' if exact else
        'Exact UPDATE34 baseline unavailable: legacy JSON lacks flat sample counts/raw poses/all-edge masks. Proxy is not a pass/fail result.'}

def write_table(paths,output):
    rows=[dict(source=str(p),**summarize(json.loads(Path(p).read_text()))) for p in paths]
    output=Path(output).resolve()
    allowed=(Path(__file__).resolve().parents[2],Path('/home/user/gd_project/notes'))
    if not any(output.is_relative_to(p) for p in allowed):raise ValueError('unapproved output')
    output.mkdir(parents=True,exist_ok=True)
    (output/'pitch_summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    keys=['condition','approaches','pitch_approaches','reached','clean','foot_drop','front_apex_cm','prep_absolute_deg','flat_pitch_deg','delta_pitch_deg','pass_delta_ge2','legacy_flat_episode_mean_deg','legacy_run_delta_proxy_deg']
    lines=['# UPDATE34 pitch results','','Exact delta = approach-mean prep pitch − same-run geometry-verified flat sample mean. Legacy proxy is NOT the exact criterion.','','| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
    for r in rows:
        lines.append('| '+' | '.join('N/A' if r[k] is None else f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |')
    for r in rows:lines.extend(['',r['condition']+': '+r['note']])
    (output/'pitch_summary.md').write_text('\n'.join(lines)+'\n')
    return rows
