#!/usr/bin/env python3
"""Offline diagnostic counterfactuals ONLY; never feeds a supervisor or policy.

Read a temporal_probe_*.pt from the authorized local diagnostic. Run after the
UPDATE169 GPU proof is integrated; current reference is deliberately slow.
"""
import argparse,json
from dataclasses import replace
from pathlib import Path
import torch
from gd_lab.gast_nav import schema as S
from gd_lab.gast_nav.map_types import WIDE_GRID,NEAR_GRID
from gd_lab.gast_nav.near_clearance import extended_near_grid
from gd_lab.gast_nav.obstacle_map import ray_evidence
from gd_lab.gast_nav.free_evidence167 import limit_by_current_returns

def variants(m):
    yield 'actual',m
    yield 'latest_capture_only',replace(m,captures=m.captures[:1],travel_m=m.travel_m[:,:1])
    yield 'no_history_or_between_ticks',replace(m,settings=replace(m.settings,historical_obstacle_speed_bound_mps=None))
    # Counterfactual upper limits; NOT approved operating settings.
    p2=replace(m.settings,pose_sigma=0.,pose_slip_fraction=0.,pose_delay_s=0.)
    yield 'diagnostic_no_pose_erosion',replace(m,settings=p2)
    yield 'diagnostic_no_erosion',replace(m,settings=replace(p2,range_error_floor_m=0.,range_error_fraction=0.,sensor_angle_error_deg=0.,support_error_m=0.))

def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    data=torch.load(a.snapshot,map_location=a.device,weights_only=False)
    result={'warning':'Counterfactual diagnostic only; overlapping effects are not additive causal contributions.',
        'inventory':data['producer_stats'],'envelope_unavailable':int((~data['envelope_available']).sum()),'variants':{}}
    m=data['measurements'][0];rr=data['resolved']
    result['frustum_valid_before_clip']=sum(int(c.frusta.volume_valid.sum()) for c in m.captures)
    result['frustum_valid_after_clip']={}
    for label,rays in [('all',rr),('self_only',replace(rr,occupied_hit=torch.zeros_like(rr.occupied_hit))),('occupied_only',replace(rr,self_hit=torch.zeros_like(rr.self_hit)))]:
        clipped=limit_by_current_returns(m,rays)
        result['frustum_valid_after_clip'][label]=sum(int(c.frusta.volume_valid.sum()) for c in clipped.captures)
    result['travel_bound_max_m']=float(m.travel_m.max())
    for i,grid in enumerate((WIDE_GRID,extended_near_grid()[0])):
        for label,m in variants(data['measurements'][i]):
            state,_,age=ray_evidence(data['resolved'],data['supports'][i],grid=grid,measurement=m,certified=True,return_age=True)
            if i:
                pad=extended_near_grid()[1];state=state[...,pad:pad+NEAR_GRID.ny,pad:pad+NEAR_GRID.nx]
            if label=='actual' and ('expected_near' if i else 'expected_wide') in data:
                assert torch.equal(state,data['expected_near' if i else 'expected_wide']), 'live/replay certificate mismatch'
            required=data['required_near' if i else 'required_wide']
            known=(state[:,:2]!=S.UNKNOWN).all(1)
            missing=(required&~known).flatten(1).sum(1)
            result['variants'].setdefault(label,{})['near' if i else 'wide']={
                'observed_envs':int((missing==0).sum()),'missing_cells_by_env':missing.tolist(),
                'required_cells_by_env':required.flatten(1).sum(1).tolist(),
                'band_unknown_required':[int(((state[:,b]==S.UNKNOWN)&required).sum()) for b in range(3)],
                'band_free_required':[int(((state[:,b]==S.FREE)&required).sum()) for b in range(3)],
                'band_occupied_required':[int(((state[:,b]==S.OCCUPIED)&required).sum()) for b in range(3)]}
    a.snapshot.with_suffix('.analysis.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
