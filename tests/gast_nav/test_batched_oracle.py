import math
import torch
from gd_lab.gast_nav.batched_oracle import BatchedOracle
from gd_lab.gast_nav.terrains import build_scene,KINDS
from gd_lab.gast_nav.scene_types import project_route
from gd_lab.gast_nav.scene_config import load_scene_config
from gd_lab.gast_nav.feasibility import passage_feasible,turn_feasible
from gd_lab.gast_nav.scene_geometry import floor_height,point_box_distance

def test_batched_oracle_matches_scalar_nominal_geometry():
    settings=load_scene_config()
    tiles=[build_scene(k,d,seed=16201)[2] for k in KINDS for d in (0.,.1,.5,.9)]
    oracle=BatchedOracle(tiles,settings,device='cpu')
    ids=torch.arange(len(tiles)).repeat_interleave(7)
    xy=torch.tensor([((1-f)*t.route_polyline[0][0]+f*t.route_polyline[-1][0],(1-f)*t.route_polyline[0][1]+f*t.route_polyline[-1][1]+.037) for t in tiles for f in (0.,.1,.2,.4,.6,.9,1.)])
    for turns in (False,True):
        result=oracle.evaluate(ids,xy,torch.full((len(ids),),turns),torch.full((len(ids),),.1),swept_width=.5,turn_clear_radius=.65,sdf_clip=2.)
        for j,(i,p) in enumerate(zip(ids.tolist(),xy.tolist())):
            t=tiles[i];s,offset,axis=project_route(t,p)
            assert abs(float(result['s'][j])-s)<1e-5
            assert abs(float(result['offset'][j])-offset)<1e-5
            assert abs(float(result['axis'][j])-axis)<1e-5
            ref=turn_feasible(t,p,margin_m=.1,config=settings) if turns else passage_feasible(t,p,margin_m=.1,config=settings)
            assert bool(result['feasible'][j])==ref.feasible
            z=floor_height(t,p);radius=.65 if turns else .25
            clearance=min([2.]+[point_box_distance(p,b)-radius for b in t.obstacles if b.kind!='step' and b.center[2]+b.half[2]>z])
            assert abs(float(result['clearance'][j])-clearance)<1e-5

def test_reset_held_rows_fail_closed_without_corrupting_stored_observations():
    from gd_lab.gast_nav.obs_terms import invalidate_held_rows
    wide=torch.rand(3,27,43,49);near=torch.rand(3,3,25,37)
    old_wide=wide.clone();old_near=near.clone()
    updated,fine=invalidate_held_rows(wide,near,torch.tensor([1]))
    assert torch.equal(wide,old_wide) and torch.equal(near,old_near)
    assert torch.equal(updated[[0,2]],wide[[0,2]]) and torch.equal(fine[[0,2]],near[[0,2]])
    assert (updated[1,0::3]==0).all() and (updated[1,1::3]==0).all() and (updated[1,2::3]==1).all()
    assert (fine[1,:2]==0).all() and (fine[1,2]==1).all()
