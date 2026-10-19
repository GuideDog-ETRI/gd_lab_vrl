import torch
from gd_lab.gast_nav.certification165 import CertificateHistory,unknown_risk
from gd_lab.gast_nav.map_types import Grid


def test_r2_components_ratio_and_off():
    required=torch.ones(1,10,10,dtype=torch.bool);known=required.clone()
    known[0,1,1]=False
    assert unknown_risk(known,required,resolution=.05)[0].item()
    assert not unknown_risk(known,required,resolution=.05,relaxed=True)[0].item()
    known[0,1,1:5]=False  # connected 1x4 must not slip through a 2x2-only check
    assert unknown_risk(known,required,resolution=.05,relaxed=True)[0].item()
    known=required.clone();known[:,::3,::3]=False
    assert unknown_risk(known,required,resolution=.05,relaxed=True)[0].item()  # >10% separated holes
    assert unknown_risk(known,required,resolution=.1,relaxed=True)[0].item()


def test_r1_expiry_epoch_and_no_refresh_from_unknown():
    grid=Grid(5,5,.05,-.1,-.1);cache=CertificateHistory(grid,2.)
    known=torch.ones(1,5,5,dtype=torch.bool);missing=~known;height=torch.zeros(1,5,5);pose=torch.zeros(1,4);eligible=torch.ones(1,dtype=torch.bool)
    original=known.clone();cache.update(known,height,pose,0.,eligible)
    assert cache.update(missing,height,pose,1.9,eligible).all()
    assert not cache.update(missing,height,pose,2.01,eligible).any()
    assert torch.equal(known,original)
    cache.update(known,height,pose,3.,eligible);cache.reset([0])
    assert not cache.update(missing,height,pose,3.1,eligible).any()


def test_r1_dynamic_rows_and_changed_floor_fail_closed():
    grid=Grid(3,3,.05,-.05,-.05);cache=CertificateHistory(grid,2.)
    known=torch.ones(1,3,3,dtype=torch.bool);height=torch.zeros(1,3,3);pose=torch.zeros(1,4);yes=torch.tensor([True])
    cache.update(known,height,pose,0.,yes)
    assert not cache.update(~known,height+.2,pose,.1,yes).any()
    assert not cache.update(~known,height,pose,.2,~yes).any()


def test_r1_does_not_fill_never_observed_wall_shadow():
    grid=Grid(5,3,.05,-.1,-.05);cache=CertificateHistory(grid,2.)
    known=torch.ones(1,3,5,dtype=torch.bool);known[:,:,3:]=False
    height=torch.zeros(1,3,5);pose=torch.zeros(1,4);yes=torch.tensor([True])
    cache.update(known,height,pose,0.,yes)
    result=cache.update(torch.zeros_like(known),height,pose,1.,yes)
    assert not result[:,:,3:].any()
    assert unknown_risk(result,torch.ones_like(result),resolution=.05,relaxed=True)[0].item()


import pytest
@pytest.mark.parametrize('r1,r2', [(True,False),(False,True),(True,True)])
def test_thin_occupied_pillar_remains_clearance_hazard(r1,r2):
    from gd_lab.gast_nav.near_clearance import clearance_from_state
    from gd_lab.gast_nav.supervisor import required_observed
    from gd_lab.gast_nav import schema as S
    grid=Grid(7,7,.05,-.15,-.15)
    state=torch.full((1,3,7,7),S.FREE,dtype=torch.uint8);state[:,0,3,3]=S.OCCUPIED
    original=state.clone();support=torch.ones(1,7,7,dtype=torch.bool)
    before=clearance_from_state(state,grid=grid)
    known=required_observed(state,support,['B1','B2'])
    if r1:known=CertificateHistory(grid).update(known,torch.zeros(1,7,7),torch.zeros(1,4),0.,torch.tensor([True]))
    unknown_risk(known,support,resolution=.05,relaxed=r2)
    after=clearance_from_state(state,grid=grid)
    assert torch.equal(state,original) and torch.equal(before,after)
    assert (after[:,:1]<.1).any()  # independent unchanged occupied guard


@pytest.mark.parametrize('r1,r2', [(True,False),(False,True),(True,True)])
def test_real_first_hit_wall_person_counterexample(r1,r2):
    from dataclasses import replace
    from test_maps import full_rays
    from gd_lab.gast_nav.map_types import DynamicCapsules
    from gd_lab.gast_nav.ray_geometry import resolve_first_hits
    from gd_lab.gast_nav.obstacle_map import ray_evidence
    from gd_lab.gast_nav.supervisor import required_observed
    grid=Grid(25,1,.05,-.3,0.)
    rays=full_rays();rays=replace(rays,distances=torch.full_like(rays.distances,.5))
    person=DynamicCapsules(torch.tensor([[[.8,0.,0.]]]),torch.tensor([[[.8,0.,1.8]]]),torch.tensor([[.2]]))
    state,_=ray_evidence(resolve_first_hits(rays,capsules=person),torch.zeros(1,1,25),grid=grid)
    original=state.clone();support=torch.ones(1,1,25,dtype=torch.bool)
    known=required_observed(state,support,['B1','B2'])
    if r1:
        cache=CertificateHistory(grid);cache.update(known,torch.zeros(1,1,25),torch.zeros(1,4),0.,torch.tensor([True]))
        known=cache.update(known,torch.zeros(1,1,25),torch.zeros(1,4),.5,torch.tensor([True]))
    assert unknown_risk(known,support,resolution=.05,relaxed=r2)[0].item()
    assert torch.equal(state,original) and (state[:,:,:,12:]==0).all()


@pytest.mark.parametrize('r1,r2', [(True,False),(False,True),(True,True)])
def test_document_new_blind_pillar_safety_limit(r1,r2):
    # Deliberately exposes a relaxation limitation, NOT a safety-pass fixture.
    # A previously seen static pillar becomes occluded and occupied TTL expires.
    from gd_lab.gast_nav.near_clearance import clearance_from_state
    from gd_lab.gast_nav.supervisor import required_observed
    from gd_lab.gast_nav import schema as S
    grid=Grid(7,7,.05,-.15,-.15);support=torch.ones(1,7,7,dtype=torch.bool)
    old=torch.full((1,3,7,7),S.FREE,dtype=torch.uint8);old[:,0,3,3]=S.OCCUPIED
    current=old.clone();current[:,:,3,3]=S.UNKNOWN
    known=required_observed(current,support,['B1','B2'])
    if r1:
        cache=CertificateHistory(grid);cache.update(required_observed(old,support,['B1','B2']),torch.zeros(1,7,7),torch.zeros(1,4),0.,torch.tensor([True]))
        known=cache.update(known,torch.zeros(1,7,7),torch.zeros(1,4),.6,torch.tensor([True]))
    unknown_guard=unknown_risk(known,support,resolution=.05,relaxed=r2)[0]
    occupied_guard=(clearance_from_state(current,grid=grid)<.1).any()
    assert not unknown_guard.any() and not occupied_guard  # requested rules can miss this hazard


def test_carried_occupied_certificate_does_not_refresh_age():
    grid=Grid(3,3,.05,-.05,-.05);cache=CertificateHistory(grid,2.)
    known=torch.ones(1,3,3,dtype=torch.bool);height=torch.zeros(1,3,3);pose=torch.zeros(1,4);yes=torch.tensor([True])
    cache.update(known,height,pose,0.,yes)
    cache.update(known,height,pose,.4,yes,fresh=~known)
    assert not cache.update(~known,height,pose,2.1,yes).any()
