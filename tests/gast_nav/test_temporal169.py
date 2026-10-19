import torch
import pytest
from gd_lab.gast_nav.temporal169 import TemporalEvidence169,current_support169,transform_returns169
from gd_lab.gast_nav.free_evidence167 import FreeEvidenceConfig,volume_free_proof
from gd_lab.gast_nav.map_config import load_config
from tests.gast_nav.test_free_evidence167 import prism,GRID,rays

CFG=dict(context='training_static_c',historical_obstacle_speed_bound_mps=0.,max_captures=6,linear_speed_bound_mps=1.,yaw_rate_bound_radps=1.5)

def setup():
    h=TemporalEvidence169(1,'cpu',CFG);p=torch.zeros(1,4);t=torch.tensor([1000]);e=torch.tensor([0]);s=torch.zeros(1,1,1)
    h.observe(p,torch.zeros(1),torch.zeros(1),0.)
    h.push(prism(),p,(s,s),t,e,dynamic_enabled=torch.tensor([False]))
    return h,p,t,e,s

def test_between_capture_proof_and_expiry_age_not_refreshed():
    h,p,t,e,s=setup()
    h.observe(p,torch.zeros(1),torch.zeros(1),.01)
    m=h.measurement(0,p,t+10,e)
    proof=volume_free_proof(s,m,grid=GRID)
    assert proof.free.all() and torch.allclose(proof.age_s,torch.full_like(proof.age_s,.1))
    assert not volume_free_proof(s,h.measurement(0,p,t+501,e),grid=GRID).free.any()
    assert FreeEvidenceConfig().historical_obstacle_speed_bound_mps is None

def test_return_to_origin_keeps_integrated_path():
    h,p,t,e,s=setup()
    q=p.clone();q[:,0]=.05
    h.observe(q,torch.tensor([.5]),torch.zeros(1),.1)
    h.observe(p,torch.tensor([.5]),torch.zeros(1),.1)
    m=h.measurement(0,p,t+200,e)
    assert torch.allclose(m.travel_m,torch.tensor([[.2]]))

def test_reset_row_cannot_reuse_old_epoch_or_support():
    h,p,t,e,s=setup();h.reset(torch.tensor([0]));e=e+1
    m=h.measurement(0,p,t+10,e)
    assert not volume_free_proof(s,m,grid=GRID).free.any()
    support=current_support169(h.histories[0].captures[0],p,t+10,e,GRID,load_config())
    assert torch.isnan(support).all()

def test_violation_invalidates_old_capture_even_after_recovery():
    h,p,t,e,s=setup();h.observe(p,torch.tensor([1.1]),torch.zeros(1),.01)
    assert not volume_free_proof(s,h.measurement(0,p,t+10,e),grid=GRID).free.any()
    h.push(prism(),p,(s,s),t+100,e,dynamic_enabled=torch.tensor([False]))
    assert not h.histories[0].captures[0].frusta.volume_valid.any()
    h.observe(p,torch.zeros(1),torch.zeros(1),.1)
    h.push(prism(),p,(s,s),t+200,e,dynamic_enabled=torch.tensor([False]))
    assert h.histories[0].captures[0].frusta.volume_valid.all()
    assert h.histories[0].captures[2].epoch.item()==-1

def test_displacement_and_yaw_violation_close_evidence():
    for axis,delta in [(0,.2),(3,.3)]:
        h,p,t,e,s=setup();p[:,axis]=delta
        h.observe(p,torch.zeros(1),torch.zeros(1),.1)
        assert h.invalid.all()

def test_cylinder_rejected_and_cache_bounded():
    h,p,t,e,s=setup()
    with pytest.raises(ValueError,match='cylinders'):
        h.push(prism(),p,(s,s),t+100,e,dynamic_enabled=torch.tensor([True]))
    for i in range(1,20):h.push(prism(),p,(s,s),t+i*100,e,dynamic_enabled=torch.tensor([False]))
    assert len(h.anchors)==len(h.histories[0].captures)==6

def test_old_support_expires_and_return_frame_rotates():
    h,p,t,e,s=setup();c=h.histories[0].captures[0]
    assert torch.isnan(current_support169(c,p,t+501,e,GRID,load_config())).all()
    q=p.clone();q[:,3]=torch.pi/2
    r=transform_returns169(rays(dist=.5),p,q,torch.tensor([True]))
    assert torch.allclose(r.directions,torch.tensor([[[0.,-1.,0.]]]),atol=1e-6)
    r=transform_returns169(rays(dist=.5),p,q,torch.tensor([False]))
    assert not r.occupied_hit.any()
