import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
import yaml
from gd_lab.gast_nav.person_contact import PersonContactTracker,sample_person_contacts,person_contact_termination,person_invalid_termination
from gd_lab.gast_nav.scene_adapter import EpisodeCylinder
from gd_lab.gast_nav.scene_types import SceneIndex
from gd_lab.gast_nav.terrains import build_scene
ROOT=Path(__file__).resolve().parents[2]
CFG=yaml.safe_load((ROOT/'configs/gast_nav/environment.yaml').read_text())

def test_any_link_short_contact_and_opposing_forces():
    t=PersonContactTracker(2,1.);t.reset([0,1])
    force=torch.zeros(2,2,17,1,3)
    force[0,1,16,0,0]=2 # previous physics step, even a rear foot
    force[1,0,0,0,0]=2;force[1,0,1,0,0]=-2 # do not sum across links
    t.observe(force,torch.ones(2,dtype=torch.bool))
    assert t.commit()[0].tolist()==[True,True]
    t.commit();assert t.person_contact.tolist()==[1,1]

def test_threshold_disabled_and_partial_reset():
    t=PersonContactTracker(3,1.);t.reset([0,1,2])
    f=torch.tensor([[[1.,0,0]],[[1.001,0,0]],[[100.,0,0]]])
    t.observe(f,torch.tensor([True,True,False]))
    assert t.commit()[0].tolist()==[False,True,False]
    t.reset([1]);assert not t.contact.any() and t.person_contact.tolist()==[0,1,0]
    t.observe(f,torch.tensor([True,True,False]));t.commit()
    assert t.person_contact.tolist()==[0,2,0]

def test_invalid_initial_overlap_is_separate_not_policy_failure():
    t=PersonContactTracker(2,1.);t.reset([0,1],initial_overlap=[True,False])
    t.observe(torch.ones(2,1,3)*3,torch.ones(2,dtype=torch.bool))
    policy,invalid=t.commit()
    assert policy.tolist()==[False,True] and invalid.tolist()==[True,False]
    assert t.person_contact.tolist()==[0,1] and t.invalid.tolist()==[1,0]
    assert t.last_invalid_code.tolist()==[1,0]
    assert t.summary()['invalid_by_reason']['INITIAL_OVERLAP']==1
    assert t.summary()['valid_episode_denominator']==1
    with pytest.raises(RuntimeError):t.mark_invalid([1],'INITIAL_OVERLAP')

def test_nonfinite_contact_is_invalid_and_not_person_failure():
    t=PersonContactTracker(2,1.);t.reset([0,1])
    t.observe(torch.tensor([[[float('nan'),0,0]],[[float('nan'),0,0]]]),torch.tensor([True,False]))
    policy,invalid=t.commit()
    assert invalid.tolist()==[True,False] and not policy.any()
    assert t.last_invalid_code.tolist()==[5,0]

def fake_env():
    tracker=PersonContactTracker(2,1.);tracker.reset([0,1])
    links=['trunk','RR_foot'];scene={}
    for link in links:
        f=torch.zeros(2,2,1,1,3)
        if link=='RR_foot':f[0,1,0,0,2]=2.
        scene['person_contact_'+link]=SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=f))
    # unrelated ground net forces must never be inspected
    scene['contact_forces']=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.ones(2,17,3)*1000))
    robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2,13),joint_pos=torch.zeros(2,12),joint_vel=torch.zeros(2,12)))
    nav=SimpleNamespace(cfg={'person_contact':{'robot_links':links}},person_contacts=tracker,robot=robot,
                        cylinder=SimpleNamespace(enabled=torch.ones(2,dtype=torch.bool),initial_overlap=torch.ones(2,dtype=torch.bool)))
    return SimpleNamespace(nav=nav,scene=scene,num_envs=2,device='cpu')

@pytest.mark.parametrize('reverse',[False,True])
def test_termination_order_ground_exclusion_and_late_proximity(reverse):
    env=fake_env()
    funcs=[person_contact_termination,person_invalid_termination]
    if reverse:funcs.reverse()
    for fn in funcs:fn(env)
    t=env.nav.person_contacts
    assert t.last_contact.tolist()==[True,False]
    assert t.invalid.sum()==0 # cylinder.initial_overlap is deliberately ignored after reset
    assert t.person_contact.sum()==1
    for fn in funcs:fn(env)
    assert t.person_contact.sum()==1

def test_nonfinite_robot_prioritizes_invalid():
    env=fake_env();env.nav.robot.data.joint_vel[0,0]=torch.nan
    assert not person_contact_termination(env).any()
    assert person_invalid_termination(env).tolist()==[True,False]
    assert env.nav.person_contacts.last_invalid_code.tolist()==[2,0]

def test_sway_stub_and_future_owner_api_forwarding():
    tile=build_scene('nav_corridor',0.,seed=1)[2]
    model=EpisodeCylinder(SceneIndex([tile,tile]),stage='B')
    a=torch.tensor([[1.,-2.],[-3.,4.]])
    model.update(.005,a)
    torch.testing.assert_close(model.last_a_train_xy,a)
    a[0,0]=10.;assert model.last_a_train_xy[0,0]==1 # snapshot owns storage
    calls=[]
    with patch('gd_lab.gast_nav.scene_adapter.MovingCylinder.update',lambda self,dt,a:calls.append((dt,a.clone())),create=True):
        model.update(.005,a)
    assert not model.sway_api_stub and calls[0][0]==.005
    torch.testing.assert_close(calls[0][1],a)
    with pytest.raises(ValueError):model.update(0.,a)
    with pytest.raises(ValueError):model.update(.005,torch.ones(2,3))

def test_wiring_order_and_all_urdf_rigid_links():
    # Static check is deliberately not presented as an Isaac sensor runtime test.
    import xml.etree.ElementTree as ET
    urdf=ET.parse(ROOT/'assets/robots/rbq10/urdf/rbq10_simple.urdf').getroot()
    collapsed={j.find('child').get('link') for j in urdf.findall('joint')
               if j.get('type')=='fixed' and j.get('dont_collapse')!='true'}
    expected={e.get('name') for e in urdf.findall('link')}-collapsed
    assert set(CFG['person_contact']['robot_links'])==expected
    tree=ast.parse((ROOT/'src/gd_lab/gast_nav/env.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='physics_step')
    text=ast.unparse(fn)
    assert text.index('self.inertia.step')<text.index('self.cylinder.update')<text.index('self.cylinder.step')<text.index('self.cylinder.write_colliders')
    assert 'a[:, :2]' in text

def test_initial_overlap_only_reset_window_and_shape_query():
    from gd_lab.gast_nav.person_contact import reset_overlap_mask
    cylinder=SimpleNamespace(enabled=torch.tensor([True,False,True]),position=torch.tensor([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]),
                             radius=torch.tensor([.2,.3,.4]),height=torch.tensor([1.7,1.7,1.7]))
    calls=[]
    def query(p,r,h,prefix):
        calls.append((p,r,h,prefix));return prefix.endswith('env_2/Robot/')
    result=reset_overlap_mask(cylinder,[0,1,2],query)
    assert result.tolist()==[False,False,True] and len(calls)==2
    assert calls[1][0]==[6.,7.,8.] and calls[1][1]==pytest.approx(.4)
    t=PersonContactTracker(3,1.);t.reset([0,1,2],initial_overlap=result)
    t.begin_action()
    with pytest.raises(RuntimeError,match='reset-only'):t.mark_invalid([0],'INITIAL_OVERLAP')
    t.observe(torch.ones(3,1,3)*2,torch.ones(3,dtype=torch.bool))
    policy,invalid=t.commit()
    assert policy.tolist()==[True,True,False] and invalid.tolist()==[False,False,True]
