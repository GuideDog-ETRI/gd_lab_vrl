from pathlib import Path
import pytest
import torch
import yaml
from gd_lab.gast_nav.rewards import reward_terms
from gd_lab.gast_nav.curriculum167 import easy_scene_settings,easy_command_settings
from gd_lab.gast_nav.diagnostics167 import required_causes
from gd_lab.gast_nav.scene_config import load_scene_config

ROOT=Path(__file__).resolve().parents[2]
CFG=yaml.safe_load((ROOT/'configs/gast_nav/environment.yaml').read_text())
C=yaml.safe_load((ROOT/'configs/gast_nav/curriculum_C.yaml').read_text())


def args():
    n=4;z=torch.zeros(n);o=torch.ones(n)
    return dict(mode=torch.tensor([1,1,2,2]),go=o.bool(),phase=z.long(),feasible=o.bool(),
        observed=torch.tensor([False,True,False,True]),v_path=o*.3,v_cap=o*.6,speed=z,
        yaw_rate=z,turn_rate=o*-.6,positive_yaw=z,clearance=o,margin=o*.1,offset_error=z,
        yaw_error=z,turn_drift=z,contact=z,entered_forbidden=z,cfg={**CFG['rewards'],**C['reward_overrides']})


def test_unknown_reward_requires_actual_inputs_and_turn_observed():
    a=args()
    with pytest.raises(ValueError,match='support_stable'):reward_terms(**a)
    a['support_stable']=torch.ones(4,dtype=torch.bool)
    with pytest.raises(ValueError,match='swept entry'):reward_terms(**a)
    a['entered_unknown']=torch.tensor([True,False,False,False])
    terms,w=reward_terms(**a)
    assert terms['unknown_stop'].tolist()==[1,0,1,0]
    assert w['unknown_entry'].tolist()==pytest.approx([-.05,0,0,0])
    assert terms['turn_tracking'].tolist()==pytest.approx([0,0,0,.36])
    assert w['progress'][1]>.05
    for key,value in [('support_stable',torch.zeros(4,dtype=torch.bool)),('contact',torch.ones(4)),('speed',torch.ones(4)),('yaw_rate',torch.ones(4)),('clearance',torch.zeros(4))]:
        bad={**a,key:value};assert not reward_terms(**bad)[0]['unknown_stop'].any()


def test_easy_profile_does_not_mutate_legacy_or_base():
    base=load_scene_config();result=easy_scene_settings(base,C['scene_limits'])
    assert base['corridor']['width'][0]==.5
    assert min(result['corridor']['width'])>=1 and min(result['door']['width'])>=1
    assert result['corridor']['box_count']==[0,0] and result['stairs']['box_count']==[0,0]
    assert result['turn_room']['delta_hard_m']>=.2
    assert result['terrain']==base['terrain']
    runtime=yaml.safe_load((ROOT/'configs/gast_nav/runtime.yaml').read_text())
    # Runtime root is the command config used by the protocol.
    commands=runtime.get('command',runtime.get('commands',runtime))
    assert easy_command_settings(commands,C['scene_limits'])['v_cap_range'][1]<=.6


def test_attribution_does_not_invent_missing_provenance():
    required=torch.ones(1,1,3,dtype=torch.bool)
    state=torch.ones(1,3,1,3,dtype=torch.uint8);state[:,0,0,1]=0;state[:,1,0,2]=0
    support=required.clone();support[:,:,0]=False
    missing,masks=required_causes(required,state,support)
    assert missing.all()
    assert [masks[k].sum().item() for k in ('support_missing','B1_subbin_incomplete','B2_subbin_incomplete')]==[1,1,1]
    assert masks['self_blocked'] is None and masks['ttl_expired'] is None


def test_swept_union_keeps_gaps_and_future_foot_without_body_subtraction():
    from gd_lab.gast_nav.swept167 import trajectory_required
    from gd_lab.gast_nav.map_types import Grid
    class Feet:
        def mask(self,xy,pos,quat,pose,margin):
            return ((xy[None,None]-pos[:,:,:2,None,None].permute(0,1,3,4,2)).square().sum(-1)<=(.02+margin)**2).any(1)
    pos=torch.tensor([[[[0.,-.3,0.],[0.,.3,0.]],[[.1,-.3,0.],[.1,.3,0.]]]])
    quat=torch.zeros(1,2,2,4);quat[...,0]=1
    grid=Grid(9,17,.05,-.1,-.4)
    result=trajectory_required(Feet(),grid,pos,quat,torch.zeros(1,4),projected_error_m=.01,between_sample_motion_bound_m=.1)
    assert result[0,2,4] and result[0,14,4]  # future feet
    assert not result[0,8,4]  # gap between legs, no global bounding hull
    with pytest.raises(ValueError):trajectory_required(Feet(),grid,pos[:,:1],quat[:,:1],torch.zeros(1,4),projected_error_m=.01,between_sample_motion_bound_m=.1)


@pytest.mark.parametrize('case',range(1,7))
def test_pending_evidence_six_counterexamples_fail_closed(case):
    from gd_lab.gast_nav.evidence167 import certified_cells
    # No model provenance is available: old/free/expired/misaligned inputs
    # cannot certify. These validate the STUB only, not the future model.
    state=torch.ones(1,3,5,5,dtype=torch.uint8);support=torch.ones(1,5,5,dtype=torch.bool)
    if case==2:state[:,0,2,2]=2  # occupied is still not free B2 evidence
    if case==3:state[:,:,2,2]=0
    if case==4:state[:,:,:,2]=0
    if case==5:support[:,2,2]=False
    if case==6:support[:]=False
    assert not certified_cells(state,support).any()


def test_unknown_entry_is_new_cell_and_reset_safe():
    from gd_lab.gast_nav.evidence167 import entered_unknown_cells
    before=torch.zeros(2,3,3,dtype=torch.bool);before[:,1,1]=True
    now=before.clone();now[:,1,2]=True
    known=torch.ones_like(now);known[:,1,2]=False
    assert entered_unknown_cells(now,before,known,torch.tensor([True,False])).tolist()==[True,False]
    assert not entered_unknown_cells(before,before,known,torch.ones(2,dtype=torch.bool)).any()


def test_candidate_profile_and_p2_slip_semantics():
    from gd_lab.gast_nav.swept167 import candidate_stop_profile,p2_projected_inflation
    c=C['swept_candidate'];p=candidate_stop_profile(.6,'NAV',c)
    assert len(p)==125 and p[:5].tolist()==[.6]*5 and p[-1]==0
    with pytest.raises(ValueError):candidate_stop_profile(-.6,'TURN',c)
    turn=candidate_stop_profile(-.6,'TURN',c,turn_deceleration=1.)
    assert len(turn)==65 and (turn<=0).all()
    inflation=p2_projected_inflation(p,c,radius_m=.76)
    assert inflation['horizontal_m']>0 and inflation['vertical_m']==.02


def test_stub_does_not_hide_current_occupied_clearance():
    from gd_lab.gast_nav.evidence167 import certified_cells
    from gd_lab.gast_nav.near_clearance import clearance_from_state
    from gd_lab.gast_nav.map_types import Grid
    s=torch.zeros(1,3,3,3,dtype=torch.uint8);s[:,0,1,1]=2
    before=clearance_from_state(s,grid=Grid(3,3,.05,-.05,-.05))
    certified_cells(s,torch.ones(1,3,3,dtype=torch.bool))
    assert before[0,0,1,1]==0
    assert torch.equal(before,clearance_from_state(s,grid=Grid(3,3,.05,-.05,-.05)))


def test_envelope_missing_command_and_grid_clipping_fail_closed():
    from gd_lab.gast_nav.envelope167 import select_envelope,resample_required
    from gd_lab.gast_nav.map_types import Grid
    mask=torch.zeros(3,9,dtype=torch.bool);mask[1,0]=True;mask[1,8]=True
    entry=dict(mask=mask,grid=dict(nx=9,ny=3,res=.05,x0=-.2,y0=-.05))
    assert select_envelope({'NAV_0.4':entry},'NAV',.41) is None
    assert select_envelope({'NAV_0.4':entry},'NAV',-.4) is None
    assert select_envelope({'TURN_0.3':entry},'TURN',.3) is None
    result,contained=resample_required(entry,Grid(9,3,.05,-.2,-.05),device='cpu')
    assert contained and not result[1,4]
    _,contained=resample_required(entry,Grid(3,3,.05,-.05,-.05),device='cpu')
    assert not contained


def test_latest_3906_command_uses_observation_scale_and_preserves_history():
    from gd_lab.gast_nav.obs_terms import replace_latest_command
    obs=torch.arange(100,dtype=torch.float32)[None];old=obs.clone()
    replace_latest_command(obs,torch.tensor([[.4,0.,-.3]]),(2.,2.,.25))
    torch.testing.assert_close(obs[:,42:45],torch.tensor([[.8,0.,-.075]]))
    assert torch.equal(obs[:,:42],old[:,:42]) and torch.equal(obs[:,45:],old[:,45:])


def test_c_command_sampler_uses_measured_combinations_only_when_available():
    import random
    from gd_lab.gast_nav.command_generator import CommandGenerator
    runtime=yaml.safe_load((ROOT/'configs/gast_nav/runtime.yaml').read_text())['command']
    gen=CommandGenerator([],{**CFG,'curriculum_C_limits':C['scene_limits']},easy_command_settings(runtime,C['scene_limits']),seed=1)
    r=random.Random(1)
    assert {gen._speed_cap(r) for _ in range(100)}=={.4,.6}
    assert {gen._turn_rate(r,.6) for _ in range(100)}=={-.3,-.6}
    assert gen._turn_rate(r,.3)==-.3
