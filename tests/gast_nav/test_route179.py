import math
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import ast
import pytest
import torch
from cpu_fixture179 import observations, policy_pair, CHECKPOINT, SHA
from gd_lab.gast_nav import schema
from gd_lab.gast_nav.commands import RouteHint, Conditions, NavCommand, encode, decode, policy_features
from gd_lab.gast_nav.command_generator import route_hint
from gd_lab.gast_nav.route_obs179 import route_features179
from gd_lab.gast_nav.migration import migrate_checkpoint, migrate_state_dict
from gd_lab.gast_nav.trainable import apply_manifest, assert_frozen
import yaml

torch.set_num_threads(1)
STAGE=Path(__file__).resolve().parents[2]
BASE=STAGE


def tile(path):
    return SimpleNamespace(route_polyline=path,segments=[('FLAT',0.,sum(math.dist(a,b) for a,b in zip(path,path[1:])))])


def feat(path,xy=(0.,0.),yaw=0.,mode='NAV'):
    hint,*_=route_hint(tile(path),xy)
    return route_features179(hint,yaw,mode=mode)


def test_selected_left_right_same_scene():
    # Identical geometry/robot/commands; only gdm selected branch differs.
    left=feat(((0.,0.),(2.,2.)));right=feat(((0.,0.),(2.,-2.)))
    assert left[0]>0 and right[0]<0 and left[1]==right[1]
    assert schema.ROUTE_FEATURE_DIM==3
    assert schema.ROUTE_FEATURE_KEYS==('route_yaw_sin','route_yaw_cos','route_lateral_error_m')


def test_parallel_routes_lateral_and_yoff_independent():
    assert feat(((-2.,1.),(2.,1.)))[2]==-1.
    assert feat(((-2.,-1.),(2.,-1.)))[2]==1.
    h=route_hint(tile(((-2.,0.),(2.,0.))),(.3,.2))[0]
    assert route_features179(h,0.,mode='NAV')==[0.,1.,.2]
    # Existing 16-wide conditions still hold desired y_off separately.
    old=policy_features({'mode':'NAV','effective_go':True,'v_cap':.5,'velocity':(.5,0.,0.)},Conditions(.1,.05,.3),'APPROACH',0.)
    assert len(old)==16 and old[-3]==.1


def test_frame_rotation_translation_and_wrap():
    p=((-2.,0.),(2.,0.)); q=(.5,.25); yaw=.3
    expected=feat(p,q,yaw)
    theta=2.3; tx,ty=100.,-70.
    def transform(x):return (math.cos(theta)*x[0]-math.sin(theta)*x[1]+tx,math.sin(theta)*x[0]+math.cos(theta)*x[1]+ty)
    assert feat(tuple(map(transform,p)),transform(q),yaw+theta)==pytest.approx(expected,abs=1e-12)
    assert feat(p,q,yaw+2*math.pi)==pytest.approx(expected,abs=1e-12)
    assert feat(p,q,math.pi-1e-8)[0]<0
    assert feat(p,q,-math.pi+1e-8)[0]>0


def test_bent_route_and_reset_no_cached_route():
    p=((0.,0.),(1.,0.),(1.,2.))
    assert feat(p,(1.1,1.))[0:2]==pytest.approx([1.,0.],abs=1e-12)
    assert feat(p,(1.1,1.))[2]==pytest.approx(-.1)
    assert feat(((0.,0.),(2.,0.)))==[0.,1.,0.]
    assert feat(((0.,0.),(0.,-2.)))==pytest.approx([-1.,0.,0.],abs=1e-12)


def test_missing_invalid_and_legacy_turn():
    assert route_features179(None,0.,mode='LEGACY')==[0.,0.,0.]
    assert route_features179(None,0.,mode='TURN')==[0.,0.,0.]
    h=RouteHint(0,'FLAT',0.,1.,'FLAT')
    with pytest.raises(ValueError,match='MISSING'):route_features179(h,0.,mode='NAV')
    for bad in (float('nan'),float('inf'),True):
        with pytest.raises(ValueError,match='INVALID'):route_features179(replace(h,route_lateral_error_m=bad),0.,mode='NAV')
    with pytest.raises(ValueError,match='INVALID'):route_features179(replace(h,route_lateral_error_m=0.),float('nan'),mode='NAV')


def test_nav1_roundtrip_and_old_optional_field():
    cfg=yaml.safe_load((BASE/'configs/gast_nav/runtime.yaml').read_text())['command']
    hint=RouteHint(0,'FLAT',.4,1.,'FLAT',route_lateral_error_m=-.25)
    cmd=NavCommand(1,0.,'NAV',True,.5,0.,Conditions(**cfg['defaults']),1,1,1,hint)
    assert decode(encode(cmd),cfg=cfg,now_ms=0.)==cmd
    old=replace(cmd,route_hint=replace(hint,route_lateral_error_m=None))
    assert decode(encode(old),cfg=cfg,now_ms=0.)==old
    # Actual old sender payload omits the new key entirely.
    import json,zlib
    from gd_lab.gast_nav.commands import HEADER,VERSION
    payload=json.loads(encode(old)[HEADER.size:])
    del payload['route_hint']['route_lateral_error_m']
    raw=json.dumps(payload,separators=(',',':')).encode()
    packet=HEADER.pack(b'NAV1',VERSION,len(raw),zlib.crc32(raw))+raw
    assert decode(packet,cfg=cfg,now_ms=0.)==old
    for v in (True,'bad',float('inf')):
        with pytest.raises(ValueError):decode(encode(replace(cmd,route_hint=replace(hint,route_lateral_error_m=v))),cfg=cfg,now_ms=0.)


@pytest.fixture(scope='module')
def migrated():
    torch.manual_seed(179)
    obs=observations(4);ref,nav=policy_pair(obs)
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    ref.load_state_dict(checkpoint['model_state_dict'],strict=True)
    record=migrate_checkpoint(nav,CHECKPOINT,expected_sha256=SHA)
    ref.eval();nav.eval()
    return obs,ref,nav,record


def test_actual_3906_zero_init_actor_critic_parity(migrated):
    obs,ref,nav,record=migrated
    with torch.no_grad():
        a=ref.act_inference(obs);v=ref.evaluate(obs)
        assert torch.equal(a,nav.act_inference(obs))
        assert torch.equal(v,nav.evaluate(obs))
        changed=obs.clone()
        for key in ('nav_wide','nav_near','nav_features','nav_route_features','nav_privileged'):changed[key]=torch.randn_like(changed[key])*10
        assert torch.equal(a,nav.act_inference(changed))
        assert torch.equal(v,nav.evaluate(changed))
        x0=nav._actor_input(obs,inference=True);x1=nav._actor_input(changed,inference=True)
        assert not torch.equal(x0[:,-3:],x1[:,-3:])
    assert nav.actor[0].W_new.shape[-1]==67
    assert nav.critic[0].W_new.shape[-1]==75
    assert record['route_observation']['key']=='nav_route_features'
    assert any(k.startswith('nav_route_normalizer.') for k in record['fresh_keys'])
    assert any(k.startswith('nav_critic_route_normalizer.') for k in record['fresh_keys'])
    # Every inherited weight/buffer equals actual source3906, no substitutions.
    for k,t in ref.state_dict().items():assert torch.equal(t,nav.state_dict()[k]),k


def test_3906_migration_rejects_unapproved_and_old_nav(migrated):
    obs,ref,nav,_=migrated
    src=ref.state_dict();target=nav.state_dict()
    with pytest.raises(ValueError):migrate_state_dict({**src,'unexpected':torch.ones(1)},target)
    with pytest.raises(ValueError):migrate_state_dict({k:v for k,v in src.items() if k!='actor.0.bias'},target)
    with pytest.raises(ValueError):migrate_checkpoint(nav,CHECKPOINT,expected_sha256='0'*64)
    with pytest.raises(ValueError):migrate_state_dict({**src,'actor.0.W_new':torch.zeros(512,64)},target)
    merged,fresh=migrate_state_dict(src,target)
    assert not merged['actor.0.W_new'].count_nonzero()
    assert set(k for k in fresh if k.startswith('nav_route_normalizer.'))==set(k for k in target if k.startswith('nav_route_normalizer.'))


def test_shape_missing_route_is_not_silently_zero(migrated):
    obs,_,nav,_=migrated
    with pytest.raises(KeyError):nav.act_inference(obs.exclude('nav_route_features'))
    bad=obs.clone();bad['nav_route_features']=torch.zeros(4,2)
    with pytest.raises(ValueError,match='schema'):nav.act_inference(bad)


def test_A0_frozen_and_normalization_and_resume(migrated):
    import copy
    obs,ref,initial,_=migrated
    nav=copy.deepcopy(initial)
    _,manifest=apply_manifest(nav,BASE/'configs/gast_nav/trainable_A0.yaml',learning_rate=.001)
    before={k:v.clone() for k,v in nav.state_dict().items()}
    nav.train()
    nav.update_normalization(obs)
    nav.eval()
    assert nav.nav_route_normalizer.count.item()>0
    assert nav.nav_critic_route_normalizer.count.item()>0
    assert_frozen(nav,manifest['frozen'])
    for k,t in ref.state_dict().items():
        # Critic parameters may be trainable A0, but no optimizer/learning runs here.
        assert torch.equal(before[k],nav.state_dict()[k]),k
    _,restored=policy_pair(obs)
    restored.load_state_dict(nav.state_dict(),strict=True);restored.eval()
    for k,v in nav.state_dict().items():assert torch.equal(v,restored.state_dict()[k]),k
    with torch.no_grad():
        x=nav._actor_input(obs,inference=True);y=restored._actor_input(obs,inference=True)
        delta=(nav.act_inference(obs)-restored.act_inference(obs)).abs().max().item()
        print(f"resume_max_abs={delta}; actor_input_max_abs={(x-y).abs().max().item()}")
        torch.testing.assert_close(nav.act_inference(obs),restored.act_inference(obs),atol=1e-5,rtol=1e-6)


def test_env_wiring_without_loading_isaac():
    source=(STAGE/'src/gd_lab/gast_nav/env.py').read_text()
    ast.parse(source)
    assert 'route_features179(hint,yaw,mode=' in source
    assert 'nav_route_features=self.route_features.clone()' in source
    assert '(d.root_pos_w-env.scene.env_origins)[:,:2]' in source


def test_route_mirror_and_packed_roundtrip(migrated):
    from gd_lab.gast_nav.ppo import mirror_legacy
    from gd_lab.gast_nav.packed_rollout import CapturePool
    obs,_,_,_=migrated
    obs=obs.clone();obs['nav_route_features']=torch.tensor([[.3,.95,.2]]).expand(4,-1).clone()
    def mirror(base,a,env):return torch.cat((base,base),0),torch.cat((a,-a),0)
    out,_,ids=mirror_legacy(obs,torch.zeros(4,12),torch.tensor([0,1,2,0]),base_mirror=mirror,privileged_mirror=lambda x:x)
    assert ids.tolist()==[0,3]
    torch.testing.assert_close(out['nav_route_features'],torch.tensor([[-.3,.95,-.2]]).expand(2,-1))
    # Real capture pool retains the per-step route; no map-frame route caching.
    obs['nav_wide'].zero_();obs['nav_near'].zero_()
    source=SimpleNamespace(pipeline=SimpleNamespace(mapper=SimpleNamespace(history=[SimpleNamespace(sample_time_ms=torch.zeros(4,dtype=torch.int64)) for _ in range(3)])))
    pool=CapturePool(source)
    small=pool.references(obs)
    restored=pool.unpack(small)
    assert torch.equal(obs['nav_route_features'],restored['nav_route_features'])
    obs['nav_route_features'].zero_()
    assert restored['nav_route_features'].abs().sum()>0


def test_route_columns_receive_gradient_without_training(migrated):
    import copy
    obs,_,initial,_=migrated
    nav=copy.deepcopy(initial)
    nav.zero_grad(set_to_none=True)
    nav.act_inference(obs).square().sum().backward()
    assert nav.actor[0].W_new.grad[:,-3:].abs().sum()>0
    # No optimizer constructed/stepped: backward only proves input connectivity.
    assert nav.actor[0].W_new.count_nonzero()==0
