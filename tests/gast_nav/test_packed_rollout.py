import copy
from types import SimpleNamespace as NS
import torch
from cpu_fixture import observations,policy_pair,ROOT,runtime
from gd_lab.gast_nav.map_types import MapFrame,NearFrame
from gd_lab.gast_nav.map_config import load_config
from gd_lab.gast_nav.map_inputs import network_inputs
from gd_lab.gast_nav.packed_rollout import CapturePool
from gd_lab.gast_nav.ppo import GastNavPPO
from gd_lab.gast_nav.obs_terms import invalidate_held_rows


def source_fixture(n=3):
    frames=[]
    for i in range(3):
        pose=torch.zeros(n,4);pose[:,0]=i*.03;pose[:,3]=i*.01
        frames.append(MapFrame(torch.randint(0,3,(n,3,43,49),dtype=torch.uint8),
            torch.randint(0,8,(n,3,43,49)).float()*.1,
            torch.ones(n,43,49,dtype=torch.uint8),pose,torch.full((n,),1000-i*100),
            torch.full((n,),3-i),torch.ones(n,dtype=torch.long)))
    near=NearFrame(torch.randint(0,256,(n,2,25,37),dtype=torch.uint8),
                   torch.rand(n,25,37)>.9,torch.zeros(n,25,37))
    src=NS(last_capture=1.,pipeline=NS(mapper=NS(history=frames,epoch=frames[0].epoch.clone()),near=near,config=load_config()),
           env=NS(common_step_counter=100,step_dt=.01),pose=lambda:frames[0].sample_pose)
    obs=observations(n);obs['nav_wide'],obs['nav_near']=network_inputs(frames,near,frames[0].sample_pose,frames[0].sample_time_ms)
    return src,obs


def test_packed_history_exact_pin_reset_and_shared_pool():
    torch.manual_seed(163);source,obs=source_fixture();pool=CapturePool(source)
    refs=pool.references(obs);size=len(pool.records)
    again=pool.references(obs)
    assert size==1 and len(pool.records)==2
    assert 'nav_wide' not in refs and 'nav_near' not in refs
    assert not torch.equal(refs['nav_capture_ids'],again['nav_capture_ids'])
    # Later reset must not invalidate previously pinned rollout observations.
    source.pipeline.mapper.epoch[1]+=1
    reset_obs=obs.clone();reset_obs['nav_wide'],reset_obs['nav_near']=invalidate_held_rows(obs['nav_wide'],obs['nav_near'],torch.tensor([1]))
    reset_refs=pool.references(reset_obs)
    # Mutation of producer storage and future arrival cannot change pinned bytes.
    for frame in source.pipeline.mapper.history:frame.state.fill_(0)
    from dataclasses import replace
    source.pipeline.mapper.history=[replace(frame,sample_time_ms=frame.sample_time_ms+1000,
        seq=frame.seq+10) for frame in source.pipeline.mapper.history]
    source.last_capture=2.
    actual=pool.unpack(refs);reset=pool.unpack(reset_refs)
    for key in ['nav_wide','nav_near']:
        torch.testing.assert_close(actual[key],obs[key],rtol=0,atol=0)
        torch.testing.assert_close(reset[key],reset_obs[key],rtol=0,atol=0)
    assert pool.bytes()<obs['nav_wide'].numel()*4
    pool.clear();assert pool.bytes()==0


def test_dense_packed_action_logprob_and_chunk_gradient_parity():
    torch.manual_seed(17);src,obs=source_fixture();pool=CapturePool(src);packed=pool.unpack(pool.references(obs))
    _,p=policy_pair(obs)
    with torch.no_grad():p.actor[0].W_new.normal_(0,.01);p.critic[0].W_new.normal_(0,.01)
    q=copy.deepcopy(p)
    actions=torch.randn(3,12)
    p.act(obs);lp=p.get_actions_log_prob(actions);value=p.evaluate(obs)
    q.act(packed);qlp=q.get_actions_log_prob(actions)
    torch.testing.assert_close(p.action_mean,q.action_mean,rtol=0,atol=0)
    torch.testing.assert_close(lp,qlp,rtol=0,atol=0)
    (lp.sum()/3+value.square().sum()/3).backward()
    q.zero_grad(set_to_none=True)
    for ids in [torch.tensor([0,1]),torch.tensor([2])]:
        part=pool.unpack(pool.references(obs)[ids])
        q.act(part);loss=q.get_actions_log_prob(actions[ids]).sum()/3+q.evaluate(part).square().sum()/3
        loss.backward()
    for (name,a),(_,b) in zip(p.named_parameters(),q.named_parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-4,msg=name)


def test_packed_storage_ppo_steps_and_parity():
    torch.manual_seed(19);source,obs=source_fixture(2);_,p=policy_pair(obs);q=copy.deepcopy(p)
    cfg=dict(runtime()['ppo']);cfg['update_chunk_size']=1
    def algorithm(policy,packed):
        alg=GastNavPPO(policy,nav_config=cfg,device='cpu',schedule='fixed',num_learning_epochs=1,num_mini_batches=1,
            trainable_manifest=ROOT/'configs/gast_nav/trainable_A0.yaml')
        if packed:alg.map_source=source
        alg.init_storage('rl',2,2,obs,[12]);return alg
    dense=algorithm(p,False);packed=algorithm(q,True)
    for alg in [dense,packed]:
        torch.manual_seed(25)
        with torch.no_grad():
            for _ in range(2):
                control=torch.tensor([0,1]);action=alg.act_selected(obs,control)
                alg.process_env_step(obs,torch.ones(2),torch.zeros(2,dtype=torch.bool),
                    {'nav_controller':control,'nav_executed_action':action,'nav_override_reason':control})
            alg.compute_returns(obs)
        assert all('nav_wide' not in v and 'nav_near' not in v for v in alg.nav_normalization_obs)
    assert 'nav_wide' not in packed.storage.observations
    for key in ['actions','actions_log_prob','values','advantages']:
        torch.testing.assert_close(getattr(dense.storage,key),getattr(packed.storage,key),rtol=0,atol=0)
    for alg in [dense,packed]:torch.manual_seed(44);alg.update()
    for key,v in p.state_dict().items():torch.testing.assert_close(v,q.state_dict()[key],rtol=2e-5,atol=2e-6,msg=key)
    assert packed.capture_pool.bytes()==0


def test_postwarp_decode_never_warps_and_preserves_elapsed_age(monkeypatch):
    from gd_lab.gast_nav import map_inputs
    source,obs=source_fixture(2)
    # Non-decimal capture offsets exercise exact float lookup, not requantization.
    source.pipeline.mapper.history[1].sample_time_ms[:]=863
    source.pipeline.mapper.history[2].sample_time_ms[:]=711
    obs['nav_wide'],obs['nav_near']=network_inputs(source.pipeline.mapper.history,
        source.pipeline.near,source.pose(),source.pipeline.mapper.history[0].sample_time_ms)
    pool=CapturePool(source);refs=pool.references(obs)
    def forbidden(*a,**kw):raise AssertionError('update must not warp')
    monkeypatch.setattr(map_inputs,'network_inputs',forbidden)
    monkeypatch.setattr(map_inputs,'warp_categorical',forbidden)
    actual=pool.unpack(refs)
    for key in ('nav_wide','nav_near'):assert torch.equal(actual[key],obs[key])
    assert pool.bank['bits'].dtype==torch.uint8
    assert pool.bank['near'].dtype==torch.uint8
