import copy
import tempfile
import unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cpu_fixture import *
from gd_lab.gast_nav.ppo import *
from gd_lab.gast_nav.trainable import apply_manifest


def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=3)
    try:
        p=torch.nn.Parameter(torch.tensor(2.))
        x=torch.tensor([1.,2.])+rank*2
        mask=[torch.tensor([True,True]),torch.tensor([False,False]),torch.tensor([True,False])][rank]
        loss,n=masked_global_mean(p*x,mask)
        assert n==3
        loss.backward(); average_gradients([p])
        torch.testing.assert_close(p.grad,torch.tensor(8/3))
        norm=normalize_valid(x,mask,1e-8)
        local=torch.stack((norm[mask].sum(),norm[mask].square().sum()))
        dist.all_reduce(local)
        torch.testing.assert_close(local,torch.tensor([0.,3.]),atol=1e-6,rtol=0)
        p.grad=None
        loss,n=masked_global_mean(p*x,torch.zeros(2,dtype=torch.bool))
        assert n==0
        # All ranks explicitly skip optimizer actor step; no stale Adam state changes.
        old=p.detach().clone()
        if n: loss.backward()
        assert p.grad is None and torch.equal(old,p)
    finally: dist.destroy_process_group()

class PPOTests(unittest.TestCase):
    def test_gloo_three_ranks_empty_rank_and_global_empty(self):
        path=str(ROOT/'tests/gast_nav/.runtime/gloo_fixture')
        if Path(path).exists(): Path(path).unlink()
        mp.spawn(worker,args=(path,),nprocs=3,join=True)
    def test_intervention_first_hold_tick_only(self):
        r=intervention_rewards(torch.ones(4),torch.tensor([0,1,1,2]),torch.tensor([0,0,1,1]),-2.)
        torch.testing.assert_close(r,torch.tensor([1.,-1.,1.,1.]))
    def test_masked_gradient(self):
        p=torch.tensor([1.,2.,3.],requires_grad=True)
        loss,n=masked_global_mean(p,torch.tensor([True,False,True]));loss.backward()
        torch.testing.assert_close(p.grad,torch.tensor([.5,0,.5]));self.assertEqual(n,2)
    def test_native_ppo_storage_and_all_hold_actor_skip(self):
        obs=observations(3)
        _,p=policy_pair(obs)
        cfg=runtime()['ppo']
        alg=GastNavPPO(p,nav_config=cfg,device='cpu',schedule='fixed',num_learning_epochs=1,num_mini_batches=1)
        groups,_=apply_manifest(p,ROOT/'configs/gast_nav/trainable_A0.yaml',learning_rate=.001)
        alg.optimizer=torch.optim.Adam(groups)
        alg.init_storage('rl',3,2,obs,[12])
        before=p.actor[0].W_new.detach().clone()
        with torch.no_grad():
            for i in range(2):
                proposal=alg.act_selected(obs,torch.ones(3,dtype=torch.long))
                alg.process_env_step(obs,torch.ones(3),torch.zeros(3,dtype=torch.bool),
                    {'nav_controller':torch.ones(3,dtype=torch.long),'nav_executed_action':torch.zeros_like(proposal),
                     'nav_override_reason':torch.ones(3,dtype=torch.long)})
            alg.compute_returns(obs)
        self.assertTrue(torch.equal(alg.nav_masks,torch.zeros(2,3,dtype=torch.bool)))
        self.assertTrue(torch.equal(alg.storage.rewards[0],torch.zeros(3,1)))
        result=alg.update()
        self.assertEqual(result['actor_skipped'],1)
        self.assertTrue(torch.equal(before,p.actor[0].W_new))
    def test_mirror_only_legacy(self):
        obs=observations(3); actions=torch.arange(36.).reshape(3,12)
        def mirror(base,a,env):
            # Mock only environment-dependent joint permutation; map/command reflection real.
            return torch.cat((base,base),0),torch.cat((a,-a),0)
        out,a,ids=mirror_legacy(obs,actions,torch.tensor([0,1,2]),base_mirror=mirror,privileged_mirror=lambda x:x)
        self.assertEqual(ids.tolist(),[0])
        torch.testing.assert_close(out['nav_wide'],obs['nav_wide'][0:1].flip(-2))
        torch.testing.assert_close(out['nav_near'],obs['nav_near'][0:1].flip(-2))
        torch.testing.assert_close(out['nav_features'][:,13],-obs['nav_features'][0:1,13])
        torch.testing.assert_close(out['nav_features'][:,:3],obs['nav_features'][0:1,:3])
        torch.testing.assert_close(a,-actions[0:1])

class PPOIntegrationTests(unittest.TestCase):
    def test_mixed_ticks_A_update_and_preselection_rejection(self):
        torch.manual_seed(159)
        obs=observations(3)
        _,p=policy_pair(obs)
        alg=GastNavPPO(p,nav_config=runtime()['ppo'],device='cpu',schedule='fixed',
                       num_learning_epochs=1,num_mini_batches=1,
                       trainable_manifest=ROOT/'configs/gast_nav/trainable_A.yaml')
        alg.init_storage('rl',3,2,obs,[12])
        initial=p.actor[0].W_new.detach().clone()
        with torch.no_grad():
            for tick in range(2):
                controllers=torch.tensor([0,1,2])
                proposal=alg.act_selected(obs,controllers)
                with self.assertRaises(ValueError):
                    alg.process_env_step(obs,torch.ones(3),torch.zeros(3,dtype=torch.bool),
                                         {'nav_controller':torch.zeros(3,dtype=torch.long)})
                alg.process_env_step(obs,torch.tensor([1.,3.,2.]),torch.zeros(3,dtype=torch.bool),
                    {'nav_controller':controllers,'nav_executed_action':proposal,
                     'nav_override_reason':torch.tensor([0,1,2])})
            alg.compute_returns(obs)
        self.assertEqual(alg.nav_masks.tolist(),[[True,False,False],[True,False,False]])
        self.assertEqual(alg.storage.rewards[:,1,0].tolist(),[2.,3.])
        result=alg.update()
        self.assertFalse(torch.equal(initial,p.actor[0].W_new))
        self.assertIn('gast_reconstruction',result)
        self.assertTrue(all(torch.isfinite(v).all() for v in p.state_dict().values()))

class NativeMirrorTests(unittest.TestCase):
    def test_native_proprio_terrain_mirror_involution(self):
        from types import SimpleNamespace as NS
        from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC
        spec=DREAMWAQ_SPEC
        groups={k:v.resolve(**{t.name:187 for t in v.terms if t.dim is None}) for k,v in spec.groups.items()}
        inner=NS(device='cpu',scene={'robot':NS(data=NS(joint_names=[f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HIP','THIGH','CALF')]))},
                 cfg=NS(scene=NS(height_scanner=NS(pattern_cfg=NS(size=(1.6,1.),resolution=.1)))),
                 observation_manager=NS(active_terms={k:[t.name for t in v.terms] for k,v in groups.items()},
                                        group_obs_dim={k:(v.total,) for k,v in groups.items()}))
        env=NS(unwrapped=inner)
        obs=observations(3);actions=torch.randn(3,12)
        first,a,ids=mirror_legacy(obs,actions,torch.tensor([0,1,2]),env=env,privileged_mirror=lambda x:x)
        second,b,_=mirror_legacy(first,a,torch.tensor([0]),env=env,privileged_mirror=lambda x:x)
        for key in obs.keys(): torch.testing.assert_close(second[key],obs[key][0:1])
        torch.testing.assert_close(b,actions[0:1])
        self.assertEqual(ids.tolist(),[0])
