import copy
import unittest
import torch
from cpu_fixture import *
from gd_lab.gast_nav.migration import migrate_checkpoint, migrate_state_dict
from gd_lab.gast_nav.trainable import apply_manifest, assert_frozen

torch.set_num_threads(1)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(159)
        cls.obs = observations()
        cls.reference, cls.policy = policy_pair(cls.obs)
        cls.reference.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
        cls.record = migrate_checkpoint(cls.policy, CHECKPOINT, expected_sha256=SHA)

    def test_3906_parity_and_new_input_invariance(self):
        with torch.no_grad():
            old = self.reference.act_inference(self.obs)
            new = self.policy.act_inference(self.obs)
            torch.testing.assert_close(old, new, atol=1e-6, rtol=0)
            changed = self.obs.clone()
            for key in ('nav_wide','nav_near','nav_features','nav_privileged'): changed[key] = torch.randn_like(changed[key]) * 3
            torch.testing.assert_close(old, self.policy.act_inference(changed), atol=1e-6, rtol=0)
        self.assertEqual(self.record['source_iteration'],3906)
        self.assertEqual(self.record['nav_iteration'],0)

    def test_migration_rejects_sha_and_unapproved_keys(self):
        with self.assertRaises(ValueError): migrate_checkpoint(self.policy,CHECKPOINT,expected_sha256='0'*64)
        source = self.reference.state_dict()
        with self.assertRaises(ValueError): migrate_state_dict({**source,'unexpected':torch.ones(1)},self.policy.state_dict())
        with self.assertRaises(ValueError): migrate_state_dict({k:v for k,v in source.items() if k!='actor.0.bias'},self.policy.state_dict())

    def test_A0_freeze_and_gradient_sequence(self):
        p = copy.deepcopy(self.policy)
        groups, manifest = apply_manifest(p,ROOT/'configs/gast_nav/trainable_A0.yaml',learning_rate=.001)
        optimizer = torch.optim.Adam(groups)
        old_ids={id(v) for g in groups for v in g['params']}
        self.assertNotIn(id(p.actor[0].weight), old_ids)
        self.assertFalse(p.nav_cenet_step); self.assertFalse(p.nav_aux_step)
        loss=p.act_inference(self.obs).square().mean(); loss.backward()
        self.assertGreater(p.actor[0].W_new.grad.abs().sum().item(),0)
        self.assertEqual(p.nav_wide_encoder[0].weight.grad.abs().sum().item(),0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        p.act_inference(self.obs).square().mean().backward()
        self.assertGreater(p.nav_wide_encoder[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(p.nav_near_encoder[0].weight.grad.abs().sum().item(),0)
        p.update_normalization(self.obs)
        assert_frozen(p,manifest['frozen'])
        self.assertGreater(p.nav_normalizer.count.item(),0)
        self.assertTrue(set(map(id,p.nav_wide_encoder.parameters())).isdisjoint(map(id,p.nav_critic_wide.parameters())))

    def test_A_groups(self):
        p=copy.deepcopy(self.policy)
        groups,_=apply_manifest(p,ROOT/'configs/gast_nav/trainable_A.yaml',learning_rate=.001)
        self.assertTrue(all(v.requires_grad for v in p.parameters()))
        self.assertEqual({g['gd_lab_group']:g['lr'] for g in groups},{'old':.0001,'new':.001})
