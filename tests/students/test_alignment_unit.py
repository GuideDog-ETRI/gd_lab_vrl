"""CPU unit checks for capture alignment and visible-only RVLD/GAVD targets."""
import unittest
from types import SimpleNamespace
import torch
from gd_lab.core.camera_contract import load_camera_contract
from gd_lab.core.camera_geometry import mounted_camera_world_poses
from gd_lab.students.alignment import validate_teacher_camera_capture
from gd_lab.students.gavd.distillation import AttentionDistillation
from gd_lab.students.gavd.model import GridAttentionStudent, spatial_loss
from gd_lab.students.rvld.distillation import TerrainAlignmentHead

class AlignmentTests(unittest.TestCase):
    def _capture(self):
        n=2; contract=load_camera_contract("vendor_legacy")
        body_pos=torch.zeros(n,1,3); body_quat=torch.zeros(n,1,4); body_quat[...,0]=1
        robot=SimpleNamespace(body_names=["trunk"],data=SimpleNamespace(body_pos_w=body_pos,body_quat_w=body_quat))
        positions,rotations=mounted_camera_world_poses(body_pos[:,0],body_quat[:,0],contract)
        k=torch.tensor([[contract.fx,0,contract.width/2],[0,contract.fy,contract.height/2],[0,0,1.]]).expand(n,4,3,3).clone()
        terrain=torch.zeros(n,374)
        env=SimpleNamespace(common_step_counter=7,cfg=SimpleNamespace(camera_profile="vendor_legacy"),scene={"robot":robot},
            _vrl_camera_snapshot=(torch.zeros(n,4,2,45,80),torch.ones(n,4,45,80),positions,rotations,k),
            _vrl_camera_snapshot_steps=torch.full((n,),7,dtype=torch.long),
            _vrl_teacher_terrain_capture_steps=torch.full((n,),7,dtype=torch.long),
            _vrl_teacher_terrain_snapshot=terrain.clone(),_vrl_teacher_terrain_contract=contract.manifest())
        return env,{"terrain":terrain},contract

    def test_exact_capture_passes(self):
        env,obs,contract=self._capture()
        self.assertTrue(torch.equal(validate_teacher_camera_capture(env,obs,contract),obs["terrain"]))

    def test_stale_camera_fails(self):
        env,obs,contract=self._capture(); env._vrl_camera_snapshot_steps[1]=6
        with self.assertRaisesRegex(RuntimeError,"camera frame is stale"):
            validate_teacher_camera_capture(env,obs,contract)

    def test_stale_teacher_target_fails(self):
        env,obs,contract=self._capture(); env._vrl_teacher_terrain_capture_steps[0]=6
        with self.assertRaisesRegex(RuntimeError,"Ray height/visibility target is stale"):
            validate_teacher_camera_capture(env,obs,contract)

    def test_intrinsics_mismatch_fails(self):
        env,obs,contract=self._capture(); env._vrl_camera_snapshot[4][:,:,0,0]*=.8
        with self.assertRaisesRegex(RuntimeError,"intrinsics differ"):
            validate_teacher_camera_capture(env,obs,contract)

    def test_rvld_aux_head_and_visible_only_loss(self):
        head=TerrainAlignmentHead(); hidden=torch.randn(3,64,requires_grad=True); prediction=head(hidden)
        self.assertEqual(tuple(prediction.shape),(3,187,5))
        terrain=torch.zeros(3,374); visible=torch.zeros(187,dtype=torch.bool); visible[[4,5,21,22]]=True
        terrain[:,187:]=visible.float(); terrain[:,:187]=torch.arange(187).float()[None]*.1
        total,height,vis_loss,fraction=spatial_loss(prediction,terrain,return_components=True)
        total.backward(); self.assertTrue(torch.isfinite(total)); self.assertGreater(head.decoder[-1].weight.grad.abs().sum().item(),0)
        self.assertAlmostEqual(fraction.item(),visible.float().mean().item(),places=6)
        altered=terrain.clone(); altered[:,:187]=torch.where(visible[None],terrain[:,:187],torch.full_like(terrain[:,:187],1e5))
        _,height2,vis_loss2,_=spatial_loss(prediction.detach(),altered,return_components=True)
        torch.testing.assert_close(height,height2); torch.testing.assert_close(vis_loss,vis_loss2)

    def test_gavd_same_capture_targets_and_stale_latent_gate(self):
        class Teacher:
            def __init__(self):
                self.actor = torch.nn.Linear(36, 12)
                self.actor.requires_grad_(False)
            def _actor_input(self, obs, inference=True):
                return torch.zeros(1, 36)
            def act_inference(self, obs):
                return torch.ones(1, 12)
            def act_with_terrain_latent(self, obs, latent):
                return latent[:, :1].expand(-1, 12)
        teacher = Teacher()
        model = GridAttentionStudent()
        trainer = AttentionDistillation(teacher, model, 1, "cpu", warmup=0, ramp=1)
        terrain = torch.zeros(1, 374); terrain[:, 187:] = 1
        action = torch.zeros(1, 12)
        packet = trainer.capture({}, action, torch.zeros(1, 4), terrain)
        _, _, loss = trainer.update(torch.rand(1, 4, 2, 45, 80), model.init_hidden(1, "cpu"),
                                   torch.tensor([0]), packet, 0.05, 0.1)
        loss.backward()
        self.assertGreater(model.key.weight.grad.abs().sum().item(), 0)
        self.assertTrue(trainer.ready.all())
        trainer.actions({}, 1, 0.2)
        self.assertEqual(trainer.metrics["stale_fraction"], 0.0)
        trainer.actions({}, 1, 0.5)
        self.assertEqual(trainer.metrics["stale_fraction"], 1.0)
        self.assertTrue(torch.equal(trainer.actions({}, 1, 0.5), torch.ones(1, 12)))
if __name__=="__main__": unittest.main(verbosity=2)
