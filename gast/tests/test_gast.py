import unittest
import torch
from gd_lab.gast.geometry import warp_memory, targets, reconstruction_loss
from gd_lab.gast.temporal import TemporalTerrainEncoder
from gd_lab.gast.student import GastStudent

torch.set_num_threads(2)

class GastTests(unittest.TestCase):
    def test_chunked_encoder_preserves_outputs_and_gradients(self):
        model = TemporalTerrainEncoder()
        x = torch.randn(5, 8*375)
        direct = model(x)
        direct.square().sum().backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        model.execution_chunk_size = 2
        chunked = model(x)
        torch.testing.assert_close(chunked, direct, atol=1e-6, rtol=1e-5)
        chunked.square().sum().backward()
        for p, gradient in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, gradient, atol=1e-5, rtol=1e-4)

    def test_warp_translation(self):
        memory = torch.zeros(1, 11, 17, 1)
        memory[0, 5, 10] = 1
        out = warp_memory(memory.flatten(1, 2), torch.zeros(1,3), torch.tensor([[.1,0.,0.]]))
        self.assertAlmostEqual(out.reshape(11,17)[5,9].item(), 1., places=4)

    def test_temporal_gradient_and_order(self):
        model = TemporalTerrainEncoder()
        history = torch.randn(2,8,375, requires_grad=True)
        out = model(history.flatten(1))
        self.assertEqual(out.shape,(2,32))
        out.square().mean().backward()
        self.assertGreater(history.grad[:,0].abs().sum().item(),0)
        self.assertFalse(torch.allclose(out, model(history.detach().flip(1).flatten(1))))

    def test_labels_do_not_call_unknown_a_gap(self):
        clean = torch.zeros(1,748)
        y, mask = targets(clean)
        self.assertFalse(mask[...,2].any())
        self.assertEqual(y[...,2].sum().item(),0)
        loss = reconstruction_loss(torch.zeros(1,187,6, requires_grad=True), clean)
        self.assertTrue(torch.isfinite(loss))

    def test_student_geometry_uses_only_teacher_visible_heights(self):
        clean = torch.zeros(1, 748)
        clean[:, 187:374] = 1  # all underlying scan points are valid
        clean[:, 561:748] = 1  # semantic labels are known, but mask still applies
        teacher = torch.zeros(1, 374)
        visibility = torch.zeros(187, dtype=torch.bool)
        visibility[[4, 5, 21, 22]] = True
        teacher[:, 187:] = visibility.float()
        teacher[:, :187] = torch.arange(187).float()[None] * 0.1
        labels, mask = targets(clean, teacher)
        self.assertTrue(torch.equal(mask[0, :, 0], visibility))
        self.assertTrue(torch.equal(mask[0, :, 1], torch.ones(187, dtype=torch.bool)))
        self.assertTrue(torch.equal(labels[0, :, 1].bool(), visibility))
        torch.testing.assert_close(labels[0, visibility, 0], teacher[0, :187][visibility] / 5)

        prediction = torch.zeros(1, 187, 6)
        loss_a, height_a, vis_a, _ = reconstruction_loss(prediction, clean, teacher, return_components=True)
        changed_hidden = clean.clone()
        changed_hidden[:, :187] = torch.randn(1, 187) * 1000
        changed_teacher = teacher.clone()
        hidden_heights = torch.randn(int((~visibility).sum())) * 1000
        changed_teacher[0, :187] = torch.where(
            visibility, teacher[0, :187], teacher.new_zeros(187).masked_scatter(~visibility, hidden_heights))
        loss_b, height_b, vis_b, _ = reconstruction_loss(prediction, changed_hidden, changed_teacher, return_components=True)
        torch.testing.assert_close(height_a, height_b)
        torch.testing.assert_close(vis_a, vis_b)
        torch.testing.assert_close(loss_a, loss_b)

    def test_teacher_visibility_target_is_independent_of_height_validity(self):
        clean = torch.zeros(1, 748)
        clean[:, 187:374] = 1
        clean[:, 187 + 5] = 0  # scanner height invalid, but teacher visibility remains known
        teacher = torch.zeros(1, 374)
        teacher[0, 5] = 0.25
        teacher[0, 187 + 5] = 1
        labels, mask = targets(clean, teacher)
        self.assertFalse(mask[0, 5, 0])  # no height regression at invalid scanner cell
        self.assertTrue(mask[0, 5, 1])   # still supervise the teacher visibility prediction
        self.assertTrue(labels[0, 5, 1])

    def test_all_unobserved_cells_have_no_height_loss_but_train_mask(self):
        clean = torch.zeros(2, 748)
        clean[:, 187:374] = 1
        teacher = torch.zeros(2, 374)
        pred = torch.randn(2, 187, 6, requires_grad=True)
        total, height, visibility, fraction = reconstruction_loss(pred, clean, teacher, return_components=True)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(height.item(), 0.)
        self.assertGreater(visibility.item(), 0.)
        self.assertEqual(fraction.item(), 0.)
        total.backward()
        self.assertEqual(pred.grad[..., 0].abs().sum().item(), 0.)
        self.assertGreater(pred.grad[..., 1].abs().sum().item(), 0.)

    def test_student_missing_and_history(self):
        model = GastStudent()
        frames = torch.rand(2,4,2,45,80)
        pose = torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]).repeat(2,1)
        hidden = model.init_hidden(2,'cpu')
        latent, memory, spatial, _, _ = model.encode(frames, hidden, pose, available=torch.zeros(2))
        self.assertEqual(torch.count_nonzero(latent).item(),0)
        self.assertEqual(spatial.shape,(2,187,6))
        out = model.encode(frames, memory, pose, age_seconds=.4)[0]
        self.assertEqual(torch.count_nonzero(out).item(),0)
        model.encode(frames, memory, pose)[0].square().mean().backward()
        self.assertIsNotNone(model.cell_gru.weight_hh.grad)

if __name__ == '__main__': unittest.main()
