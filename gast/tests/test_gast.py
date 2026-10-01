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
