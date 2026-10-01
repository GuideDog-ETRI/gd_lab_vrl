"""CPU tests for geometry supervision, gradients, memory and legacy coexistence."""

from types import SimpleNamespace

import torch

from gd_lab.rl.attention_distillation import AttentionDistillation
from gd_lab.rl.perception import CameraPerceptionEncoder
from gd_lab.rl.perception_attention import GridAttentionStudent, spatial_loss, spatial_targets


def test_shapes_invalid_and_memory():
    torch.set_num_threads(2)
    m = GridAttentionStudent()
    x = torch.ones(2, 4, 2, 45, 80)
    x[0, 0, 0, 0, 0] = float("nan")
    h = m.init_hidden(2, "cpu")
    z, h1 = m(x, h)
    assert z.shape == (2, 32) and h1.shape == (2, 64)
    assert torch.isfinite(z).all() and not h1[:, 63].any()
    assert not torch.allclose(m(x, h1)[0], z)
    h[:, 63] = .5
    assert not torch.allclose(m(x, h)[0], z)
    legacy = CameraPerceptionEncoder()
    assert legacy(torch.zeros_like(x), h)[0].shape == (2, 32)


def test_visible_signed_edges_and_hidden_labels():
    terrain = torch.zeros(1, 374)
    terrain[:, 187:] = 1
    terrain[:, :187].reshape(1, 11, 17)[:, :, 8:] = -.5
    target, mask = spatial_targets(terrain)
    assert target.reshape(1, 11, 17, 5)[0, 0, 7, 2] == 1  # rising terrain
    terrain[:, 187:] = 0
    _, mask = spatial_targets(terrain)
    assert not mask[..., [0, 2, 3, 4]].any()
    pred = torch.zeros(1, 187, 5, requires_grad=True)
    spatial_loss(pred, terrain).backward()
    assert torch.isfinite(pred.grad).all()
    assert not pred.grad[..., [0, 2, 3, 4]].any()


def test_frozen_actor_gradient_and_reset():
    torch.manual_seed(2)
    m = GridAttentionStudent()
    actor = torch.nn.Linear(36, 12).requires_grad_(False)
    trainer = AttentionDistillation(SimpleNamespace(actor=actor), m, 2, "cpu")
    terrain = torch.zeros(2, 374)
    terrain[:, 187:] = 1
    extra = (torch.zeros(2, 4), torch.zeros(2, 12), terrain)
    z, h, loss = trainer.update(torch.rand(2, 4, 2, 45, 80), m.init_hidden(2, "cpu"),
                                torch.arange(2), extra, .05)
    loss.backward()
    assert m.key.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in actor.parameters())
    assert trainer.ready.all()
    trainer.reset(torch.tensor([True, False]))
    assert not trainer.ready[0] and trainer.ready[1]
    assert not trainer.latent[0].any()


def test_checkpoint_roundtrip(tmp_path):
    m = GridAttentionStudent().eval()
    path = tmp_path / "student.pt"
    torch.save({"model": m.state_dict()}, path)
    other = GridAttentionStudent().eval()
    other.load_state_dict(torch.load(path, weights_only=True)["model"], strict=True)
    x, h = torch.rand(1, 4, 2, 45, 80), torch.zeros(1, 64)
    assert torch.equal(m(x, h)[0], other(x, h)[0])


def test_bptt_and_torchscript(tmp_path):
    m = GridAttentionStudent()
    x = torch.rand(2, 4, 2, 45, 80)
    h = m.init_hidden(2, "cpu")
    loss = 0
    for i in range(3):
        z, memory = m(x, h)
        h = h.index_copy(0, torch.arange(2), memory)
        if i == 1:
            h = h * torch.tensor([[0.], [1.]])
        loss = loss + z.square().mean()
    loss.backward()
    assert m.gru.weight_hh.grad.abs().sum() > 0
    traced = torch.jit.trace(m.eval(), (x[:1], h[:1].detach()))
    assert torch.allclose(traced(x[:1], h[:1])[0], m(x[:1], h[:1])[0])


def test_onnx_export(tmp_path):
    import numpy as np
    import onnxruntime as ort

    from gd_lab.deploy.export_student_vrl import export_student_vrl

    m = GridAttentionStudent().eval()
    _, path = export_student_vrl(m, str(tmp_path / "policy.onnx"))
    x = torch.rand(1, 4, 2, 45, 80)
    h = torch.rand(1, 64)
    expected = m(x, h)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map == {
        "camel.student_arch": "grid_attention_v1",
        "camel.student_age": "hidden63_seconds_clipped_0_1",
    }
    actual = session.run(None, {"frames": x.numpy(), "hidden_in": h.numpy()})
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e.detach().numpy(), atol=1e-5, rtol=1e-4)
