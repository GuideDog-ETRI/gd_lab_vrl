"""Transport timing, capture-time labels, and reset/out-of-order isolation."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from gd_lab.core.camera_contract import camera_contract_for_policy
from gd_lab.core.camera_transport import CameraTransport, CameraTransportConfig, delivery_mask


def test_arm4_contract_preserves_physical_interval():
    arm4 = camera_contract_for_policy("vendor_legacy", 0.01)
    assert arm4.period_steps == 8 and arm4.policy_dt == 0.01
    assert arm4.period_steps * arm4.policy_dt == 0.08
    assert camera_contract_for_policy("vendor_legacy", 0.02).period_steps == 4


@pytest.mark.parametrize("kwargs", [
    {"interval_ms": (0, 80)}, {"interval_ms": (100, 70)},
    {"delay_ms": (-1, 50)}, {"delay_ms": (0, float("nan"))},
    {"drop_probability": 1}, {"drop_probability": -0.1},
])
def test_invalid_transport_settings(kwargs):
    with pytest.raises(ValueError):
        CameraTransportConfig(**kwargs)


def test_timing_is_seeded_bounded_and_delivers_only_once():
    def rollout(seed):
        transport = CameraTransport(CameraTransportConfig(), 0.01, seed)
        step, intervals, packets = 0, [], []
        for _ in range(100):
            next_step = transport.next_capture(step)
            intervals.append(next_step - step)
            step = next_step
            transport.capture(step, ("label", step))
        packets = transport.receive(step + 5)
        assert not transport.receive(step + 6)
        assert set(intervals) == {7, 8, 9, 10}
        assert len(packets) + transport.dropped == 100
        assert transport.dropped > 0
        assert all(0 <= p.delivery_step - p.capture_step <= 5 for p in packets)
        assert all(p.payload[1] == p.capture_step for p in packets)
        return intervals, packets, transport.dropped
    assert rollout(42) == rollout(42)


def test_fixed_timing_no_delay_no_drop():
    transport = CameraTransport(CameraTransportConfig((80, 80), (0, 0), 0), 0.01, 1)
    assert transport.next_capture(10) == 18
    transport.capture(18, "at-capture")
    assert not transport.receive(17)
    assert transport.receive(18)[0].payload == "at-capture"


def test_reordered_delivery_cannot_rewind_student_state():
    transport = CameraTransport(CameraTransportConfig((80, 80), (0, 100), 0), 0.01, 1)
    delays = iter([10, 0])
    transport.rng = NS(random=lambda: 0.5, randint=lambda *bounds: next(delays))
    transport.capture(8, "old")
    transport.capture(16, "new")
    packet = transport.receive(16)[0]
    assert packet.payload == "new"
    last = torch.tensor([packet.capture_step])
    late_packet = transport.receive(18)[0]
    assert late_packet.payload == "old"
    assert not delivery_mask(late_packet.capture_step, torch.tensor([0]), torch.tensor([0]), last).any()


def test_nonrepresentable_substep_interval_is_rejected():
    with pytest.raises(ValueError, match="integer policy step"):
        CameraTransport(CameraTransportConfig((71, 79)), 0.01, 42)


def test_delay_does_not_relabel_and_resets_and_reordering_are_rejected():
    transport = CameraTransport(CameraTransportConfig((80, 80), (30, 30), 0), 0.01, 1)
    epochs = torch.tensor([0, 0])
    label = torch.tensor([8., 8.])
    transport.capture(8, (label.clone(), epochs.clone()))
    label.fill_(11)
    epochs[0] += 1
    assert not transport.receive(10)
    packet = transport.receive(11)[0]
    saved_label, saved_epochs = packet.payload
    assert saved_label.tolist() == [8., 8.]
    assert delivery_mask(packet.capture_step, saved_epochs, epochs, torch.tensor([-1, -1])).tolist() == [False, True]
    assert not delivery_mask(packet.capture_step, saved_epochs, epochs, torch.tensor([9, 9])).any()


def test_capture_function_freezes_frames_and_teacher_labels():
    # Load the production packet builder without booting Isaac. The strict
    # camera/calibration contract has dedicated CPU tests in test_alignment_unit.py.
    path = Path(__file__).parents[1] / "scripts/distill_student.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "capture_teacher_packet"]
    frame = torch.ones(2, 4, 2, 3, 3)
    terrain = torch.ones(2, 374)
    latent = torch.tensor([[1., 2.], [3., 4.]])
    env = NS(common_step_counter=8, _vrl_camera_snapshot=(frame,),
             _vrl_camera_snapshot_steps=torch.tensor([8, 8]),
             scene=NS(env_origins=None))
    teacher = NS(terrain_latent=lambda obs: latent, _height_scan_slice=slice(0, 187),
                 _actor_input=lambda obs, inference: torch.ones(2, 5),
                 act_inference=lambda obs: torch.ones(2, 3),
                 terrain_encoder=NS(grid_shape=(11, 17)))
    def validate(e, obs, contract):
        if not (e._vrl_camera_snapshot_steps == e.common_step_counter).all():
            raise RuntimeError("camera frame is stale")
        return obs["terrain"]
    ns = dict(torch=torch, validate_teacher_camera_capture=validate,
              terrain_family_gate=lambda *a: torch.zeros(2),
              augment_student_camera_frames=lambda frames, *a: frames.clone(),
              height_discontinuity_metres=lambda h, *a: h.mean(-1))
    exec(compile(tree, str(path), "exec"), ns)
    payload = ns["capture_teacher_packet"](env, {"terrain": terrain}, teacher,
                                             torch.zeros(2), None, None, object())
    expected_terrain = terrain.clone()
    frame.zero_(); latent.zero_(); terrain.zero_()
    assert payload[0].sum() > 0 and payload[1].sum() == 10
    assert payload[2].shape == (2, 374) and torch.equal(payload[2], expected_terrain)
    assert payload[8].shape == (2, 3)
    env._vrl_camera_snapshot_steps[0] = 7
    with pytest.raises(RuntimeError, match="stale"):
        ns["capture_teacher_packet"](env, {"terrain": terrain}, teacher,
                                      torch.zeros(2), None, None, object())
