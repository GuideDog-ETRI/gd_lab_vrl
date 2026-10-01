"""CPU regression tests: no simulator or robot is started."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from gd_lab.residuals.bavrl import BAVRL, ResidualConfig, load_blind_teacher
from gd_lab.residuals.bavrl.export import BAVRLActorExport, BAVRLVisionExport
from gd_lab.residuals.bavrl.ppo import ResidualPPO, gae
from gd_lab.rl.actor_critic import DreamwaqActorCritic


def observations():
    return TensorDict({"policy": torch.randn(2, 230), "critic": torch.randn(2, 298)}, batch_size=[2])


def make_model():
    teacher = DreamwaqActorCritic(
        observations(), {"policy": ["policy"], "critic": ["critic"]}, 12,
        history_length=5, policy_term_dims=[3, 3, 3, 12, 12, 12, 1],
        velocity_target_slice=(45, 48), actor_history_steps=4, actor_obs_normalization=True,
        critic_obs_normalization=True, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
        cenet_encoder_hidden_dims=[32, 16], cenet_decoder_hidden_dims=[16, 32])
    return BAVRL(teacher)


def batch_inputs():
    return (observations(), torch.rand(2, 4, 2, 45, 80), torch.zeros(2, 64),
            torch.zeros(2, 12), torch.zeros(2), torch.ones(2, dtype=torch.bool))


def test_initial_anchor_and_limits():
    model = make_model().train()
    inputs = batch_inputs()
    base, dist, _, _, _, good = model(*inputs)
    action, delta = model.compose(base, dist.mean, inputs[3], good)
    assert torch.equal(action, model.teacher.act_inference(inputs[0]))
    assert torch.count_nonzero(delta) == 0
    _, delta = model.compose(base, torch.full_like(base, 100.), inputs[3], good)
    assert delta.abs().max() <= model.config.slew
    assert not model.teacher.training
    assert not any(p.requires_grad for p in model.teacher.parameters())


@pytest.mark.parametrize("age", [1., float("nan"), -1.])
def test_invalid_age_falls_back(age):
    model = make_model()
    obs, frames, hidden, previous, _, available = batch_inputs()
    previous += .1
    base, dist, _, _, _, good = model(obs, frames, hidden, previous, torch.full((2,), age), available)
    action, delta = model.compose(base, dist.sample(), previous, good)
    assert torch.equal(action, base)
    assert not delta.any()


def test_invalid_frames_and_missing():
    model = make_model()
    obs, frames, hidden, previous, age, available = batch_inputs()
    frames[0, 0, 0, 0, 0] = float("nan")
    available[1] = False
    assert not model(obs, frames, hidden, previous, age, available)[-1].any()


def test_missing_video_bypasses_encoder_and_nonzero_learned_residual(monkeypatch):
    model = make_model()
    with torch.no_grad():
        model.residual[-1].bias.fill_(3.)
    def forbidden(*args):
        raise AssertionError("Visual encoder must not run without images")
    monkeypatch.setattr(model.vision, "encode", forbidden)
    obs, _, hidden, previous, age, available = batch_inputs()
    previous += .15
    hidden += .5
    base, dist, _, memory, _, good = model(obs, None, hidden, previous, age, available)
    action, delta = model.compose(base, dist.sample(), previous, good)
    assert torch.equal(action, model.teacher.act_inference(obs))
    assert not delta.any()
    assert not memory.any()


def test_ppo_freezes_entire_teacher_and_replays_log_prob():
    model = make_model().train()
    original = deepcopy(model.teacher.state_dict())
    learner = ResidualPPO(model)
    obs, frames, hidden, previous, age, available = batch_inputs()
    with torch.no_grad():
        _, dist, _, _, _, _ = model(obs, frames, hidden, previous, age, available)
        raw = dist.sample()
        old = dist.log_prob(raw).sum(-1)
    batch = dict(obs=obs, frames=frames, hidden=hidden, previous=previous, age=age,
                 available=available, raw=raw, log_prob=old, advantage=torch.tensor([1., -1.]),
                 terrain=torch.cat((torch.zeros(2, 187), torch.ones(2, 187)), -1),
                 supervised=available, **{"return": torch.ones(2)})
    metrics = learner.update(batch)
    assert metrics["ratio_mean"] == pytest.approx(1.)
    assert all(torch.equal(original[k], v) for k, v in model.teacher.state_dict().items())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert model.residual[-1].weight.grad.abs().sum() > 0
    assert model.vision.cnn[0].weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_and_rejection(tmp_path):
    model = make_model()
    inputs = batch_inputs()
    file = tmp_path / "bavrl.pt"
    torch.save(model.checkpoint("hash"), file)
    saved = torch.load(file, weights_only=True)
    other = deepcopy(model)
    other.restore(saved, "hash")
    assert torch.equal(model(*inputs)[1].mean, other(*inputs)[1].mean)
    with pytest.raises(ValueError):
        other.restore(saved, "wrong")
    saved["model"]["teacher.actor.0.weight"] += 1
    with pytest.raises(ValueError, match="teacher state"):
        other.restore(saved, "hash")


def test_gae_stops_at_episode_boundary():
    r = torch.tensor([[1.], [100.]])
    a, _ = gae(r, torch.zeros_like(r), torch.zeros_like(r), torch.tensor([[True], [True]]))
    assert torch.equal(a, r)


def test_config_rejects_unbounded():
    with pytest.raises(ValueError):
        ResidualConfig(bound=float("inf"))


def test_split_deploy_matches_training_with_held_frame():
    model = make_model().eval()
    with torch.no_grad():
        model.residual[-1].weight.normal_(std=.01)
    actor, vision = BAVRLActorExport(model), BAVRLVisionExport(model)
    obs, frames, hidden, previous, age, available = batch_inputs()
    age += .15
    capture_age = torch.full_like(age, .03)
    deploy_hidden = hidden.clone()
    deploy_hidden[:, 63] = capture_age
    latent, _ = vision(frames, deploy_hidden)
    history = obs['policy'][:, torch.argsort(actor.reorder)]
    direct = obs['policy'][:, model.teacher.latest_idx]
    for valid in (True, False):
        available[:] = valid
        packed = torch.cat((latent, previous, age[:, None], available.float()[:, None]), -1)
        actual, _, delta = actor(direct, history, packed)
        base, dist, _, _, _, good = model(obs, frames, hidden, previous, age, available, capture_age)
        expected, expected_delta = model.compose(base, dist.mean, previous, good)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(delta, expected_delta)


def test_real_38000_checkpoint():
    root = Path(__file__).resolve().parents[1] / "checkpoints/teachers/blind/arm4_38000"
    checkpoint = root / "teacher/model_38000.pt"
    if not checkpoint.is_file():
        pytest.skip("Local teacher bundle unavailable")
    obs = observations()
    teacher, digest = load_blind_teacher(checkpoint, root / "teacher/params/agent.yaml", obs)
    model = BAVRL(teacher)
    inputs = list(batch_inputs())
    inputs[0] = obs
    base, dist, _, _, _, good = model(*inputs)
    assert len(digest) == 64
    assert torch.equal(model.compose(base, dist.mean, inputs[3], good)[0], teacher.act_inference(obs))
