"""CPU unit tests for the DreamWaQ RL stack: no simulator, a random dummy env."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from gd_lab.rl import DreamwaqActorCritic, DreamwaqRunner

NUM_ENVS = 8
NUM_ACTIONS = 12
HISTORY = 5
TERM_DIMS = [3, 3, 3, 12, 12, 12]
ONE_STEP = sum(TERM_DIMS)
POLICY_DIM = ONE_STEP * HISTORY
CRITIC_DIM = 294
V_SLICE = (45, 48)


def _obs(num_envs: int = NUM_ENVS) -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(num_envs, POLICY_DIM), "critic": torch.randn(num_envs, CRITIC_DIM)},
        batch_size=[num_envs],
    )


def _policy_cfg() -> dict:
    return dict(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        history_length=HISTORY,
        policy_term_dims=list(TERM_DIMS),
        velocity_target_slice=V_SLICE,
        cenet_latent_dim=16,
        cenet_encoder_hidden_dims=[32, 16],
        cenet_decoder_hidden_dims=[16, 32],
        adaboot_enabled=True,
        adaboot_min_updates=1,
        adaboot_ema=0.8,
    )


def _make_policy() -> DreamwaqActorCritic:
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    return DreamwaqActorCritic(_obs(), obs_groups, NUM_ACTIONS, **_policy_cfg())


def test_actor_input_width():
    policy = _make_policy()
    assert policy.one_step_obs_dim == ONE_STEP
    assert policy.cenet.code_dim == 3 + 16
    assert policy.actor[0].in_features == ONE_STEP + 19


def test_act_shapes_and_inference():
    policy = _make_policy()
    obs = _obs()
    actions = policy.act(obs)
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert policy.evaluate(obs).shape == (NUM_ENVS, 1)
    mean_actions = policy.act_inference(obs)
    assert mean_actions.shape == (NUM_ENVS, NUM_ACTIONS)


def test_adaboot_injection_uses_ground_truth():
    policy = _make_policy()
    policy.cenet.inject_gt.fill_(1)
    obs = _obs()
    actor_in = policy._actor_input(obs, inference=False)
    code_velocity = actor_in[..., ONE_STEP : ONE_STEP + 3]
    assert torch.allclose(code_velocity, policy.velocity_target(obs))
    # Inference never injects, and never consumes the coin.
    actor_in_inf = policy._actor_input(obs, inference=True)
    assert not torch.allclose(actor_in_inf[..., ONE_STEP : ONE_STEP + 3], policy.velocity_target(obs))


def test_adaboot_coin_resampled_once_per_iteration():
    policy = _make_policy()
    policy.cenet.adaboot_update_count.fill_(10)
    policy.cenet.bootstrap_prob.fill_(0.0)  # always inject once resampled
    policy.notify_iteration_done()
    with torch.inference_mode():
        policy.act(_obs())
    assert policy.cenet.should_inject_gt()
    assert not policy._adaboot_pending_resample


def test_cenet_update_learns():
    policy = _make_policy()
    history = torch.randn(256, POLICY_DIM)
    velocity = torch.randn(256, 3)
    target = torch.randn(256, ONE_STEP)
    first = policy.cenet.update(history, velocity, target)
    for _ in range(50):
        last = policy.cenet.update(history, velocity, target)
    assert last["cenet_velocity"] < first["cenet_velocity"]
    assert last["cenet_recon"] < first["cenet_recon"]


class _DummyEnv:
    """Random-observation vectorized env implementing the rsl_rl VecEnv surface."""

    def __init__(self):
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.device = "cpu"
        self.max_episode_length = 24
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.cfg = {}
        self._step = 0

    def get_observations(self) -> TensorDict:
        return _obs()

    def step(self, actions: torch.Tensor):
        self._step += 1
        self.episode_length_buf += 1
        rewards = torch.randn(NUM_ENVS)
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        return _obs(), rewards, dones, {}


def _train_cfg() -> dict:
    return {
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 8,
        "save_interval": 10_000,
        "logger": "tensorboard",
        "policy": {"class_name": "gd_lab.rl.actor_critic:DreamwaqActorCritic", **_policy_cfg()},
        "algorithm": {
            "class_name": "gd_lab.rl.ppo:DreamwaqPPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.01,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def test_runner_learns_and_resumes(tmp_path):
    # Note: the upstream runner requires a log_dir (it saves at iteration 0).
    runner = DreamwaqRunner(_DummyEnv(), _train_cfg(), log_dir=str(tmp_path / "log"), device="cpu")
    runner.learn(num_learning_iterations=3)
    # The CENet aux pass ran and produced finite schedule state.
    assert torch.isfinite(runner.alg.policy.cenet.bootstrap_prob)

    runner.alg.learning_rate = 3.0e-5
    ckpt = str(tmp_path / "model.pt")
    runner.save(ckpt)

    fresh = DreamwaqRunner(_DummyEnv(), _train_cfg(), log_dir=str(tmp_path / "log2"), device="cpu")
    fresh.load(ckpt)
    assert fresh.alg.learning_rate == 3.0e-5
    assert fresh.current_learning_iteration == runner.current_learning_iteration + 1
    fresh.learn(num_learning_iterations=1)
