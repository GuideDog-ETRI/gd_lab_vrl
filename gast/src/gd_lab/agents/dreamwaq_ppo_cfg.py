"""RSL-RL runner configuration for blind DreamWaQ.

Class names are ``"<module>:<attr>"`` entry-point strings resolved by the runner;
observation-layout fields are derived from the DreamWaQ spec, never written by hand.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC, HISTORY_LENGTH, VELOCITY_TARGET_SLICE


@configclass
class DreamwaqActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "gd_lab.rl.actor_critic:DreamwaqActorCritic"

    # Layout (derived from the spec).
    history_length: int = HISTORY_LENGTH
    policy_term_dims: list[int] = [t.dim for t in DREAMWAQ_SPEC.policy.terms]
    velocity_target_slice: tuple[int, int] = VELOCITY_TARGET_SLICE
    # Fits inside the stored history, so the observation and ONNX contracts are
    # unchanged - but checkpoints do not resume across a change of this value.
    actor_history_steps: int = 4

    # CENet.
    cenet_latent_dim: int = 16
    cenet_encoder_hidden_dims: list[int] = [128, 64]
    cenet_decoder_hidden_dims: list[int] = [64, 128]
    cenet_beta: float = 1.0
    cenet_learning_rate: float = 1.0e-3
    cenet_velocity_loss_coef: float = 1.0
    cenet_reconstruction_loss_coef: float = 1.0

    # AdaBoot.
    adaboot_enabled: bool = True
    adaboot_min_updates: int = 10
    adaboot_ema: float = 0.8


@configclass
class DreamwaqAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "gd_lab.rl.ppo:DreamwaqPPO"

    # Above rsl-rl's hardcoded 1e-5: a run parked at the floor stops learning
    # while still reporting a healthy KL.
    min_learning_rate: float = 3.0e-5


@configclass
class DreamwaqRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "gd_lab.rl.runner:DreamwaqRunner"

    num_steps_per_env = 100
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "blind_rbq10_dreamwaq"
    logger = "wandb"
    wandb_project = "gd_lab"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = DreamwaqActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        # Required: the CENet feeds its inputs and reconstruction target through
        # the actor observation normalizer.
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = DreamwaqAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func="gd_lab.methods.dreamwaq.symmetry:compute_symmetric_states",
            mirror_loss_coeff=0.0,
        ),
    )
