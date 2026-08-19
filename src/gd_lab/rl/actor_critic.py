"""DreamWaQ actor-critic on top of the upstream rsl_rl ActorCritic."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict

from .cenet import CENet


class DreamwaqActorCritic(ActorCritic):
    """Actor input = cat(latest one-step obs, CENet code); asymmetric privileged critic.

    The policy observation group is the flattened proprio history (term-major, each
    term's history oldest -> newest). The CENet encodes the full history; the actor
    additionally receives the latest one-step slice. The code enters the actor
    detached: the CENet is trained only by its own decoupled loss, never by PPO.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        history_length: int,
        policy_term_dims: list[int],
        velocity_target_slice: tuple[int, int],
        cenet_latent_dim: int = 16,
        cenet_encoder_hidden_dims: list[int] | None = None,
        cenet_decoder_hidden_dims: list[int] | None = None,
        cenet_beta: float = 1.0,
        cenet_learning_rate: float = 1.0e-3,
        cenet_velocity_loss_coef: float = 1.0,
        cenet_reconstruction_loss_coef: float = 1.0,
        adaboot_enabled: bool = True,
        adaboot_min_updates: int = 10,
        adaboot_ema: float = 0.8,
        **kwargs: Any,
    ) -> None:
        if len(obs_groups["policy"]) != 1 or len(obs_groups["critic"]) != 1:
            raise ValueError(f"Expected exactly one policy and one critic obs group, got {obs_groups}")
        # The CENet reconstruction and velocity targets pass through the actor
        # observation normalizer; without it they chase raw unbounded values.
        if not kwargs.get("actor_obs_normalization", False):
            raise ValueError("DreamwaqActorCritic requires actor_obs_normalization=True")

        super().__init__(obs, obs_groups, num_actions, **kwargs)

        policy_group = obs_groups["policy"][0]
        actor_obs_dim = obs[policy_group].shape[-1]
        one_step = int(sum(policy_term_dims))
        if one_step * history_length != actor_obs_dim:
            raise ValueError(
                f"Obs spec mismatch: sum(term dims)={one_step} x history={history_length} "
                f"!= policy obs dim {actor_obs_dim}"
            )
        self.history_length = int(history_length)
        self.one_step_obs_dim = one_step
        self._velocity_target_slice = slice(*velocity_target_slice)

        # Flat indices of the newest history step in the term-major layout.
        latest = []
        offset = 0
        for dim in policy_term_dims:
            start = offset + (history_length - 1) * dim
            latest.extend(range(start, start + dim))
            offset += dim * history_length
        self.register_buffer("latest_idx", torch.tensor(latest, dtype=torch.long), persistent=False)

        self.cenet = CENet(
            input_dim=actor_obs_dim,
            obs_dim_per_step=one_step,
            latent_dim=cenet_latent_dim,
            velocity_dim=self._velocity_target_slice.stop - self._velocity_target_slice.start,
            encoder_hidden_dims=cenet_encoder_hidden_dims or [128, 64],
            decoder_hidden_dims=cenet_decoder_hidden_dims or [64, 128],
            beta=cenet_beta,
            learning_rate=cenet_learning_rate,
            velocity_loss_coef=cenet_velocity_loss_coef,
            reconstruction_loss_coef=cenet_reconstruction_loss_coef,
            adaboot_enabled=adaboot_enabled,
            adaboot_min_updates=adaboot_min_updates,
            adaboot_ema=adaboot_ema,
        )

        # Replace the actor built by the base class (its input was the full history).
        self.actor = MLP(
            one_step + self.cenet.code_dim,
            num_actions,
            kwargs.get("actor_hidden_dims", [512, 256, 128]),
            kwargs.get("activation", "elu"),
        )

        # Set by the algorithm after each update; the first rollout act of the
        # next iteration (grad disabled) resamples the AdaBoot coin once.
        self._adaboot_pending_resample = False

    # -- actor input -------------------------------------------------------
    def _normalized_history(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_obs_normalizer(obs[self.obs_groups["policy"][0]])

    def normalize_latest(self, raw_latest: torch.Tensor) -> torch.Tensor:
        """Apply the actor normalizer's latest-step statistics to a raw one-step obs."""
        norm = self.actor_obs_normalizer
        if not hasattr(norm, "_mean"):
            return raw_latest
        mean = norm._mean[0, self.latest_idx]
        std = norm._std[0, self.latest_idx]
        return (raw_latest - mean) / (std + norm.eps)

    def velocity_target(self, obs: TensorDict) -> torch.Tensor:
        """Ground-truth velocity sliced from the raw critic group (obs-manager scaling included)."""
        return obs[self.obs_groups["critic"][0]][..., self._velocity_target_slice].detach()

    def _should_inject_gt(self, inference: bool) -> bool:
        if inference:
            return False
        if not torch.is_grad_enabled() and self._adaboot_pending_resample:
            self.cenet.resample_adaboot()
            self._adaboot_pending_resample = False
        return self.cenet.should_inject_gt()

    def _actor_input(self, obs: TensorDict, *, inference: bool) -> torch.Tensor:
        history = self._normalized_history(obs)
        latest = history[..., self.latest_idx]
        out = self.cenet(history, deterministic=inference)
        inject = self._should_inject_gt(inference)
        velocity_gt = self.velocity_target(obs) if inject else None
        code = self.cenet.get_code(out, velocity_gt=velocity_gt, inject_gt_velocity=inject)
        return torch.cat((latest, code.detach()), dim=-1)

    # -- rsl_rl API --------------------------------------------------------
    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        self.update_distribution(self._actor_input(obs, inference=False))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self._actor_input(obs, inference=True))

    def notify_iteration_done(self) -> None:
        self._adaboot_pending_resample = True
