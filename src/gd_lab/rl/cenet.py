"""Context Estimator Network: a proprioceptive-history beta-VAE with a supervised
velocity head, trained by its own optimizer, decoupled from PPO."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP
from torch import optim


class CENetOut(NamedTuple):
    velocity: torch.Tensor
    latent: torch.Tensor
    mean_latent: torch.Tensor
    logvar_latent: torch.Tensor
    decode: torch.Tensor


class CENet(nn.Module):
    """encoder(obs history) -> deterministic v_hat + latent z; decoder(cat(v_hat, z)) -> o_{t+1}.

    The velocity head is trained only by its MSE-to-ground-truth term (it enters the
    decoder detached), the latent/decoder only by reconstruction + beta * mean-KL.
    """

    def __init__(
        self,
        input_dim: int,
        obs_dim_per_step: int,
        latent_dim: int = 16,
        velocity_dim: int = 3,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (64, 128),
        beta: float = 1.0,
        learning_rate: float = 1.0e-3,
        velocity_target_limit: float = 10.0,
        recon_target_limit: float = 10.0,
        obs_history_limit: float = 10.0,
        loss_skip_threshold: float = 50.0,
        velocity_loss_coef: float = 1.0,
        reconstruction_loss_coef: float = 1.0,
        activation: str = "elu",
        adaboot_enabled: bool = False,
        adaboot_min_updates: int = 10,
        adaboot_ema: float = 0.8,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.obs_dim_per_step = int(obs_dim_per_step)
        self.latent_dim = int(latent_dim)
        self.velocity_dim = int(velocity_dim)
        self.beta = float(beta)
        self.velocity_loss_coef = float(velocity_loss_coef)
        self.reconstruction_loss_coef = float(reconstruction_loss_coef)
        self.code_dim = self.velocity_dim + self.latent_dim

        enc = list(encoder_hidden_dims)
        if not enc:
            raise ValueError("CENet requires non-empty encoder_hidden_dims")
        self.encoder = MLP(self.input_dim, enc[-1], enc[:-1] if len(enc) > 1 else [enc[0]], activation=activation)
        self.mean_latent = nn.Linear(enc[-1], self.latent_dim)
        self.logvar_latent = nn.Linear(enc[-1], self.latent_dim)
        self.mean_velocity = nn.Linear(enc[-1], self.velocity_dim)
        self.decoder = MLP(self.code_dim, self.obs_dim_per_step, list(decoder_hidden_dims), activation=activation)

        # AdaBoot: p_boot = 1 - tanh(CV(episode rewards)) is the probability of
        # trusting the estimate. The ground-truth injection decision is ONE coin
        # per learning iteration, applied uniformly to the rollout acts and the
        # PPO update forward of that iteration; per-step or rollout-only sampling
        # makes the recomputed log-probs disagree with the stored ones and the
        # adaptive learning rate collapses. Buffers so the schedule survives
        # checkpoint save/resume.
        self.adaboot_enabled = bool(adaboot_enabled)
        self.adaboot_min_updates = int(adaboot_min_updates)
        self.adaboot_ema = float(adaboot_ema)
        self.register_buffer("bootstrap_prob", torch.ones(()))
        self.register_buffer("adaboot_cv", torch.zeros(()))
        self.register_buffer("adaboot_update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("inject_gt", torch.zeros((), dtype=torch.long))

        # Outlier guards on the *targets*. velocity_target arrives in raw m/s
        # (ppo.py reads it from the unnormalised critic obs) and the robot is
        # commanded at most 1.2 m/s, so 10 m/s is pure headroom -- only a physics
        # blow-up reaches it. recon_target is already normalised, so its limit is
        # in sigma.
        self.velocity_target_limit = float(velocity_target_limit)
        self.recon_target_limit = float(recon_target_limit)
        self.obs_history_limit = float(obs_history_limit)
        # Last line of defence, and the only one that does not care which input
        # went bad. Normal cenet loss is ~0.5; the four observed blow-ups were
        # 7e3 / 5.7e4 / 2.2e4 / 3.2e4. 50 is 100x normal and still 140x below the
        # smallest blow-up.
        self.loss_skip_threshold = float(loss_skip_threshold)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs_history: torch.Tensor, deterministic: bool = False) -> CENetOut:
        """``deterministic`` affects the latent only; the velocity head is always deterministic."""
        feat = self.encoder(obs_history)
        mean_latent = self.mean_latent(feat)
        logvar_latent = torch.clamp(self.logvar_latent(feat), min=-10.0, max=10.0)
        velocity = self.mean_velocity(feat)
        if deterministic:
            z = mean_latent
        else:
            z = mean_latent + torch.randn_like(mean_latent) * torch.exp(0.5 * logvar_latent)
        decode = self.decoder(torch.cat((velocity.detach(), z), dim=-1))
        return CENetOut(velocity, z, mean_latent, logvar_latent, decode)

    def get_code(
        self,
        out: CENetOut,
        *,
        velocity_gt: torch.Tensor | None = None,
        inject_gt_velocity: bool = False,
    ) -> torch.Tensor:
        v = velocity_gt if (inject_gt_velocity and velocity_gt is not None) else out.velocity
        return torch.cat((v, out.latent), dim=-1)

    def kl_divergence(self, mean_latent: torch.Tensor, logvar_latent: torch.Tensor) -> torch.Tensor:
        """beta-VAE KL with MEAN reduction over batch and latent dims."""
        return -0.5 * torch.mean(1.0 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())

    # -- AdaBoot schedule -------------------------------------------------
    def update_bootstrap_from_episode_rewards(self, episode_rewards) -> None:
        """EMA-update p_boot from completed-episode total rewards; call once per iteration.

        Skipped until at least two episodes have completed (CV undefined; p_boot
        stays at its no-injection start value).
        """
        if not self.adaboot_enabled:
            return
        with torch.no_grad():
            r = torch.as_tensor(list(episode_rewards), dtype=torch.float32, device=self.bootstrap_prob.device)
            if r.numel() < 2:
                return
            cv = r.std(unbiased=False) / (r.mean().abs() + 1e-8)
            cv = torch.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)
            self.adaboot_cv.copy_(cv)
            self.bootstrap_prob.mul_(self.adaboot_ema).add_((1.0 - self.adaboot_ema) * (1.0 - torch.tanh(cv)))
            self.adaboot_update_count.add_(1)

    def should_inject_gt(self) -> bool:
        return self.adaboot_enabled and bool(self.inject_gt.item())

    def resample_adaboot(self) -> None:
        """Draw the next iteration's single injection coin (probability 1 - p_boot)."""
        if not self.adaboot_enabled or int(self.adaboot_update_count.item()) < self.adaboot_min_updates:
            self.inject_gt.fill_(0)
            return
        self.inject_gt.fill_(int((torch.rand(()) >= self.bootstrap_prob).item()))

    # -- decoupled optimizer step ----------------------------------------
    def update(
        self,
        obs_history: torch.Tensor,
        velocity_target: torch.Tensor,
        next_obs_target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One optimizer step. ``valid_mask`` restricts reconstruction to non-terminal rows."""
        # Guard the targets, not the gradients. A PhysX blow-up in a few envs puts
        # absurd values in both targets at once; one such batch destroyed the
        # estimator on the blind run (arm 1, 2026-09-20, iter 10,796 -- cenet loss
        # 0.605 -> 7124, policy down 600 iterations, see
        # logs/incident_notes/arm1_gpu0_seed42_40k.md INC-003).
        #
        # Gradient clipping is the reflex here and it does **not** work: Adam
        # already normalises the step, so clipping measured no effect at all
        # (3.13 vs 3.02 relative drift). Clamping the targets brings drift back to
        # the no-outlier baseline (1.31 vs 1.24). The counts are logged rather than
        # swallowed, so an upstream physics problem still shows up.
        vlim, rlim, hlim = self.velocity_target_limit, self.recon_target_limit, self.obs_history_limit
        n_clamped_velocity = int((velocity_target.abs() > vlim).sum())
        n_clamped_recon = int((next_obs_target.abs() > rlim).sum())
        n_clamped_history = int((obs_history.abs() > hlim).sum())
        velocity_target = velocity_target.clamp(-vlim, vlim)
        next_obs_target = next_obs_target.clamp(-rlim, rlim)
        # The encoder input needs the same guard as the targets. On 2026-09-21
        # (arm 1, iter 14,138) the targets were clamped and the loss was still
        # 31,531: velocity_target never exceeded its limit (clamp count 0) yet the
        # velocity loss was 11,036, so sqrt() puts the *prediction* near 105 m/s.
        # An extreme history drives the forward pass, not the targets.
        obs_history = obs_history.clamp(-hlim, hlim)

        out = self.forward(obs_history, deterministic=False)
        velocity_loss = F.mse_loss(out.velocity, velocity_target.detach())
        target = next_obs_target.detach()
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1).bool()
            if valid_mask.any():
                recon_loss = F.mse_loss(out.decode[valid_mask], target[valid_mask])
            else:
                recon_loss = out.decode.new_tensor(0.0)
        else:
            recon_loss = F.mse_loss(out.decode, target)
        kl_loss = self.kl_divergence(out.mean_latent, out.logvar_latent)
        loss = (
            self.velocity_loss_coef * velocity_loss
            + self.reconstruction_loss_coef * recon_loss
            + self.beta * kl_loss
        )
        # Clamping bounds the inputs but cannot bound everything: the loss is the
        # one place where every failure path becomes visible. If it is absurd,
        # the batch has nothing to teach -- take no step rather than a wrong one.
        skipped = not torch.isfinite(loss) or float(loss) > self.loss_skip_threshold
        self.optimizer.zero_grad(set_to_none=True)
        if not skipped:
            loss.backward()
            self.optimizer.step()

        metrics = {
            "cenet_clamped_velocity": float(n_clamped_velocity),
            "cenet_clamped_recon": float(n_clamped_recon),
            "cenet_clamped_history": float(n_clamped_history),
            "cenet_skipped_steps": float(skipped),
            "cenet_velocity": velocity_loss.item(),
            "cenet_recon": recon_loss.item(),
            "cenet_kl": kl_loss.item(),
            "cenet": loss.item(),
        }
        if self.adaboot_enabled:
            metrics["adaboot_p_boot"] = float(self.bootstrap_prob.item())
            metrics["adaboot_cv"] = float(self.adaboot_cv.item())
            metrics["adaboot_inject"] = float(self.inject_gt.item())
        return metrics
