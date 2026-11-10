"""Runner / PPO configuration for the handstand skill, plus the four-leg stand anchor.

The policy layout is the blind DreamWaQ layout with one extra per-step term (``skill_command``, 3).
Mirror augmentation is off: the DreamWaQ mirror permutations are bound to the blind spec.
"""
from __future__ import annotations

import copy
import os

import torch
from isaaclab.utils import configclass

from gd_lab.agents.dreamwaq_ppo_cfg import DreamwaqActorCriticCfg, DreamwaqAlgorithmCfg, DreamwaqRunnerCfg
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC
from gd_lab.rl.ppo import DreamwaqPPO

SKILL_CMD_DIM = 3
BLIND_STEP = DREAMWAQ_SPEC.policy.per_step            # 46
POLICY_TERM_DIMS = [t.dim for t in DREAMWAQ_SPEC.policy.terms] + [SKILL_CMD_DIM]
GRAVITY = slice(3, 6)                                  # projected_gravity in one step
REQUEST, ABORT = BLIND_STEP, BLIND_STEP + 1            # skill_command in one step


class StandAnchorPPO(DreamwaqPPO):
    """DreamWaQ PPO + a regression of the live mean action toward the frozen start policy on idle stand samples.

    Idle = no request, no abort, trunk level (|g_b . z| > 0.95). Keeps the four-leg stand (and so the hand-over to
    the walking policy) while the skill is learned. The target is a frozen copy taken at the first update, i.e. the
    widened DWB-38000 (new columns zero, so identical to DWB on these samples).
    """

    anchor_weight = float(os.environ.get("HANDSTAND_ANCHOR_WEIGHT", "1.0"))
    # S2b: also anchor requested-but-not-walking samples (lift-up and hold) to the frozen start policy (S1 model_400)
    hold_anchor_weight = float(os.environ.get("HANDSTAND_HOLD_ANCHOR_WEIGHT", "0.0"))
    anchor_max_samples = int(os.environ.get("HANDSTAND_ANCHOR_SAMPLES", "65536"))

    def update(self) -> dict[str, float]:
        if not hasattr(self, "_anchor_target"):
            self._anchor_target = copy.deepcopy(self.policy).eval()
            for p in self._anchor_target.parameters():
                p.requires_grad_(False)
        out = super().update()
        if self.anchor_weight > 0:
            out.update(self._anchor_pass())
        return out

    def _anchor_pass(self) -> dict[str, float]:
        st = self.storage
        group = self.policy.obs_groups["policy"][0]
        obs = st.observations.flatten(0, 1)
        latest = obs[group][..., self.policy.latest_idx]
        idle = (latest[:, REQUEST] < 0.5) & (latest[:, ABORT] < 0.5) & (latest[:, GRAVITY][:, 2] < -0.95)
        weights = torch.full_like(latest[:, 0], self.anchor_weight)
        if self.hold_anchor_weight > 0 and self.policy.one_step_obs_dim > ABORT + 2:
            hold = (latest[:, REQUEST] > 0.5) & (latest[:, ABORT + 2] < 0.5)        # walk flag is right after skill_command
            weights = torch.where(hold, torch.full_like(weights, self.hold_anchor_weight), weights)
            idle = idle | hold
        idx = idle.nonzero().squeeze(1)
        if idx.numel() == 0:
            return {"anchor_mse": 0.0, "anchor_fraction": 0.0}
        if idx.numel() > self.anchor_max_samples:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.anchor_max_samples]]
        batch = obs[idx]
        with torch.no_grad():
            target = self._anchor_target.act_inference(batch)
        pred = self.policy.actor(self.policy._actor_input(batch, inference=True))
        per = (pred - target).square().mean(dim=1)
        loss = per.mean()
        self.optimizer.zero_grad()
        (weights[idx] * per).mean().backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.reduce_parameters()
        self.optimizer.step()
        return {"anchor_mse": float(loss.detach()), "anchor_fraction": float(idle.float().mean())}


@configclass
class HandstandActorCriticCfg(DreamwaqActorCriticCfg):
    policy_term_dims: list[int] = POLICY_TERM_DIMS


@configclass
class HandstandAlgorithmCfg(DreamwaqAlgorithmCfg):
    class_name: str = "skill_handstand.agent:StandAnchorPPO"


@configclass
class HandstandRunnerCfg(DreamwaqRunnerCfg):
    max_iterations = 500
    save_interval = 50
    experiment_name = "skill_handstand"
    logger = "tensorboard"
    policy = HandstandActorCriticCfg(
        init_noise_std=1.0, noise_std_type="scalar", actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu")
    algorithm = HandstandAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5,
        num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01,
        max_grad_norm=1.0, symmetry_cfg=None)


@configclass
class HandstandS2ActorCriticCfg(DreamwaqActorCriticCfg):
    policy_term_dims: list[int] = POLICY_TERM_DIMS + [1]   # + skill_walk


@configclass
class HandstandS2RunnerCfg(HandstandRunnerCfg):
    policy = HandstandS2ActorCriticCfg(
        init_noise_std=1.0, noise_std_type="scalar", actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu")
