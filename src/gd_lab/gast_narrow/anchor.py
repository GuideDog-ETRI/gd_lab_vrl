"""GAST-Narrow v2.1: PPO + a retention anchor toward the frozen 6566 policy (design: GAST_NARROW_V21_DESIGN.md).

After every PPO/CENet/terrain update, one extra pass over the rollout pulls the deterministic action of the policy
toward the 6566 action on the samples that are NOT on narrow tiles (gap, stairs, slopes, rough, ring fence):
    anchor loss = w * mean_{non-narrow samples} || mu_policy(o) - mu_6566(o) ||^2
mu = act_inference (posterior-mean CENet code, deterministic terrain/LiDAR latents). The 6566 policy runs as a copy of
the live policy with the 6566 weights widened by zero columns (LiDAR latent + margin) -- exactly 6566's actions.
Narrow tiles (privileged flag narrow_priv[..., 5] = 1) keep plain PPO, so stand / pass keep being learned there.

Env: NARROW_ANCHOR_CKPT (6566 checkpoint), NARROW_ANCHOR_WEIGHT (default 1.0), NARROW_ANCHOR_EPOCHS (default 1).
"""
from __future__ import annotations

import copy
import os

import torch
from torch import nn

from gd_lab.gast.teacher import GastPPO
from gd_lab.gast_narrow.model import PRIV_GROUP

NARROW_FLAG = 5  # narrow_priv index of the "on a narrow tile" flag (narrow_privileged_v2)


def widen_to(state: dict, target: dict) -> dict:
    """6566-shaped state -> the live policy's shapes: zero columns / zero LiDAR encoder (acts exactly as 6566)."""
    out = {}
    for name, value in target.items():
        if name not in state:
            if not name.startswith("lidar_encoder."):
                raise ValueError(f"anchor checkpoint lacks {name}")
            out[name] = torch.zeros_like(value)
            continue
        s = state[name].to(dtype=value.dtype, device=value.device)
        if s.shape != value.shape:
            if s.dim() != 2 or s.shape[0] != value.shape[0] or s.shape[1] > value.shape[1]:
                raise ValueError(f"{name}: anchor {tuple(s.shape)} cannot widen to {tuple(value.shape)}")
            grown = torch.zeros_like(value)
            grown[:, :s.shape[1]] = s
            s = grown
        out[name] = s
    return out


def _detached_copy(module: nn.Module) -> nn.Module:
    """deepcopy of a live policy: graph-attached cached tensors / distributions are set aside during the copy."""
    stash = []
    for m in module.modules():
        for key, value in list(vars(m).items()):
            if (isinstance(value, torch.Tensor) and value.grad_fn is not None) or \
                    isinstance(value, torch.distributions.Distribution):
                stash.append((m, key, value))
                setattr(m, key, None)
    try:
        return copy.deepcopy(module)
    finally:
        for m, key, value in stash:
            setattr(m, key, value)


class NarrowAnchorPPO(GastPPO):
    def _build_anchor(self):
        path = os.environ.get("NARROW_ANCHOR_CKPT")
        if not path:
            raise RuntimeError("NARROW_ANCHOR_CKPT (6566 checkpoint) is required for GAST-Narrow v2.1")
        state = torch.load(path, map_location=self.device, weights_only=False)["model_state_dict"]
        anchor = _detached_copy(self.policy)
        anchor.load_state_dict(widen_to(state, anchor.state_dict()), strict=True)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
        self._anchor = anchor
        self._anchor_weight = float(os.environ.get("NARROW_ANCHOR_WEIGHT", "1.0"))
        self._anchor_epochs = int(os.environ.get("NARROW_ANCHOR_EPOCHS", "1"))
        print(f"[NARROW_ANCHOR] 6566 anchor from {path}: weight={self._anchor_weight} epochs={self._anchor_epochs}",
              flush=True)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict.update(self._anchor_update())
        return loss_dict

    def _anchor_update(self) -> dict[str, float]:
        if getattr(self, "_anchor", None) is None:
            self._build_anchor()
        obs_all = self.storage.observations.flatten(0, 1)
        retention = obs_all[PRIV_GROUP][..., NARROW_FLAG] < 0.5
        n = retention.shape[0]
        mini = n // self.num_mini_batches
        total, frac, steps = 0.0, float(retention.float().mean()), 0
        for _ in range(self._anchor_epochs):
            order = torch.randperm(self.num_mini_batches * mini, device=self.device)
            for i in range(self.num_mini_batches):
                idx = order[i * mini:(i + 1) * mini]
                obs, mask = obs_all[idx], retention[idx]
                with torch.no_grad():
                    target = self._anchor.act_inference(obs)
                mu = self.policy.act_inference(obs)
                err = (mu - target).square().sum(-1)
                # Every rank runs backward + all-reduce, also with no retention sample (no collective mismatch).
                count = mask.sum()
                loss = self._anchor_weight * (err * mask).sum() / count.clamp_min(1)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total += float(loss.detach()) / max(self._anchor_weight, 1e-9)
                steps += 1
        return {"anchor_mse": total / max(steps, 1), "anchor_retention_fraction": frac}
