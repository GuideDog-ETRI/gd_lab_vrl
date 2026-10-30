"""GAST-Narrow v2.1: PPO + a retention anchor toward the frozen 6566 policy (design: GAST_NARROW_V21_DESIGN.md).

After every PPO/CENet/terrain update, one extra pass over the rollout pulls the deterministic action of the policy
toward the 6566 action on the samples that are NOT on narrow tiles (gap, stairs, slopes, rough, ring fence):
    anchor loss = w * mean_{non-narrow samples} || mu_policy(o) - mu_6566(o) ||^2
mu = act_inference (posterior-mean CENet code, deterministic terrain/LiDAR latents). The 6566 policy runs as a copy of
the live policy with the 6566 weights widened by zero columns (LiDAR latent + margin) -- exactly 6566's actions.
Narrow tiles (privileged flag narrow_priv[..., 5] = 1) keep plain PPO, so stand / pass keep being learned there.

Env: NARROW_ANCHOR_CKPT (6566 checkpoint), NARROW_ANCHOR_WEIGHT (default 1.0), NARROW_ANCHOR_EPOCHS (default 1).
v2.3: NARROW_ANCHOR_GAP_NOSE=1 also drops the samples the gap_nose_up reward flags (gap tile AND nose-up above its
limit; tasks_v23 hands the mask over with set_gap_nose_mask each step). 6566 lifts its front 20-30 deg crossing gaps
forward; gap_nose_up corrects those moments and the anchor must not undo them. A pitch-only rule (no tile check) also
dropped stair / slope climbing (5 % of all samples in the 30-update test) and so weakened the stair anchor.
v2.4c stand anchor (NARROW_STAND_ANCHOR_WEIGHT, default 1.0; 0 = off; needs the narrow_turn_block group): on the
samples whose turn-block flag is 1 the policy's deterministic action is pulled toward the action of a FROZEN copy of
the start policy (NARROW_STAND_ANCHOR_CKPT, else NARROW_SOURCE = 602, else the 6566 anchor) on the same observation
with the velocity command (all history frames) and the flag zeroed:
    stand loss = w_s * mean_{flag samples} || mu(o) - mu_frozen(o | command = 0, flag = 0) ||^2
i.e. "flag on = behave as the start policy does under a zero command" (it stands). The first version used the live
policy itself as the target: a moving, self-referential target whose gradient also reshaped the shared command
columns -- the run diverged at update 13 (stand mse 0.56 -> 2e4, actions unbounded). A frozen target is a plain
regression like the 6566 anchor. Reward shaping alone did not
produce it: the stop-tile episodes are ~1 % of the samples and end within a second (wall contact / drift), and the
policy prior turns on every wz command, so standing was never sampled (v2.4b run 0 % stop in 226 updates, v2.4c
run 1 0 % in 104). The anchor gives the direct gradient the rollouts cannot.
"""
from __future__ import annotations

import copy
import os

import torch
from torch import nn

from gd_lab.gast.teacher import GastPPO
from gd_lab.gast_narrow.model import PRIV_GROUP, TURN_BLOCK_GROUP
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC
from gd_lab.rl.distributed import flush_normalizer

NARROW_FLAG = 5  # narrow_priv index of the "on a narrow tile" flag (narrow_privileged_v2)
_gap_nose_mask = None  # [num_envs] bool, this env step (set by tasks_v23.gap_nose_up)


def set_gap_nose_mask(mask: torch.Tensor) -> None:
    global _gap_nose_mask
    _gap_nose_mask = mask


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
        self._stand_weight = float(os.environ.get("NARROW_STAND_ANCHOR_WEIGHT", "1.0"))
        self._stand_anchor, stand_path = None, None
        if self._stand_weight > 0:
            stand_path = os.environ.get("NARROW_STAND_ANCHOR_CKPT") or os.environ.get("NARROW_SOURCE") or path
            sstate = torch.load(stand_path, map_location=self.device, weights_only=False)["model_state_dict"]
            stand_anchor = _detached_copy(self.policy)
            stand_anchor.load_state_dict(widen_to(sstate, stand_anchor.state_dict()), strict=True)
            stand_anchor.eval()
            for p in stand_anchor.parameters():
                p.requires_grad_(False)
            self._stand_anchor = stand_anchor
        print(f"[NARROW_ANCHOR] 6566 anchor from {path}: weight={self._anchor_weight} epochs={self._anchor_epochs} "
              f"gap_nose={self._gap_nose_enabled()} stand_weight={self._stand_weight} stand_target={stand_path}",
              flush=True)

    def _stand_target_obs(self, obs):
        """The same observations with the velocity command (every history frame) and the turn-block flag zeroed."""
        key = self.policy.obs_groups["policy"][0]
        proprio = obs[key]
        if proprio.shape[-1] != DREAMWAQ_SPEC.policy.total:
            raise RuntimeError(f"stand anchor: policy group {key} has {proprio.shape[-1]} inputs, "
                               f"spec {DREAMWAQ_SPEC.policy.total}")
        out = obs.clone()
        proprio = proprio.clone()
        proprio[..., DREAMWAQ_SPEC.policy.slice("velocity_commands")] = 0.0
        out[key] = proprio
        out[TURN_BLOCK_GROUP] = torch.zeros_like(obs[TURN_BLOCK_GROUP])
        return out

    @staticmethod
    def _gap_nose_enabled() -> bool:
        return os.environ.get("NARROW_ANCHOR_GAP_NOSE", "0") == "1"

    def process_env_step(self, obs, rewards, dones, extras):
        # NARROW_HEARTBEAT=k: rank 0 prints a timestamp every k rollout steps (first rollouts of large runs are silent).
        beat = int(os.environ.get("NARROW_HEARTBEAT", "0"))
        everyone = os.environ.get("NARROW_HEARTBEAT_ALL", "0") == "1"
        if beat and self.storage.step % beat == 0 and (everyone or int(os.environ.get("RANK", "0")) == 0):
            import socket
            import time
            print(f"[HEARTBEAT] {time.strftime('%H:%M:%S')} rank {os.environ.get('RANK', '0')} "
                  f"{socket.gethostname().split('.')[0]} rollout step {self.storage.step}", flush=True)
        # Before super(): storage.step is still the index of the transition this step completes.
        if self._gap_nose_enabled():
            if _gap_nose_mask is None:
                raise RuntimeError("NARROW_ANCHOR_GAP_NOSE=1 but no gap_nose_up mask (task without the term?)")
            buf = getattr(self, "_gap_nose_buf", None)
            if buf is None:
                buf = self._gap_nose_buf = torch.zeros(self.storage.num_transitions_per_env, _gap_nose_mask.shape[0],
                                                       dtype=torch.bool, device=self.device)
            buf[self.storage.step] = _gap_nose_mask.to(self.device)
        return super().process_env_step(obs, rewards, dones, extras)

    @staticmethod
    def _phase(name: str) -> None:
        """NARROW_HEARTBEAT_ALL=1: every rank stamps the update phases (finding the rank others wait for)."""
        if os.environ.get("NARROW_HEARTBEAT_ALL", "0") == "1":
            import socket
            import time
            print(f"[PHASE] {time.strftime('%H:%M:%S')} rank {os.environ.get('RANK', '0')} "
                  f"{socket.gethostname().split('.')[0]} {name}", flush=True)

    def update(self) -> dict[str, float]:
        self._phase("update-start")
        # GD_LAB_NORMALIZER_DEFER=1: the rollout's normalizer moments are reduced once here (same ranks, same order).
        for name in ("actor_obs_normalizer", "critic_obs_normalizer"):
            normalizer = getattr(self.policy, name, None)
            if normalizer is not None and hasattr(normalizer, "_deferred_moments"):
                flush_normalizer(normalizer)
        self._phase("ppo-start")
        loss_dict = super().update()
        self._phase("ppo-done")
        loss_dict.update(self._anchor_update())
        self._phase("anchor-done")
        return loss_dict

    def _anchor_update(self) -> dict[str, float]:
        if getattr(self, "_anchor", None) is None:
            self._build_anchor()
        obs_all = self.storage.observations.flatten(0, 1)
        retention = obs_all[PRIV_GROUP][..., NARROW_FLAG] < 0.5
        nose_frac = 0.0
        if self._gap_nose_enabled():
            nose_up = self._gap_nose_buf.flatten(0, 1)  # [T, N] -> T-major, as storage.observations.flatten(0, 1)
            nose_frac = float((nose_up & retention).float().mean())
            retention = retention & ~nose_up
        stand_on = self._stand_weight > 0 and TURN_BLOCK_GROUP in obs_all.keys()
        stand = obs_all[TURN_BLOCK_GROUP][..., 0] > 0.5 if stand_on else None
        n = retention.shape[0]
        mini = n // self.num_mini_batches
        total, frac, steps = 0.0, float(retention.float().mean()), 0
        stand_total, stand_frac = 0.0, float(stand.float().mean()) if stand_on else 0.0
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
                if stand_on:
                    smask = stand[idx]
                    with torch.no_grad():
                        stand_target = self._stand_anchor.act_inference(self._stand_target_obs(obs))
                    stand_err = (mu - stand_target).square().sum(-1)
                    stand_loss = self._stand_weight * (stand_err * smask).sum() / smask.sum().clamp_min(1)
                    loss = loss + stand_loss
                    stand_total += float(stand_loss.detach()) / self._stand_weight
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total += float(loss.detach()) / max(self._anchor_weight, 1e-9)
                steps += 1
        return {"anchor_mse": total / max(steps, 1), "anchor_retention_fraction": frac,
                "anchor_nose_excluded_fraction": nose_frac,
                "stand_anchor_mse": stand_total / max(steps, 1), "stand_anchor_fraction": stand_frac}
