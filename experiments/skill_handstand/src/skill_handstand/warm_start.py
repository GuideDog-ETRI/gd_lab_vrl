"""Load DWB-38000 into the handstand policy: same weights, zero columns / rows for the new inputs.

Policy history is term-major (each term's 5 frames together), so the appended ``skill_command`` term occupies the
last 15 history positions; the actor sees 4 frames newest-first, each frame = 46 blind values + 3 new values.
At iteration 0 the policy therefore acts exactly like DWB-38000 whatever the command is.
"""
from __future__ import annotations

import hashlib

import torch

from .agent import BLIND_STEP, SKILL_CMD_DIM



def _pad_cols(old: torch.Tensor, new_shape, mapping) -> torch.Tensor:
    out = torch.zeros(new_shape, dtype=old.dtype)
    for dst, src in mapping:
        out[:, dst] = old[:, src]
    return out


def widen_state(old: dict, new: dict, *, history: int, actor_frames: int, code_dim: int, critic_extra: int,
                min_std: float, old_step: int = BLIND_STEP, new_step: int = BLIND_STEP + SKILL_CMD_DIM) -> dict:
    """Appended per-step terms (term-major history): new columns / rows start at zero."""
    step_new = new_step
    BLIND = old_step
    out = {}
    for key, ref in new.items():
        src = old.get(key)
        if src is None:
            raise KeyError(f"DWB checkpoint has no {key}")
        if src.shape == ref.shape:
            out[key] = src.clone()
            continue
        if key == "actor.0.weight":
            m = [(slice(k * step_new, k * step_new + BLIND), slice(k * BLIND, (k + 1) * BLIND))
                 for k in range(actor_frames)]
            m.append((slice(actor_frames * step_new, actor_frames * step_new + code_dim),
                      slice(actor_frames * BLIND, actor_frames * BLIND + code_dim)))
            out[key] = _pad_cols(src, ref.shape, m)
        elif key == "cenet.encoder.0.weight":
            out[key] = _pad_cols(src, ref.shape, [(slice(0, BLIND * history), slice(0, BLIND * history))])
        elif key in ("cenet.decoder.4.weight", "cenet.decoder.4.bias"):
            t = torch.zeros(ref.shape, dtype=src.dtype)
            t[:BLIND] = src
            out[key] = t
        elif key == "critic.0.weight":
            n = src.shape[1]
            assert ref.shape[1] == n + critic_extra, (key, src.shape, ref.shape)
            out[key] = _pad_cols(src, ref.shape, [(slice(0, n), slice(0, n))])
        elif key.endswith(("_mean", "_var", "_std")) and "normalizer" in key:
            fill = 0.0 if key.endswith("_mean") else 1.0
            t = torch.full(ref.shape, fill, dtype=src.dtype)
            t[:, : src.shape[1]] = src
            out[key] = t
        else:
            raise ValueError(f"unexpected shape change for {key}: {tuple(src.shape)} -> {tuple(ref.shape)}")
    out["std"] = out["std"].clamp(min=min_std)
    return out


def warm_start_from_dwb(policy, path: str, *, min_std: float, critic_extra: int) -> dict:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    new = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    state = widen_state(ck["model_state_dict"], new, history=policy.history_length,
                        actor_frames=policy.actor_history_steps, code_dim=policy.cenet.code_dim,
                        critic_extra=critic_extra, min_std=min_std)
    policy.load_state_dict({k: v.to(next(policy.parameters()).device) for k, v in state.items()}, strict=True)
    return dict(source=path, source_sha256=digest, source_iteration=int(ck.get("iter", -1)),
                min_std=min_std, new_policy_columns=SKILL_CMD_DIM, new_critic_columns=critic_extra)


def widen_from_handstand(policy, path: str, *, old_step: int, critic_extra: int) -> dict:
    """S1 -> S2: the S1 checkpoint plus zero columns for the appended per-step term(s); std kept."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    new = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    state = widen_state(ck["model_state_dict"], new, history=policy.history_length,
                        actor_frames=policy.actor_history_steps, code_dim=policy.cenet.code_dim,
                        critic_extra=critic_extra, min_std=0.0, old_step=old_step, new_step=policy.one_step_obs_dim)
    policy.load_state_dict({k: v.to(next(policy.parameters()).device) for k, v in state.items()}, strict=True)
    return dict(source=path, source_iteration=int(ck.get("iter", -1)), kind="handstand_widened",
                old_step=old_step, new_step=policy.one_step_obs_dim, new_critic_columns=critic_extra)
