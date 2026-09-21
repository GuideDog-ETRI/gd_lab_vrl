"""Deploy export for the vision-RL DreamWaQ policy (stage 1+): three
raw-observation inputs, JIT + ONNX.

Kept as a SEPARATE module from ``export.py`` (the existing blind 2-input
contract) rather than editing that file in place: gd_rbq10_deploy's
PolicyBackend.cpp only ever speaks ``export.py``'s 2-input contract today, and
the blind deploy artifacts/pipeline must stay untouched by vision-RL work. A
vision-RL checkpoint's actor is architecturally different (235-dim input, not
203 with the current K=4 -- see ``DreamwaqVrlActorCritic``) and
needs its own contract; feeding one through ``export.py`` would silently
build an actor Linear layer shape mismatch (crash) or, worse if it happened
to not crash, silently drop the terrain latent.

Same reuse-not-reimplement approach as ``export.py``: wraps the trained
modules (normalizer, CENet, terrain_encoder-fed actor) so output is
bit-identical to ``act_inference``. The one contract addition is a third
input, ``terrain_latent`` (``terrain_latent_dim``-wide, 32 by default): during
training this comes from ``terrain_encoder(height_scan)`` -- a
simulation-only privileged signal that does not exist on hardware/Mujoco. At
deploy time gd_rbq10_deploy must supply the stage-3 camera student's own
output here instead. This module deliberately does NOT compute that itself
-- the student is a separate, lower-rate async model (APT-RL's own
deployment split; see ``gd_lab.rl.perception.CameraPerceptionEncoder``), so
folding it into this same traced graph would force it to run at the actor's
full control rate for no benefit. Pilot integration is maintained separately
in the vision deployment tree.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic


def _time_major_to_term_major_index(term_dims: list[int], history: int) -> torch.Tensor:
    """Flat gather index: ``term_major = time_major[..., idx]``.

    time-major (what PolicyBackend.cpp sends on the legacy 3-input path):
    step_0(all terms) ++ step_1(all terms) ++ ... , each step in per-step term
    order. term-major (what ObsSpec/CENet/the normalizer were trained on):
    term_0(all H steps, oldest->newest) ++ term_1(all H steps) ++ ...

    Lives here rather than in ``deploy/export.py``: the contract-based blind
    exporter no longer needs it, because ``camel.policy.v1`` declares
    ``layout: term_major`` and PolicyRuntime packs the observation that way.
    Only this legacy vision graph still does the reorder itself.
    """
    idx: list[int] = []
    step_offset = [0]
    for dim in term_dims:
        step_offset.append(step_offset[-1] + dim)
    one_step = step_offset[-1]
    for dim, term_start in zip(term_dims, step_offset[:-1], strict=True):
        for t in range(history):
            time_major_start = t * one_step + term_start
            idx.extend(range(time_major_start, time_major_start + dim))
    return torch.tensor(idx, dtype=torch.long)


class DreamwaqVrlDeployPolicy(nn.Module):
    def __init__(self, policy: DreamwaqVrlActorCritic, policy_term_dims: list[int]):
        super().__init__()
        policy = copy.deepcopy(policy).to("cpu")
        policy.eval()
        self.normalizer = policy.actor_obs_normalizer
        self.cenet = policy.cenet
        self.actor = policy.actor
        self.register_buffer("latest_idx", policy.latest_idx.clone(), persistent=False)
        # The actor now consumes ``actor_history_steps`` frames, newest first, not
        # just the newest one. ``direct_obs`` still carries the newest frame (so the
        # three-input deploy signature and DreamVrlBackend stay untouched); the
        # remaining K-1 come out of the history that ``cenet_obs`` already carries.
        self.register_buffer(
            "older_frames_idx", policy.frames_idx[policy.one_step_obs_dim :].clone(), persistent=False
        )
        reorder = _time_major_to_term_major_index(policy_term_dims, policy.history_length)
        self.register_buffer("time_major_to_term_major_idx", reorder, persistent=False)

    def forward(
        self, direct_obs: torch.Tensor, cenet_obs: torch.Tensor, terrain_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Same normalization/reorder as export.py's DreamwaqDeployPolicy -- see
        # that module's docstring for why. terrain_latent arrives pre-computed
        # (by the stage-3 student on the deploy side) and is used as-is: the
        # terrain_encoder that produces it during training already ends in a
        # tanh, and PPO trained the actor against exactly that raw range, so
        # no further normalization belongs here.
        mean = self.normalizer._mean[0, self.latest_idx]
        std = self.normalizer._std[0, self.latest_idx]
        latest = (direct_obs - mean) / (std + self.normalizer.eps)

        cenet_obs = cenet_obs[..., self.time_major_to_term_major_idx]
        norm_history = self.normalizer(cenet_obs)
        out = self.cenet(norm_history, deterministic=True)
        code = self.cenet.get_code(out)
        frames = torch.cat((latest, norm_history[..., self.older_frames_idx]), dim=-1)
        actions = self.actor(torch.cat((frames, code, terrain_latent), dim=-1))
        return actions, code


def export_policy_vrl(
    policy: DreamwaqVrlActorCritic,
    out_dir: str,
    policy_term_dims: list[int],
    terrain_latent_dim: int,
    stem: str = "policy_vrl",
) -> tuple[str, str]:
    """Write ``<stem>.pt`` (TorchScript) and ``<stem>.onnx`` (opset 18); return their paths.

    Three inputs: ``direct_obs``, ``cenet_obs`` (identical meaning to
    ``export.py``'s) plus ``terrain_latent`` (deploy-supplied, see module
    docstring). ``stem`` defaults to ``"policy_vrl"``, not ``"policy"``, so a
    vision-RL export can never silently collide with a blind one in the same
    output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    module = DreamwaqVrlDeployPolicy(policy, policy_term_dims)
    one_step_dim = module.latest_idx.numel()
    example = (
        torch.zeros(1, one_step_dim),
        torch.zeros(1, policy.cenet.input_dim),
        torch.zeros(1, terrain_latent_dim),
    )

    jit_path = os.path.join(out_dir, f"{stem}.pt")
    traced = torch.jit.trace(module, example)
    traced.save(jit_path)

    onnx_path = os.path.join(out_dir, f"{stem}.onnx")
    torch.onnx.export(
        module,
        example,
        onnx_path,
        opset_version=18,
        input_names=["direct_obs", "cenet_obs", "terrain_latent"],
        output_names=["actions", "z_t"],
        dynamic_axes={
            "direct_obs": {0: "batch"},
            "cenet_obs": {0: "batch"},
            "terrain_latent": {0: "batch"},
            "actions": {0: "batch"},
            "z_t": {0: "batch"},
        },
    )
    return jit_path, onnx_path
