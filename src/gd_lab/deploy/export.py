"""Deploy export for the DreamWaQ policy: one raw-observation input, JIT + ONNX.

The exported module reuses the training modules (normalizer, CENet, actor) so
its output is bit-identical to ``act_inference``; nothing is re-implemented.
Input is the raw flattened proprio history exactly as the observation manager
emits it; outputs are the joint position actions and the estimated base velocity
(in observation scale).
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from gd_lab.rl.actor_critic import DreamwaqActorCritic


class DreamwaqDeployPolicy(nn.Module):
    def __init__(self, policy: DreamwaqActorCritic):
        super().__init__()
        policy = copy.deepcopy(policy).to("cpu")
        policy.eval()
        self.normalizer = policy.actor_obs_normalizer
        self.cenet = policy.cenet
        self.actor = policy.actor
        self.register_buffer("latest_idx", policy.latest_idx.clone(), persistent=False)

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = self.normalizer(obs_history)
        latest = norm[..., self.latest_idx]
        out = self.cenet(norm, deterministic=True)
        code = self.cenet.get_code(out)
        actions = self.actor(torch.cat((latest, code), dim=-1))
        return actions, out.velocity


def export_policy(policy: DreamwaqActorCritic, out_dir: str, stem: str = "policy") -> tuple[str, str]:
    """Write ``<stem>.pt`` (TorchScript) and ``<stem>.onnx`` (opset 18); return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    module = DreamwaqDeployPolicy(policy)
    example = torch.zeros(1, policy.cenet.input_dim)

    jit_path = os.path.join(out_dir, f"{stem}.pt")
    traced = torch.jit.trace(module, example)
    traced.save(jit_path)

    onnx_path = os.path.join(out_dir, f"{stem}.onnx")
    torch.onnx.export(
        module,
        example,
        onnx_path,
        opset_version=18,
        input_names=["obs_history"],
        output_names=["actions", "velocity_est"],
        dynamic_axes={"obs_history": {0: "batch"}, "actions": {0: "batch"}, "velocity_est": {0: "batch"}},
    )
    return jit_path, onnx_path
