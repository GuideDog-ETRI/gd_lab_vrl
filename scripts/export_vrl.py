"""Export a trained vision-RL checkpoint (stage 1+, terrain-latent-augmented
actor) to TorchScript + ONNX without launching the simulator.

Separate from ``scripts/export.py`` (the blind, 2-input contract) by design
-- see ``gd_lab.deploy.export_vrl``'s module docstring for why the two must
not be mixed. The policy is rebuilt from the agent config and the checkpoint
alone; observation dims come from the DreamWaQ spec, so no environment is
needed (only a booted Kit runtime -- ``isaaclab_rl.rsl_rl`` pulls in ``omni``
transitively even though this script never touches a simulator).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict

from gd_lab.deploy.export_vrl import export_policy_vrl
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC, POLICY_OBS_DIM
from gd_lab.rl.actor_critic_vrl import DreamwaqVrlActorCritic


class _RunConfigLoader(yaml.SafeLoader):
    pass


# IsaacLab's YAML writer emits tuples; permit that data tag without Python object construction.
_RunConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a gd_lab vision-RL DreamWaQ checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to a model_*.pt checkpoint.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: <ckpt dir>/exported).")
    parser.add_argument("--num-actions", type=int, default=12)
    parser.add_argument(
        "--height-scan-dim", type=int, default=187, help="Ray count of the critic height scanner (17x11 grid)."
    )
    parser.add_argument(
        "--agent-config", type=Path, default=None, help="Saved agent YAML (default: <ckpt dir>/params/agent.yaml)."
    )
    args = parser.parse_args()

    critic = DREAMWAQ_SPEC.critic.resolve(height_scan=args.height_scan_dim)
    obs = TensorDict(
        {"policy": torch.zeros(1, POLICY_OBS_DIM), "critic": torch.zeros(1, critic.total)},
        batch_size=[1],
    )
    # Read the saved run config instead of importing the cfg class: that import
    # pulls isaaclab_rl -> isaaclab.envs -> omni.log, which needs a booted Kit.
    # Export is pure torch, so it has no business starting a simulator.
    agent_path = args.agent_config or Path(args.checkpoint).parent / "params" / "agent.yaml"
    if not agent_path.is_file():
        parser.error(f"Saved agent config not found: {agent_path}. Pass --agent-config for a moved checkpoint.")
    with agent_path.open() as stream:
        agent_cfg = yaml.load(stream, Loader=_RunConfigLoader)
    policy_cfg = dict(agent_cfg["policy"])
    policy_cfg.pop("class_name", None)
    # Older smoke runs predate explicit layout injection; their weights are compatible.
    policy_cfg.setdefault("height_scan_start", DREAMWAQ_SPEC.critic.offset("height_scan"))
    policy = DreamwaqVrlActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, args.num_actions, **policy_cfg)

    loaded = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    policy.load_state_dict(loaded["model_state_dict"])

    out_dir = args.out or os.path.join(os.path.dirname(args.checkpoint), "exported")
    policy_term_dims = [t.dim for t in DREAMWAQ_SPEC.policy.terms]
    jit_path, onnx_path = export_policy_vrl(
        policy, out_dir, policy_term_dims, terrain_latent_dim=policy_cfg["terrain_latent_dim"]
    )
    print(f"Exported: {jit_path}\n          {onnx_path}")


if __name__ == "__main__":
    main()
