"""Export a trained checkpoint to TorchScript + ONNX without launching the simulator.

The policy is rebuilt from the saved run's agent config and checkpoint; observation
dims come from the DreamWaQ spec, so no simulator imports are needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict

from gd_lab.deploy.export import export_policy
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC, POLICY_OBS_DIM
from gd_lab.rl.actor_critic import DreamwaqActorCritic


class _RunConfigLoader(yaml.SafeLoader):
    pass


# IsaacLab's YAML writer emits tuples; permit that data tag without Python object construction.
_RunConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a gd_lab DreamWaQ checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to a model_*.pt checkpoint.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: <ckpt dir>/exported).")
    parser.add_argument(
        "--agent-config", type=Path, default=None, help="Saved agent YAML (default: <ckpt dir>/params/agent.yaml)."
    )
    parser.add_argument("--num-actions", type=int, default=12)
    parser.add_argument(
        "--height-scan-dim", type=int, default=187, help="Ray count of the critic height scanner (17x11 grid)."
    )
    parser.add_argument(
        "--no-metadata", action="store_true", help="Export a bare graph without the deployment contract."
    )
    args = parser.parse_args()

    critic = DREAMWAQ_SPEC.critic.resolve(height_scan=args.height_scan_dim)
    obs = TensorDict(
        {"policy": torch.zeros(1, POLICY_OBS_DIM), "critic": torch.zeros(1, critic.total)},
        batch_size=[1],
    )
    agent_path = args.agent_config or Path(args.checkpoint).parent / "params" / "agent.yaml"
    if not agent_path.is_file():
        parser.error(f"Saved agent config not found: {agent_path}. Pass --agent-config for a moved checkpoint.")
    with agent_path.open() as stream:
        agent_cfg = yaml.load(stream, Loader=_RunConfigLoader)
    policy_cfg = dict(agent_cfg["policy"])
    policy_cfg.pop("class_name", None)
    policy = DreamwaqActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, args.num_actions, **policy_cfg)

    loaded = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    policy.load_state_dict(loaded["model_state_dict"])

    context = (loaded.get("infos") or {}).get("gd_lab", {}).get("deploy_context")
    if context is None and not args.no_metadata:
        raise SystemExit(
            "Checkpoint carries no deployment context; it predates metadata capture or the "
            "capture failed at training time. Re-export with --no-metadata to accept a bare graph."
        )

    out_dir = args.out or os.path.join(os.path.dirname(args.checkpoint), "exported")
    jit_path, onnx_path = export_policy(policy, out_dir, deploy_context=None if args.no_metadata else context)
    print(f"Exported: {jit_path}\n          {onnx_path}")
    if context is not None and not args.no_metadata:
        print(f"          {os.path.join(out_dir, 'deploy.json')}")


if __name__ == "__main__":
    main()
