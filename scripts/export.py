"""Export a trained checkpoint to TorchScript + ONNX without launching the simulator.

The policy is rebuilt from the agent config and the checkpoint alone; observation
dims come from the DreamWaQ spec, so no environment is needed.
"""

from __future__ import annotations

import argparse
import os

import torch
from tensordict import TensorDict

from gd_lab.agents.dreamwaq_ppo_cfg import DreamwaqRunnerCfg
from gd_lab.deploy.export import export_policy
from gd_lab.methods.dreamwaq.spec import DREAMWAQ_SPEC, POLICY_OBS_DIM
from gd_lab.rl.actor_critic import DreamwaqActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a gd_lab DreamWaQ checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to a model_*.pt checkpoint.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: <ckpt dir>/exported).")
    parser.add_argument("--num-actions", type=int, default=12)
    parser.add_argument(
        "--height-scan-dim", type=int, default=187, help="Ray count of the critic height scanner (17x11 grid)."
    )
    args = parser.parse_args()

    critic = DREAMWAQ_SPEC.critic.resolve(height_scan=args.height_scan_dim)
    obs = TensorDict(
        {"policy": torch.zeros(1, POLICY_OBS_DIM), "critic": torch.zeros(1, critic.total)},
        batch_size=[1],
    )
    policy_cfg = DreamwaqRunnerCfg().policy.to_dict()
    policy_cfg.pop("class_name")
    policy = DreamwaqActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, args.num_actions, **policy_cfg)

    loaded = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    policy.load_state_dict(loaded["model_state_dict"])

    out_dir = args.out or os.path.join(os.path.dirname(args.checkpoint), "exported")
    jit_path, onnx_path = export_policy(policy, out_dir)
    print(f"Exported: {jit_path}\n          {onnx_path}")


if __name__ == "__main__":
    main()
