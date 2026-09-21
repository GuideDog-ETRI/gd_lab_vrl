"""Export a trained vision-RL stage-3 student checkpoint (perception_*.pt) to
TorchScript + ONNX, alongside an already-exported teacher/actor ONNX.

No simulator/Kit needed -- ``gd_lab.rl.perception`` has zero isaaclab
dependency, unlike ``scripts/export_vrl.py`` (which needs a booted Kit
runtime because ``gd_lab.agents.dreamwaq_ppo_cfg`` pulls in ``isaaclab_rl``).
"""

from __future__ import annotations

import argparse

import torch

from gd_lab.deploy.export_student_vrl import export_student_vrl
from gd_lab.rl.perception import CameraPerceptionEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a gd_lab vision-RL stage-3 student checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to a perception_*.pt (stage-3) checkpoint.")
    parser.add_argument(
        "--actor-onnx",
        type=str,
        required=True,
        help="Path to the already-exported teacher/actor ONNX (from export_vrl.py) -- "
        "the student is written as a sibling file next to it.",
    )
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--camera-width", type=int, default=80)
    parser.add_argument("--camera-height", type=int, default=45)  # 80x45 = D430 16:9, vrl_rough._belly_camera 와 일치
    args = parser.parse_args()

    student = CameraPerceptionEncoder(
        num_cameras=args.num_cameras,
        gru_hidden_dim=args.gru_hidden_dim,
        latent_dim=args.latent_dim,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    student.load_state_dict(ckpt["model"])
    print(f"[INFO] Loaded student checkpoint (iteration {ckpt.get('iteration')})")

    jit_path, onnx_path = export_student_vrl(
        student, args.actor_onnx, camera_width=args.camera_width, camera_height=args.camera_height
    )
    print(f"Exported: {jit_path}\n          {onnx_path}")


if __name__ == "__main__":
    main()
