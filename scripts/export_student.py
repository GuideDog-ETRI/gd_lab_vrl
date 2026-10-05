"""Export a trained vision-RL stage-3 student checkpoint (perception_*.pt) to
TorchScript + ONNX, alongside an already-exported teacher/actor ONNX.

No simulator/Kit needed -- ``gd_lab.students.rvld.model`` has zero isaaclab
dependency, unlike ``scripts/export_vrl.py`` (which needs a booted Kit
runtime because ``gd_lab.agents.dreamwaq_ppo_cfg`` pulls in ``isaaclab_rl``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gd_lab.core.camera_contract import check_checkpoint_camera_contract, load_camera_contract, same_camera_contract
from gd_lab.deploy.export_student_vrl import export_student_vrl
from gd_lab.students.rvld.model import CameraPerceptionEncoder
from gd_lab.students.gavd.model import GridAttentionStudent


def validate_export_checkpoint(ckpt, camera_width, camera_height, latent_dim, environ=None):
    """Refuse to export a student whose camera contract is missing, unknown or inconsistent with the model.

    Returns the recorded contract (None only when an unverified legacy file was explicitly allowed).
    """
    recorded = check_checkpoint_camera_contract(ckpt.get("camera_contract"), purpose="export", environ=environ)
    architecture = ckpt.get("student_arch", "cnn_gru")
    if recorded is not None:
        profile = recorded.get("profile")
        try:
            known = load_camera_contract(profile)
        except ValueError as exc:
            raise ValueError(f"checkpoint camera contract names an unknown profile {profile!r}") from exc
        if not same_camera_contract(recorded, known, timing=False):
            raise ValueError(f"checkpoint camera contract is not the {profile!r} calibration it claims to be")
        if (recorded.get("width"), recorded.get("height")) != (camera_width, camera_height):
            raise ValueError(f"export image size {(camera_width, camera_height)} != recorded "
                             f"{(recorded.get('width'), recorded.get('height'))}")
        if architecture == "grid_attention_v1" and (ckpt.get("student_config") or {}).get("camera_profile") != profile:
            raise ValueError("student_config camera_profile differs from the recorded camera contract")
    if architecture == "cnn_gru":
        head = ckpt["model"].get("head.0.weight")
        if head is not None and head.shape[0] != latent_dim:
            raise ValueError(f"checkpoint latent size {head.shape[0]} != --latent-dim {latent_dim}")
    return recorded


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

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    recorded = validate_export_checkpoint(ckpt, args.camera_width, args.camera_height, args.latent_dim)
    student = CameraPerceptionEncoder(
        num_cameras=args.num_cameras,
        gru_hidden_dim=args.gru_hidden_dim,
        latent_dim=args.latent_dim,
    )
    architecture = ckpt.get("student_arch", "cnn_gru")
    if architecture == "grid_attention_v1":
        student = GridAttentionStudent(**ckpt["student_config"])
    elif architecture != "cnn_gru":
        raise ValueError(f"Unsupported student architecture: {architecture}")
    student.load_state_dict(ckpt["model"])
    print(f"[INFO] Loaded student checkpoint (iteration {ckpt.get('iteration')})")

    jit_path, onnx_path = export_student_vrl(
        student, args.actor_onnx, camera_width=args.camera_width, camera_height=args.camera_height,
        camera_profile=recorded.get("profile") if recorded is not None else None,
    )
    print(f"Exported: {jit_path}\n          {onnx_path}")
    Path(onnx_path + ".json").write_text(json.dumps({
        "student_arch": architecture, "age_input": ckpt.get("age_input"),
        "camera_contract": ckpt.get("camera_contract"),
        "teacher_checkpoint": ckpt.get("teacher_checkpoint"),
    }, indent=2))


if __name__ == "__main__":
    main()
