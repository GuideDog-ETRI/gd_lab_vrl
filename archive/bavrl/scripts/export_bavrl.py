"""Export a trained BAVRL snapshot for the explicitly gated MuJoCo backend."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

from gd_lab.core.camera_contract import camera_contract_for_policy
from gd_lab.deploy.export_student_vrl import export_student_vrl
from gd_lab.residuals.bavrl import BAVRL, ResidualConfig, load_blind_teacher
from gd_lab.residuals.bavrl.export import BAVRLActorExport, BAVRLVisionExport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-agent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    obs = TensorDict({"policy": torch.zeros(1, 230), "critic": torch.zeros(1, 298)}, batch_size=[1])
    teacher, digest = load_blind_teacher(args.teacher, args.teacher_agent, obs)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BAVRL(teacher, ResidualConfig(**state["config"]))
    model.restore(state, digest)
    model.eval()
    context = state["deploy_context"]
    assert abs(context["policy_dt"]-.01) < 1e-8
    assert np.allclose(context["action"]["scale"], .25)
    assert np.allclose(context["action"]["offset"], [0]*4+[.76]*4+[-1.45]*4)
    assert np.allclose(context["action"]["gains"]["kp"], [123.39]*8+[127.77]*4)
    assert np.allclose(context["action"]["gains"]["kd"], 2.4)
    assert context["action"]["clip"] is None
    assert model.config.stale_seconds == .25
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    actor = BAVRLActorExport(model)
    example = (torch.randn(1, 46), torch.randn(1, 230), torch.zeros(1, 46))
    path = output / "policy_bavrl.onnx"
    names = ["direct_obs", "cenet_obs", "terrain_latent"]
    torch.onnx.export(actor, example, str(path), opset_version=18, input_names=names,
                      output_names=["actions", "z_t", "residual_out"])
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, {"camel.bavrl": "v1_sim_only", "teacher_sha256": digest})
    onnx.save(graph, str(path))
    camera_profile = model.config.camera_profile
    model.vision.verify_camera_geometry()
    export_student_vrl(BAVRLVisionExport(model), str(path), camera_profile=camera_profile)
    student_path = output / "policy_bavrl_student.onnx"
    student_graph = onnx.load(str(student_path))
    onnx.helper.set_model_props(student_graph, {"camel.student_arch": "grid_attention_v1",
        "camel.student_age": "hidden63_seconds_clipped_0_1", "camel.bavrl_vision": "v1",
        "camel.camera_profile": camera_profile})
    onnx.save(student_graph, str(student_path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for valid in (0., 1.):
        for age in (0., .1, .3):
            packed = torch.randn(1, 46)
            packed[:, 32:44] = .1
            packed[:, 44], packed[:, 45] = age, valid
            inputs = (*example[:2], packed)
            with torch.no_grad():
                expected = actor(*inputs)
            actual = session.run(None, {k: v.numpy() for k, v in zip(names, inputs, strict=True)})
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_allclose(a, b.numpy(), atol=2e-5, rtol=2e-5)
            if not valid or age > .25:
                assert np.count_nonzero(actual[2]) == 0
    (output / "manifest.json").write_text(json.dumps({"schema": "bavrl_v1", "simulation_only": True,
        "checkpoint": str(Path(args.checkpoint).resolve()), "iteration": state["iteration"],
        "teacher_sha256": digest, "deploy_context": context, "onnx_parity": "passed",
        "camera_contract": camera_contract_for_policy(camera_profile, context["policy_dt"]).manifest()}, indent=2))
    print("BAVRL exported, actor ONNX parity and missing/stale residual=0 verified:", output)


if __name__ == "__main__":
    main()
