"""Deploy export for the vision-RL stage-3 camera student
(``gd_lab.rl.perception.CameraPerceptionEncoder``): JIT + ONNX with an
explicit GRU hidden-state input/output pair.

ONNX Runtime has no persistent state across ``Run()`` calls, so the deploy
runtime must own the hidden tensor itself and feed it back in on every call
-- exactly like gd_rbq10_deploy's Dream backend already owns the proprio
history buffer for the actor. See that side's camera/student thread for the
consumer of this contract.

Sibling-file convention, not a new config knob: given the teacher/actor's own
export path (e.g. ``.../policy_vrl.onnx``), this writes
``.../policy_vrl_student.onnx`` right next to it. gd_rbq10_deploy's
``DreamVrlBackend`` derives the student path from the actor path it is
already given -- so there is no new "where is the student model" plumbing on
the C++ side, matching this whole deploy set's own philosophy that a file's
shape/location declares its contract (see WalkConfig.hpp's module docstring).
"""

from __future__ import annotations

import copy
import os

import torch

from gd_lab.rl.perception import CameraPerceptionEncoder


def export_student_vrl(
    student: CameraPerceptionEncoder,
    actor_onnx_path: str,
    camera_width: int = 80,
    camera_height: int = 45,
) -> tuple[str, str]:
    """Write ``<actor_stem>_student.pt``/``.onnx`` next to ``actor_onnx_path``.

    ``camera_width``/``camera_height`` must match the belly cameras' own
    ``width``/``height`` (``tasks/vrl_rough.py``'s ``_belly_camera``) --
    there is no way to detect a mismatch from the student checkpoint alone,
    it would just silently see a resized/garbled image.
    """
    student = copy.deepcopy(student).cpu().eval()
    stem, _ = os.path.splitext(actor_onnx_path)
    jit_path = f"{stem}_student.pt"
    onnx_path = f"{stem}_student.onnx"

    example = (
        torch.zeros(1, student.num_cameras, 2, camera_height, camera_width),
        torch.zeros(1, student.gru_hidden_dim),
    )

    traced = torch.jit.trace(student, example)
    traced.save(jit_path)

    # No dynamic_axes: deploy always runs batch=1 (one robot), and a dynamic
    # leading dim here confuses the ONNX exporter's adaptive_avg_pool2d
    # symbolic (it needs statically-known spatial dims, which a dynamic batch
    # combined with the num_cameras-folding reshape in _CameraCNN.forward
    # obscures). Training-side (train_perception.py) never exports this
    # module -- it stays a plain nn.Module there -- so fixing the shape here
    # costs nothing.
    torch.onnx.export(
        student,
        example,
        onnx_path,
        opset_version=18,
        input_names=["frames", "hidden_in"],
        output_names=["terrain_latent", "hidden_out"],
    )
    return jit_path, onnx_path
