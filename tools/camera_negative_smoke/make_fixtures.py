"""Build resume checkpoints that the camera-contract checks MUST refuse (CPU only, no Isaac).

    PYTHONPATH=src           python tools/camera_negative_smoke/make_fixtures.py main <teacher.pt> <out_dir>
    PYTHONPATH=src           python tools/camera_negative_smoke/make_fixtures.py gast <teacher.pt> <out_dir>

Every fixture passes the checks that run BEFORE the camera check (teacher SHA256, student
architecture, supervision-contract version), so a refusal can only come from the camera contract.
The runs use the legacy calibration (the 17206 teacher was trained with it), so:
  *_missing.pt   no camera_contract                      -> refused unless GD_LAB_ALLOW_MISSING_CAMERA_CONTRACT=1
  *_mismatch.pt  vendor_new contract vs a legacy run     -> refused
  *_match.pt     the legacy contract of the run (control) -> must get PAST the check
"""

import hashlib
import sys
from pathlib import Path

import torch

POLICY_DT = 0.01  # Arm4: physics 0.005 s x decimation 2


def _contracts():
    from gd_lab.core.camera_contract import camera_contract_for_policy

    return {"missing": None,
            "mismatch": camera_contract_for_policy("vendor_new", POLICY_DT).manifest(),
            "match": camera_contract_for_policy("vendor_legacy", POLICY_DT).manifest()}


def _main_fixtures(teacher_sha256):
    """RVLD (cnn_gru) and GAVD (grid_attention_v1) resumes of scripts/distill_student.py."""
    from gd_lab.students.gavd.model import GridAttentionStudent
    from gd_lab.students.rvld.distillation import TerrainAlignmentHead
    from gd_lab.students.rvld.model import CameraPerceptionEncoder

    student, head = CameraPerceptionEncoder(), TerrainAlignmentHead(64)
    gavd = GridAttentionStudent("vendor_legacy")  # legacy model so the matching control can load
    common = {"iteration": 1, "student_alignment_contract": 2, "teacher_sha256": teacher_sha256}
    yield "rvld", {**common, "model": student.state_dict(), "alignment_head": head.state_dict(), "student_config": {},
                   "optimizer": torch.optim.Adam(list(student.parameters()) + list(head.parameters())).state_dict(),
                   "student_arch": "cnn_gru"}
    yield "gavd", {**common, "model": gavd.state_dict(), "student_config": {"camera_profile": "vendor_legacy"},
                   "optimizer": torch.optim.Adam(gavd.parameters()).state_dict(), "student_arch": "grid_attention_v1"}


def _gast_fixtures(teacher_sha256):
    """GAST resumes of scripts/gast/train_student.py (and train_student_live.py)."""
    from gd_lab.gast.student import GastStudent

    model = GastStudent("vendor_legacy")
    yield "gast", {"model": model.state_dict(), "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
                   "iteration": 1, "student_arch": "gast_spatiotemporal_v1", "gast_distillation_contract": 2,
                   "teacher_sha256": teacher_sha256}


def main(kind, teacher_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    teacher_sha256 = hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
    builder = {"main": _main_fixtures, "gast": _gast_fixtures}[kind]
    written = []
    for name, base in builder(teacher_sha256):
        for case, contract in _contracts().items():
            ckpt = dict(base)
            if contract is not None:
                ckpt["camera_contract"] = contract
            path = out / f"{name}_{case}.pt"
            torch.save(ckpt, path)
            written.append(path.name)
    print(f"teacher_sha256={teacher_sha256}")
    print("wrote:", " ".join(written))


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("main", "gast"):
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
