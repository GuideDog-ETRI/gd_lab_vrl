"""BIVT trainer shim that warm-starts from a deployed blind DreamWaQ ONNX.

The legacy ``train_bivt.py`` and its PyTorch ``--blind_init`` contract remain
unchanged. This wrapper translates ``--blind_onnx_init`` to that hook while
replacing only the checkpoint loader in-process.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gd_lab.teachers.bivt import blind_init  # noqa: E402
from gd_lab.teachers.bivt.onnx_init import load_blind_onnx  # noqa: E402


def main() -> None:
    argv = list(sys.argv[1:])
    if "--blind_onnx_init" not in argv:
        raise SystemExit("Use --blind_onnx_init /path/to/policy.onnx")
    if "--blind_init" in argv:
        raise SystemExit("Choose only --blind_onnx_init; do not also pass --blind_init")
    index = argv.index("--blind_onnx_init")
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise SystemExit("--blind_onnx_init requires an ONNX file path")
    onnx_path = Path(argv[index + 1]).expanduser().resolve()
    if onnx_path.suffix.lower() != ".onnx" or not onnx_path.is_file():
        raise SystemExit(f"Expected an existing .onnx file, got: {onnx_path}")

    original_loader = blind_init.load_blind_checkpoint

    def load_checkpoint(policy, path: str) -> dict:
        if Path(path).resolve() != onnx_path:
            return original_loader(policy, path)
        info = load_blind_onnx(policy, str(onnx_path))
        print(
            f"[INFO] ONNX warm-start source={onnx_path} sha256={info['source_sha256']}; "
            "critic, CENet decoder, PPO action std, and terrain encoder are fresh.",
            flush=True,
        )
        return info

    blind_init.load_blind_checkpoint = load_checkpoint
    sys.argv = [str(ROOT / "scripts" / "train_bivt.py"), *argv[:index], "--blind_init", str(onnx_path), *argv[index + 2 :]]
    runpy.run_path(str(ROOT / "scripts" / "train_bivt.py"), run_name="__main__")


if __name__ == "__main__":
    main()
