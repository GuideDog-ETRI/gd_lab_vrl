"""Export the frozen CVTT height/visibility encoder for MuJoCo teacher tests.

The regular actor export deliberately accepts an external 32-D terrain latent.
This exports the matching encoder from the *same* checkpoint, with a separate
374-D masked-height plus validity input. It does not alter the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gd_lab.teachers.cvtt.actor_critic import HeightScanCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = {
        key.removeprefix("terrain_encoder."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("terrain_encoder.")
    }
    encoder = HeightScanCNN((11, 17), 32).eval()
    encoder.load_state_dict(weights, strict=True)
    # For this fixed 11x17 grid, adaptive 5x8 bins are exactly 3x3 windows
    # with stride 2. The ONNX exporter cannot lower the non-divisible adaptive
    # pool, so use its equivalent static pool after checking parity.
    reference = HeightScanCNN((11, 17), 32).eval()
    reference.load_state_dict(weights, strict=True)
    encoder.net[4] = torch.nn.AvgPool2d(kernel_size=3, stride=2)
    with torch.no_grad():
        probe = torch.randn(16, 374)
        torch.testing.assert_close(encoder(probe), reference(probe), atol=1e-6, rtol=1e-6)
    sample = torch.zeros(1, 374)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder, sample, args.out, opset_version=18,
        input_names=["terrain_scan"], output_names=["terrain_latent"],
        dynamic_axes={"terrain_scan": {0: "batch"}, "terrain_latent": {0: "batch"}},
    )
    metadata = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "input": "masked height[11,17] then visibility[11,17]",
        "output": "terrain latent[32]",
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
