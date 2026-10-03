"""Warm-start a BIVT policy from a deployed blind DreamWaQ ONNX graph.

Only inference-path weights can be recovered from an exported graph. The
critic, CENet decoder, PPO exploration parameter, and terrain encoder are not
present in the ONNX model and intentionally keep their fresh initialization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .blind_init import ACTOR_INPUT_WEIGHT, blind_to_vrl_state_dict

_ONNX_TO_POLICY = {
    "cenet_encoder.0.weight": "cenet.encoder.0.weight",
    "cenet_encoder.0.bias": "cenet.encoder.0.bias",
    "cenet_encoder.2.weight": "cenet.encoder.2.weight",
    "cenet_encoder.2.bias": "cenet.encoder.2.bias",
    "cenet_mean_velocity.weight": "cenet.mean_velocity.weight",
    "cenet_mean_velocity.bias": "cenet.mean_velocity.bias",
    "cenet_mean_latent.weight": "cenet.mean_latent.weight",
    "cenet_mean_latent.bias": "cenet.mean_latent.bias",
}


def _onnx_initializers(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - environment-specific hint
        raise RuntimeError("ONNX warm-start requires the gd-lab 'deploy' extra (pip install -e '.[deploy]').") from exc

    model = onnx.load(str(path), load_external_data=True)
    in_shapes = {
        item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.input
    }
    out_shapes = {
        item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.output
    }
    if in_shapes.get("direct_obs", [None, None])[-1] != 46:
        raise ValueError(f"Expected DreamWaQ direct_obs width 46, got {in_shapes}")
    if in_shapes.get("cenet_obs", [None, None])[-1] != 230:
        raise ValueError(f"Expected DreamWaQ history width 230, got {in_shapes}")
    if out_shapes.get("actions", [None, None])[-1] != 12 or out_shapes.get("z_t", [None, None])[-1] != 19:
        raise ValueError(f"Expected RBQ10 DreamWaQ outputs actions=12, z_t=19, got {out_shapes}")

    weights = {
        value.name: torch.from_numpy(numpy_helper.to_array(value).copy()).float()
        for value in model.graph.initializer
    }
    return weights, {"inputs": in_shapes, "outputs": out_shapes, "opset": [x.version for x in model.opset_import]}


def onnx_to_bivt_state_dict(
    onnx_path: str | Path,
    vrl_state: dict[str, torch.Tensor],
    *,
    normalizer_eps: float = 1.0e-2,
    normalizer_pseudo_count: int = 1_000_000,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Map DWB ONNX inference weights into a freshly constructed BIVT state.

    The source ONNX stores ``std + eps`` as its division initializer, not the
    original running count. ``normalizer_pseudo_count`` is therefore an
    explicitly synthetic count that prevents the first PPO batch from
    replacing the imported normalization statistics abruptly.
    """
    if normalizer_pseudo_count < 0:
        raise ValueError("normalizer_pseudo_count must be non-negative")
    source, graph = _onnx_initializers(onnx_path)
    required = {"actor.0.weight", "normalizer._mean", "onnx::Div_124", *_ONNX_TO_POLICY}
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"ONNX graph is missing required DWB inference weights: {missing}")

    target_actor = vrl_state.get(ACTOR_INPUT_WEIGHT)
    source_actor = source[ACTOR_INPUT_WEIGHT]
    if target_actor is None or target_actor.ndim != 2:
        raise ValueError("Target policy lacks actor.0.weight")
    if source_actor.shape[0] != target_actor.shape[0] or target_actor.shape[1] <= source_actor.shape[1]:
        raise ValueError(
            f"Target actor cannot append terrain latent: source={tuple(source_actor.shape)}, "
            f"target={tuple(target_actor.shape)}"
        )

    blind_state: dict[str, torch.Tensor] = {
        name: tensor for name, tensor in source.items() if name.startswith("actor.")
    }
    for onnx_name, policy_name in _ONNX_TO_POLICY.items():
        blind_state[policy_name] = source[onnx_name]

    merged = blind_to_vrl_state_dict(blind_state, vrl_state)

    # ONNX actor input is [latest direct_obs, z_t]. BIVT actor input is
    # [K newest frames, z_t, terrain latent]. Left-aligned widening silently
    # places z_t weights in frame 2, so explicitly restore the feature layout.
    direct_dim = int(graph["inputs"]["direct_obs"][-1])
    code_dim = int(graph["outputs"]["z_t"][-1])
    terrain_weight = vrl_state.get("terrain_encoder.head.2.weight")
    if terrain_weight is None or terrain_weight.ndim != 2:
        raise ValueError("Target policy lacks terrain_encoder.head.2.weight")
    terrain_dim = int(terrain_weight.shape[0])
    target_actor = vrl_state[ACTOR_INPUT_WEIGHT]
    frame_width = target_actor.shape[1] - code_dim - terrain_dim
    if frame_width <= 0 or frame_width % direct_dim:
        raise ValueError(
            "Cannot infer BIVT actor layout from ONNX and target widths: "
            f"target={target_actor.shape[1]}, direct={direct_dim}, code={code_dim}, terrain={terrain_dim}"
        )
    frame_count = frame_width // direct_dim
    if source_actor.shape[1] != direct_dim + code_dim:
        raise ValueError(
            f"Expected source actor input direct_obs+z_t={direct_dim + code_dim}, "
            f"got {source_actor.shape[1]}"
        )
    repacked = torch.zeros_like(target_actor)
    repacked[:, :direct_dim] = source_actor[:, :direct_dim].to(repacked)
    code_start = direct_dim * frame_count
    repacked[:, code_start : code_start + code_dim] = source_actor[:, direct_dim:].to(repacked)
    merged[ACTOR_INPUT_WEIGHT] = repacked
    mean_key, std_key = "actor_obs_normalizer._mean", "actor_obs_normalizer._std"
    var_key, count_key = "actor_obs_normalizer._var", "actor_obs_normalizer.count"
    for key in (mean_key, std_key, var_key, count_key):
        if key not in vrl_state:
            raise ValueError(f"Target policy lacks required observation-normalizer state {key!r}")

    mean = source["normalizer._mean"]
    denom = source["onnx::Div_124"]
    std = denom - float(normalizer_eps)
    if mean.shape != vrl_state[mean_key].shape or std.shape != vrl_state[std_key].shape:
        raise ValueError(
            "Observation-normalizer shape mismatch: "
            f"ONNX mean/std={tuple(mean.shape)}/{tuple(std.shape)}, "
            f"BIVT={tuple(vrl_state[mean_key].shape)}/{tuple(vrl_state[std_key].shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("ONNX normalization statistics are non-finite or invalid after removing eps")
    merged[mean_key] = mean.to(vrl_state[mean_key])
    merged[std_key] = std.to(vrl_state[std_key])
    merged[var_key] = std.square().to(vrl_state[var_key])
    merged[count_key] = torch.full_like(vrl_state[count_key], normalizer_pseudo_count)

    digest = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
    info = {
        "source_format": "onnx",
        "source_path": str(Path(onnx_path).resolve()),
        "source_sha256": digest,
        "mapping": {**{key: key for key in blind_state if key.startswith("actor.")}, **_ONNX_TO_POLICY},
        "normalizer_eps": float(normalizer_eps),
        "normalizer_pseudo_count": int(normalizer_pseudo_count),
        "freshly_initialized": ["critic", "cenet.decoder", "PPO action std", "terrain_encoder"],
        "graph": graph,
    }
    return merged, info


def load_blind_onnx(policy: torch.nn.Module, path: str, *, normalizer_pseudo_count: int = 1_000_000) -> dict:
    """Load ONNX inference weights into an already constructed BIVT policy."""
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    if normalizer is None or not hasattr(normalizer, "eps"):
        raise ValueError("BIVT target must expose actor_obs_normalizer with its training eps")
    state, info = onnx_to_bivt_state_dict(
        path,
        policy.state_dict(),
        normalizer_eps=float(normalizer.eps),
        normalizer_pseudo_count=normalizer_pseudo_count,
    )
    policy.load_state_dict(state)
    return info
