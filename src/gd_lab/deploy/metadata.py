"""Deployment contract carried alongside the exported policy.

An ONNX graph says only how many floats go in and come out - not which joint is
column 3, what unit ``joint_vel`` is in, or what PD gains the actions were
trained against. A deployment that guesses any of those wrong produces a robot
that moves confidently and incorrectly, so the semantics are captured from the
resolved environment at training time and written next to the graph.

JSON only, no simulator imports, so export reads it without IsaacLab.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

METADATA_KEY = "gd_lab.policy.v1"
SCHEMA_VERSION = 1

# An unmapped term means the deploy host has no defined way to fill it.
_OBS_SOURCES = {
    "base_ang_vel": "angular_velocity",
    "projected_gravity": "gravity",
    "generated_commands": "command",
    "joint_pos_rel": "joint_position_rel",
    "joint_vel_rel": "joint_velocity",
    "last_action": "previous_action",
    "payload_mass": "payload",
}

# Software deployment guardrails, not certified actuator limits.
_GAIN_LIMITS = {"kp": 400.0, "kd": 20.0}


def _numbers(value, size: int) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    values = [float(v) for v in value]
    if len(values) == 1:
        values *= size
    if len(values) != size or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Expected {size} finite values, got {values}")
    return values


def _validate_gains(gains: dict[str, list[float]], num_joints: int) -> None:
    for name, upper in _GAIN_LIMITS.items():
        values = gains.get(name)
        if not isinstance(values, list) or len(values) != num_joints:
            raise ValueError(f"{name} requires {num_joints} joint gains")
        if any(not math.isfinite(v) or not 0.0 <= v <= upper for v in values):
            raise ValueError(f"{name} outside the deployment range [0, {upper}]: {values}")


def _actuator_gains(robot, joint_names: list[str]) -> dict[str, list[float]]:
    """Per-joint (kp, kd) in ACTION joint order, resolved from the actuator cfgs."""
    gains: dict[str, list[float]] = {"kp": [], "kd": []}
    for name in joint_names:
        matches = [a for a in robot.cfg.actuators.values() if any(re.fullmatch(p, name) for p in a.joint_names_expr)]
        if len(matches) != 1:
            raise ValueError(f"Joint '{name}' resolves to {len(matches)} actuator groups, expected 1")
        for key, attr in (("kp", "stiffness"), ("kd", "damping")):
            value = getattr(matches[0], attr)
            if isinstance(value, dict):
                selected = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
                if len(selected) != 1:
                    raise ValueError(f"Ambiguous {attr} for joint '{name}'")
                value = selected[0]
            gains[key].append(float(value))
    return gains


def _observation_terms(env, group: str) -> list[dict]:
    manager = env.observation_manager
    if not manager.group_obs_concatenate[group]:
        raise ValueError(f"Observation group '{group}' is not concatenated")
    terms = []
    for name, dim, cfg in zip(
        manager.active_terms[group],
        manager.group_obs_term_dim[group],
        manager._group_obs_term_cfgs[group],
        strict=True,
    ):
        if getattr(cfg, "modifiers", None):
            raise ValueError(f"Observation modifiers need an explicit deploy adapter: {name}")
        if cfg.history_length and not cfg.flatten_history_dim:
            raise ValueError(f"Unflattened observation history is unsupported: {name}")
        history = max(int(cfg.history_length or 0), 1)
        flat = math.prod(dim)
        if flat % history:
            raise ValueError(f"Invalid history shape for '{name}': {dim} with history {history}")
        size = int(flat // history)  # dims may be numpy ints
        func = cfg.func.__name__
        if func not in _OBS_SOURCES:
            raise ValueError(f"Unsupported observation function for deployment: {func}")
        # Ordered as the observation manager applies them.
        transforms = []
        if cfg.clip is not None:
            transforms.append({"op": "clip", "min": float(cfg.clip[0]), "max": float(cfg.clip[1])})
        if cfg.scale is not None:
            transforms.append({"op": "scale", "values": _numbers(cfg.scale, size)})
        terms.append(
            {
                "name": name,
                "source": _OBS_SOURCES[func],
                "size": size,
                "history": history,
                "transforms": transforms,
            }
        )
    return terms


def capture_context(env, policy) -> dict:
    """Snapshot the resolved deployment semantics of a live environment.

    Raises on anything the contract cannot express: a silently wrong metadata
    block is worse than none.
    """
    env = env.unwrapped
    robot = env.scene["robot"]

    action_terms = env.action_manager._terms
    if len(action_terms) != 1:
        raise ValueError(f"Deployment requires exactly one action term, found {len(action_terms)}")
    action = next(iter(action_terms.values()))
    if getattr(action.cfg, "clip", None) is not None:
        raise ValueError("Per-joint action-manager clipping needs an explicit deploy adapter")
    joint_names = list(action._joint_names)
    num_joints = len(joint_names)
    joint_ids = [robot.joint_names.index(n) for n in joint_names]
    if any(abs(v) > 1e-8 for v in robot.data.default_joint_vel[0, joint_ids].tolist()):
        raise ValueError("Nonzero default joint velocity is not supported")

    gains = _actuator_gains(robot, joint_names)
    _validate_gains(gains, num_joints)

    policy_groups = policy.obs_groups["policy"]
    if len(policy_groups) != 1:
        raise ValueError(f"Expected exactly one policy observation group, got {policy_groups}")

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_dt": float(env.step_dt),
        "joint_names": joint_names,
        "default_joint_pos": _numbers(robot.data.default_joint_pos[0, joint_ids], num_joints),
        "action": {
            "type": "joint_position",
            "scale": _numbers(action._scale, num_joints),
            "offset": _numbers(action._offset, num_joints),
            "previous_action": "raw",
            "soft_margin_deg": float(getattr(action.cfg, "soft_margin_deg", 0.0)),
            "gains": gains,
        },
        "observation": {
            "layout": "term_major",
            "history_initialization": "repeat_first",
            "terms": _observation_terms(env, policy_groups[0]),
        },
        "actor_history_steps": int(getattr(policy, "actor_history_steps", 1)),
        "provenance": {"policy_class": type(policy).__name__, "source": "resolved_environment"},
    }


def bind_graph(context: dict, model) -> dict:
    """Bind a captured context to an actual ONNX graph, checking they agree."""
    import onnx

    spec = json.loads(json.dumps(context))  # deep copy, and proves serializability
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported deployment context version {spec.get('schema_version')}")
    _validate_gains(spec["action"]["gains"], len(spec["joint_names"]))

    terms = spec["observation"]["terms"]
    expected = sum(t["size"] * t["history"] for t in terms)
    spec["inputs"], spec["outputs"] = [], []
    for value in model.graph.input:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Only float32 inputs are supported: {value.name}")
        shape = [int(d.dim_value) if d.HasField("dim_value") else 1 for d in value.type.tensor_type.shape.dim]
        if shape != [1, expected]:
            raise ValueError(f"Graph/context mismatch: {value.name} is {shape}, context implies [1, {expected}]")
        spec["inputs"].append(
            {
                "name": value.name,
                "shape": shape,
                "dtype": "float32",
                "role": "observation",
                "layout": "term_major",
                "terms": [t["name"] for t in terms],
            }
        )
    if len(spec["inputs"]) != 1:
        raise ValueError(f"Expected one graph input, found {len(spec['inputs'])}")

    names = set()
    for value in model.graph.output:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Only float32 outputs are supported: {value.name}")
        shape = [int(d.dim_value) if d.HasField("dim_value") else -1 for d in value.type.tensor_type.shape.dim]
        if shape:
            shape[0] = 1
        names.add(value.name)
        spec["outputs"].append(
            {
                "name": value.name,
                "shape": shape,
                "dtype": "float32",
                "role": "action" if value.name == "actions" else "auxiliary",
            }
        )
    if "actions" not in names:
        raise ValueError("Graph has no 'actions' output")
    spec["normalization"] = "in_graph"
    return spec


def attach_metadata(onnx_path: str, context: dict) -> dict:
    """Write the bound spec into the ONNX metadata and a sidecar ``.deploy.json``."""
    import onnx

    path = Path(onnx_path)
    model = onnx.load(path)
    spec = bind_graph(context, model)
    props = {p.key: p.value for p in model.metadata_props}
    props[METADATA_KEY] = json.dumps(spec, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    onnx.helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    temporary = path.with_suffix(".metadata.tmp")
    try:
        onnx.save(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    path.with_suffix(".deploy.json").write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    return spec
