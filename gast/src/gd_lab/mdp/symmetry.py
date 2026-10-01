"""Left-right mirror augmentation built from an observation spec.

``make_augmentation(spec_set)`` returns a callable for
``RslRlSymmetryCfg.data_augmentation_func``. Robot-dependent permutations (joint
order, scanner grid) are resolved at runtime from the articulation and sensor
configs; per-term mirror rules come from the spec's mirror tags. Every transform
is an involution, so applying the mirror twice returns the original.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from gd_lab.core.types import ObsSpec, ObsSpecSet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Sign triples under reflection through the XZ (left-right mirror) plane.
_SIGN_ANG_VEL = (-1.0, 1.0, -1.0)
_SIGN_LIN = (1.0, -1.0, 1.0)
_SIGN_COMMANDS = (1.0, -1.0, -1.0)

# 2x2 GridPattern flat order is y-outer, x-inner: a y flip swaps the rows.
_PERM_2X2_YFLIP = (2, 3, 0, 1)
# Foot blocks ordered (FL, FR, RL, RR): mirror swaps FL<->FR and RL<->RR.
_PERM_FEET = (1, 0, 3, 2)


def _mirror_joint_name(name: str) -> str:
    side = name[1]
    if side not in ("L", "R"):
        raise ValueError(f"Cannot infer side (L/R) of joint '{name}'")
    return name[0] + ("R" if side == "L" else "L") + name[2:]


def _build_joint_perm_sign(joint_names: list[str]) -> tuple[list[int], list[float]]:
    """L<->R permutation with a HIP sign flip, from the runtime joint order."""
    index = {n: i for i, n in enumerate(joint_names)}
    perm, sign = [], []
    for name in joint_names:
        partner = _mirror_joint_name(name)
        if partner not in index:
            raise ValueError(f"Mirror partner '{partner}' of joint '{name}' not found in {joint_names}")
        perm.append(index[partner])
        sign.append(-1.0 if name.endswith("_HIP") else 1.0)
    return perm, sign


def _build_scan_perm(nx: int, ny: int) -> list[int]:
    """Lateral (y) flip of a GridPattern scan (y-outer, x-inner flat order)."""
    return [(ny - 1 - iy) * nx + ix for iy in range(ny) for ix in range(nx)]


def _build_feet16_perm() -> list[int]:
    return [_PERM_FEET[f] * 4 + _PERM_2X2_YFLIP[r] for f in range(4) for r in range(4)]


def _scanner_grid_dims(env: ManagerBasedRLEnv, scanner_name: str) -> tuple[int, int]:
    pcfg = getattr(env.cfg.scene, scanner_name).pattern_cfg
    nx = round(pcfg.size[0] / pcfg.resolution) + 1
    ny = round(pcfg.size[1] / pcfg.resolution) + 1
    return nx, ny


def resolve_mirror_tag(
    tag: str,
    dim: int,
    joint_ps: tuple[list[int], list[float]],
    scan_perm: list[int],
) -> tuple[list[int] | None, list[float] | None]:
    """(perm, sign) over one history step for a mirror tag; ``None`` means identity."""
    joint_perm, joint_sign = joint_ps
    num_joints = len(joint_perm)
    if tag == "ang_vel":
        return None, list(_SIGN_ANG_VEL)
    if tag == "lin_vec":
        return None, list(_SIGN_LIN)
    if tag == "commands":
        return None, list(_SIGN_COMMANDS)
    if tag == "joint":
        return list(joint_perm), list(joint_sign)
    if tag == "scan":
        return list(scan_perm), None
    if tag == "feet4":
        return list(_PERM_FEET), None
    if tag == "feet16":
        return _build_feet16_perm(), None
    if tag == "feet_vec3":
        return [_PERM_FEET[f] * 3 + a for f in range(4) for a in range(3)], list(_SIGN_LIN) * 4
    if tag == "gain24":
        return list(joint_perm) + [num_joints + p for p in joint_perm], None
    if tag == "identity":
        return list(range(dim)), None
    raise ValueError(f"Unknown mirror tag '{tag}'")


def build_group_perm_sign(
    spec: ObsSpec,
    joint_ps: tuple[list[int], list[float]],
    scan_perm: list[int],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a group's per-term mirror rules into one flat (perm, sign) pair."""
    full_perm: list[int] = []
    full_sign: list[float] = []
    cursor = 0
    for term in spec.terms:
        if term.dim is None:
            raise ValueError(f"Obs term '{term.name}' has unresolved dim; resolve the spec first")
        perm, sign = resolve_mirror_tag(term.mirror, term.dim, joint_ps, scan_perm)
        if perm is not None and len(perm) != term.dim:
            raise ValueError(
                f"Mirror tag '{term.mirror}' of term '{term.name}' yields a {len(perm)}-dim "
                f"permutation but the term is {term.dim}-dim"
            )
        if sign is not None and len(sign) != term.dim:
            raise ValueError(
                f"Mirror tag '{term.mirror}' of term '{term.name}' yields a {len(sign)}-dim "
                f"sign vector but the term is {term.dim}-dim"
            )
        # History flattens oldest -> newest, ``dim`` entries per slot.
        for t in range(spec.history):
            base = cursor + t * term.dim
            full_perm.extend(base + p for p in (perm if perm is not None else range(term.dim)))
            full_sign.extend(sign if sign is not None else [1.0] * term.dim)
        cursor += term.dim * spec.history
    return (
        torch.tensor(full_perm, dtype=torch.long, device=device),
        torch.tensor(full_sign, dtype=torch.float, device=device),
    )


class _MirrorSpec:
    def __init__(self, group_perm_sign, action_perm, action_sign):
        self.group_perm_sign = group_perm_sign
        self.action_perm = action_perm
        self.action_sign = action_sign


def _build_mirror_spec(env: ManagerBasedRLEnv, spec_set: ObsSpecSet) -> _MirrorSpec:
    device = env.device
    joint_names = list(env.scene["robot"].data.joint_names)
    joint_ps = _build_joint_perm_sign(joint_names)
    nx, ny = _scanner_grid_dims(env, "height_scanner")
    scan_perm = _build_scan_perm(nx, ny)

    om = env.observation_manager
    group_perm_sign = {}
    for group, spec in spec_set.groups.items():
        spec = spec.resolve(**{t.name: nx * ny for t in spec.terms if t.dim is None})
        env_terms = list(om.active_terms[group])
        spec_terms = [t.name for t in spec.terms]
        if env_terms != spec_terms:
            raise ValueError(
                f"Obs group '{group}' terms {env_terms} do not match the spec {spec_terms}; "
                "the spec is the single source of truth - update it and everything derived"
            )
        perm, sign = build_group_perm_sign(spec, joint_ps, scan_perm, device)
        group_dim = om.group_obs_dim[group][0]
        if group_dim != perm.shape[0]:
            raise ValueError(f"Obs group '{group}' dim {group_dim} != spec dim {perm.shape[0]}")
        group_perm_sign[group] = (perm, sign)

    return _MirrorSpec(
        group_perm_sign,
        torch.tensor(joint_ps[0], dtype=torch.long, device=device),
        torch.tensor(joint_ps[1], dtype=torch.float, device=device),
    )


def make_augmentation(spec_set: ObsSpecSet) -> Callable:
    """Build the (obs, actions, env) -> (obs_aug, actions_aug) mirror-augmentation callable."""
    cache: dict[int, _MirrorSpec] = {}

    @torch.no_grad()
    def compute_symmetric_states(
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
        env: ManagerBasedRLEnv = None,
    ):
        unwrapped = env.unwrapped
        mirror = cache.get(id(unwrapped))
        if mirror is None:
            mirror = _build_mirror_spec(unwrapped, spec_set)
            cache[id(unwrapped)] = mirror

        obs_aug = None
        if obs is not None:
            mirrored = obs.clone()
            for group in obs.keys():
                if group not in mirror.group_perm_sign:
                    raise ValueError(f"Obs group '{group}' has no mirror spec")
                perm, sign = mirror.group_perm_sign[group]
                g = obs[group]
                if g.shape[-1] != perm.shape[0]:
                    raise ValueError(f"Obs group '{group}' dim {g.shape[-1]} != mirror spec dim {perm.shape[0]}")
                mirrored[group] = g[..., perm] * sign
            obs_aug = torch.cat([obs, mirrored], dim=0)

        actions_aug = None
        if actions is not None:
            mirrored_act = actions[..., mirror.action_perm] * mirror.action_sign
            actions_aug = torch.cat([actions, mirrored_act], dim=0)

        return obs_aug, actions_aug

    return compute_symmetric_states
