"""Event terms IsaacLab does not ship: inertia DR, trunk payload, tracked push."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from gd_lab.core.payload_math import combine_point_payload

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolved_body_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    if asset_cfg.body_ids == slice(None):
        return list(range(asset.num_bodies))
    return list(asset_cfg.body_ids)


def randomize_body_inertia_diag(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
) -> None:
    """Scale Ixx/Iyy/Izz of the selected bodies by independent per-env factors.

    IsaacLab randomizes mass and COM but not inertia, so without this a trunk
    with perturbed mass keeps a pristine rotational response. Startup only.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    body_ids = torch.tensor(_resolved_body_ids(asset, asset_cfg), dtype=torch.int, device="cpu")

    inertias = asset.root_physx_view.get_inertias()
    inertias[ids[:, None], body_ids] = asset.data.default_inertia[ids[:, None], body_ids].clone()
    lo, hi = scale_range
    scales = torch.empty(len(ids), len(body_ids), 3, device="cpu").uniform_(lo, hi)
    for axis, flat_idx in enumerate((0, 4, 8)):  # diagonal of the flattened 3x3 tensor
        inertias[ids[:, None], body_ids, flat_idx] *= scales[:, :, axis]
    asset.root_physx_view.set_inertias(inertias, ids)


def randomize_payload_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    payload_masses: Sequence[float] = (0.0, 6.0),
    payload_probabilities: Sequence[float] | None = None,
    payload_position_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Per-episode DISCRETE payload on the trunk, exposed to the policy.

    Unlike ``add_base_mass`` this is a task variable, not a perturbation: the
    drawn value is surfaced by the ``payload_mass`` observation so the policy
    conditions on the load instead of averaging over it. Each env draws from
    ``payload_masses`` with ``payload_probabilities`` (uniform when omitted).

    With ``payload_position_range`` the payload is a point mass rigidly mounted
    at a per-episode position in the link frame (m): mass, CoM and inertia are
    combined via the parallel-axis theorem. The position lands in
    ``env.payload_pos_b`` (zero when unloaded) but is not observed. Without a
    range the inertia simply scales with the mass ratio.

    The baseline is cached on the first call, which lands after the startup
    events, so a startup ``add_base_mass`` delta stays layered underneath. Every
    reset is computed from that baseline, so unloading restores it exactly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolved_body_ids(asset, asset_cfg)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    if not hasattr(env, "payload_kg"):
        env.payload_kg = torch.zeros(env.num_envs, device=env.device)
        env.payload_pos_b = torch.zeros(env.num_envs, 3, device=env.device)
        # Post-startup properties (CPU tensors per the PhysX view API).
        env._payload_base_mass = asset.root_physx_view.get_masses().clone()
        env._payload_base_inertia = asset.root_physx_view.get_inertias().clone()
        env._payload_base_com = asset.root_physx_view.get_coms().clone()

    choices = torch.tensor(payload_masses, device=env.device)
    if payload_probabilities is None:
        indices = torch.randint(len(choices), (len(env_ids),), device=env.device)
    else:
        if len(payload_probabilities) != len(payload_masses):
            raise ValueError("payload_probabilities must match the length of payload_masses")
        probs = torch.tensor(payload_probabilities, dtype=torch.float32, device=env.device)
        indices = torch.multinomial(probs, len(env_ids), replacement=True)
    picks = choices[indices]
    env.payload_kg[env_ids] = picks

    ids = env_ids.cpu()
    selected = (ids[:, None], body_ids)
    base_mass = env._payload_base_mass
    payload = picks.cpu().unsqueeze(1)
    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()

    if payload_position_range is None:
        masses[selected] = base_mass[selected] + payload
        ratios = masses[selected] / base_mass[selected]
        inertias[selected] = env._payload_base_inertia[selected] * ratios.unsqueeze(-1)
        asset.root_physx_view.set_masses(masses, ids)
        asset.root_physx_view.set_inertias(inertias, ids)
        return

    ranges = torch.tensor(
        [payload_position_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], dtype=base_mass.dtype
    )
    positions = ranges[:, 0] + torch.rand(len(env_ids), 3) * (ranges[:, 1] - ranges[:, 0])
    positions[payload[:, 0] == 0.0] = 0.0
    env.payload_pos_b[env_ids] = positions.to(env.device)

    base_com = env._payload_base_com
    mass, com, inertia = combine_point_payload(
        base_mass[selected],
        base_com[selected][..., :3],
        env._payload_base_inertia[selected].reshape(len(env_ids), len(body_ids), 3, 3),
        payload,
        positions.unsqueeze(1),
    )
    coms = asset.root_physx_view.get_coms().clone()
    masses[selected] = mass
    coms[selected] = base_com[selected]
    coms[ids[:, None], body_ids, :3] = com
    # The symmetric inertia flattens identically in row- or column-major order.
    inertias[selected] = inertia.flatten(start_dim=-2)
    asset.root_physx_view.set_masses(masses, ids)
    asset.root_physx_view.set_coms(coms, ids)
    # Inertia last: PhysX recomputes its principal-axis orientation from it.
    asset.root_physx_view.set_inertias(inertias, ids)
    # Reset-state writers must see the new CoM within this simulation step.
    com_cache = getattr(asset.data, "_body_com_pose_b", None)
    if com_cache is not None:
        com_cache.timestamp = -1.0


def push_by_setting_velocity_tracked(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Upstream ``push_by_setting_velocity`` that also records the applied delta.

    The push sets root velocity directly - no wrench - so nothing in the scene
    state tells the critic a push happened and it must price one as an
    unexplained velocity jump. The stamp feeds the ``push_delta_v`` observation.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids]
    ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=asset.device,
    )
    delta = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand_like(vel_w)
    asset.write_root_velocity_to_sim(vel_w + delta, env_ids=env_ids)

    if not hasattr(env, "push_delta_v_buf"):
        env.push_delta_v_buf = torch.zeros(env.num_envs, 3, device=env.device)
        env.push_step_buf = torch.full((env.num_envs,), -(10**9), dtype=torch.long, device=env.device)
    # Yaw frame, so the mirror augmentation is a plain y sign flip.
    env.push_delta_v_buf[env_ids] = quat_apply_inverse(yaw_quat(asset.data.root_quat_w[env_ids]), delta[:, :3])
    env.push_step_buf[env_ids] = int(env.common_step_counter)
