"""Event terms IsaacLab does not ship: inertia DR, trunk payload, tracked push."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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
    payload_masses: Sequence[float] = (0.0, 5.0),
) -> None:
    """Per-episode DISCRETE payload on the trunk, exposed to the policy.

    Unlike ``add_base_mass`` this is a task variable, not a perturbation: the
    drawn value is surfaced by the ``payload_mass`` observation so the policy
    conditions on the load instead of averaging over it.

    The baseline is cached on the first call, which lands after the startup
    events, so a startup ``add_base_mass`` delta stays layered underneath.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolved_body_ids(asset, asset_cfg)
    if not hasattr(env, "payload_kg"):
        env.payload_kg = torch.zeros(env.num_envs, device=env.device)
        env._payload_base_mass = asset.root_physx_view.get_masses().clone()
        env._payload_base_inertia = asset.root_physx_view.get_inertias().clone()
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    choices = torch.tensor(payload_masses, device=env.device)
    picks = choices[torch.randint(len(choices), (len(env_ids),), device=env.device)]
    env.payload_kg[env_ids] = picks

    ids = env_ids.cpu()
    base_mass = env._payload_base_mass
    masses = asset.root_physx_view.get_masses().clone()
    masses[ids[:, None], body_ids] = base_mass[ids[:, None], body_ids] + picks.cpu().unsqueeze(1)
    asset.root_physx_view.set_masses(masses, ids)

    ratios = masses[ids[:, None], body_ids] / base_mass[ids[:, None], body_ids]
    inertias = asset.root_physx_view.get_inertias().clone()
    inertias[ids[:, None], body_ids] = env._payload_base_inertia[ids[:, None], body_ids] * ratios.unsqueeze(-1)
    asset.root_physx_view.set_inertias(inertias, ids)


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
